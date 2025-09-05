import logging
import os

import geopandas
from osgeo import gdal, ogr
import natcap.invest.utils
from natcap.invest import datastack
from natcap.invest import spec
from natcap.invest import validation
import numpy
import pandas
import pygeoprocessing

from invest_sdr_scenario_compare import report_jinja


SCENARIO_COL_NAME = 'scenario'
FID_COL_NAME = 'watershed_id'

LOGGER = logging.getLogger(__name__)


MODEL_SPEC = spec.ModelSpec(
    model_id="sdr_compare_scenarios",
    model_title="SDR Compare Scenarios",
    userguide='',
    input_field_order=[
        ['workspace_dir'],
        ['scenarios']],
    inputs=[
        spec.DirectoryInput(
            id="workspace_dir",
            name="workspace",
            about=(
                "The folder where all the model's output files will be written. If "
                "this folder does not exist, it will be created. If data already "
                "exists in the folder, it will be overwritten."),
            contents=[],
            must_exist=False,
            permissions="rwx"
        ),
        spec.CSVInput(
            id="scenarios",
            name="Completed Scenarios",
            about=(
                "Make a CSV with two columns: 'scenarios' and 'logfiles'.\n"
                " Each row represents a scenario. The 'scenarios' column should"
                " be a simple label to describe that scenario, for example,"
                " 'baseline'. The 'logfiles' column should be the path to the"
                " InVEST logfile (.txt file) that was generated when the"
                " SDR model was run. Include as many rows as desired."
                " The first row will be treated as the baseline scenario to"
                " which all other scenarios are compared."),
            columns=[
                spec.StringInput(id='scenarios'),
                spec.FileInput(id='logfiles')]
        ),
    ],
    outputs=[
        spec.CSVOutput(
            id="watershed_results.csv",
            path='watershed_results.csv',
            about=(
                """
                The SDR model aggregated raster results by calculating the sum
                total of all pixels within each watershed polygon. This table
                includes those totals for each variable, for each scenario.
                It also includes the percent change relative to the baseline
                scenario.
                """),
        )
    ]
)


def _rename_layer_from_parent(layer):
    """Rename a GDAL vector layer to match the dataset filename."""
    lyrname = os.path.splitext(
        os.path.basename(layer._parent_ds().GetName()))[0]
    layer.Rename(lyrname)


def difference_vectors(vector_path_a, vector_path_b, field_list, target_vector_path):
    vector_a = gdal.OpenEx(vector_path_a, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    vector_b = gdal.OpenEx(vector_path_b, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('GPKG')
    if os.path.exists(target_vector_path):
        driver.Delete(target_vector_path)
    target_vector = driver.CreateCopy(target_vector_path, vector_a)

    target_layer = target_vector.GetLayer()
    _rename_layer_from_parent(target_layer)

    for field in target_layer.schema:
        if field.name in field_list:
            target_layer.DeleteField(
                target_layer.FindFieldIndex(field.name, 1))

    def _create_field(fieldname):
        field = ogr.FieldDefn(str(fieldname), ogr.OFTReal)
        target_layer.CreateField(field)

    new_field_list = []
    for fieldname in field_list:
        new_fieldname = f'diff_{fieldname}'
        new_field_list.append(new_fieldname)
        _create_field(new_fieldname)

    layer_a = vector_a.GetLayer()
    layer_b = vector_b.GetLayer()
    for feature_a, feature_b, target_feature in zip(layer_a, layer_b, target_layer):
        for field, newfield in zip(field_list, new_field_list):
            value_a = feature_a.GetField(field)
            value_b = feature_b.GetField(field)
            diff_value = value_b - value_a  # scenario - baseline
            LOGGER.info(f'Field: {newfield}; Value: {diff_value}')
            target_feature.SetField(newfield, diff_value)
            target_layer.SetFeature(target_feature)
    LOGGER.info(f'created {target_vector_path}')
    layer_a = layer_b = target_layer = None
    vector_a = vector_b = target_vector = None


def difference_rasters(base_raster_path_list,
                       scenario_raster_path_list, target_path_list):
    for (a, b, target) in zip(base_raster_path_list,
                              scenario_raster_path_list,
                              target_path_list):
        LOGGER.info(f'raster differencing for {target}')
        # b - a == scenario - baseline
        pygeoprocessing.raster_map(
            numpy.subtract, [b, a], target_path=target)


def execute(args):
    scenarios_df = natcap.invest.utils.read_csv_to_dataframe(args['scenarios'])
    scenario_logfiles = {
        k: v for k, v in zip(list(scenarios_df.scenarios), list(scenarios_df.logfiles))}
    base_scenario = list(scenario_logfiles)[0]
    LOGGER.info(scenarios_df)
    LOGGER.info(f'Baseline scenario: {base_scenario}')

    workspace = args['workspace_dir']
    if not os.path.exists(workspace):
        os.mkdir(workspace)

    def get_args_dicts(logs_dict):
        scen_args_dict = {}
        for scenario, logfile_path in logs_dict.items():
            _, ds_info = datastack.get_datastack_info(logfile_path)
            scen_args_dict[scenario] = ds_info.args
        return scen_args_dict

    scenario_args_dict = get_args_dicts(scenario_logfiles)
    baseline_args_dict = scenario_args_dict['baseline']
    baseline_workspace = baseline_args_dict['workspace_dir']
    baseline_suffix_str = natcap.invest.utils.make_suffix_string(
        baseline_args_dict, 'results_suffix')

    raster_name_list = [
        'avoided_erosion{0}.tif',
        'avoided_export{0}.tif',
        'sed_deposition{0}.tif',
        'sed_export{0}.tif',
        'rkls{0}.tif',
        'usle{0}.tif']
    baseline_raster_path_list = [
        os.path.join(baseline_workspace, _raster_name.format(baseline_suffix_str))
        for _raster_name in raster_name_list]

    field_list = ["usle_tot", "sed_export", "sed_dep", "avoid_exp", "avoid_eros"]
    results_df = pandas.DataFrame(columns=[SCENARIO_COL_NAME, FID_COL_NAME] + field_list)
    for scenario, args_dict in scenario_args_dict.items():
        scenario_workspace = args_dict['workspace_dir']
        scenario_suffix_str = natcap.invest.utils.make_suffix_string(
            args_dict, 'results_suffix')
        scenario_vector_path = os.path.join(
            scenario_workspace, f'watershed_results_sdr{scenario_suffix_str}.shp')
        ws_vector = geopandas.read_file(scenario_vector_path)
        df = ws_vector[field_list]
        df.insert(0, SCENARIO_COL_NAME, [scenario])
        df.insert(0, FID_COL_NAME, df.index)
        results_df = pandas.concat([results_df, df])

        if scenario == base_scenario:
            continue

        LOGGER.info(f'differencing for scenario {scenario}')

        target_workspace = os.path.join(workspace, scenario)
        if not os.path.exists(target_workspace):
            os.mkdir(target_workspace)

        scenario_raster_path_list = [
            os.path.join(scenario_workspace,
                         raster_name.format(scenario_suffix_str))
            for raster_name in raster_name_list]
        target_raster_path_list = [
            os.path.join(target_workspace,
                         f'diff_{raster_name}'.format(f'_{scenario}'))
            for raster_name in raster_name_list]
        difference_rasters(
            baseline_raster_path_list,
            scenario_raster_path_list,
            target_raster_path_list)

    target_watersheds_table_path = os.path.join(workspace, 'watershed_results.csv')
    long_df = pandas.melt(results_df, id_vars=[FID_COL_NAME, SCENARIO_COL_NAME])
    long_df.to_csv(target_watersheds_table_path, index=False)

    report_jinja.report(args, MODEL_SPEC)


@validation.invest_validator
def validate(args):
    return validation.validate(args, MODEL_SPEC)


if __name__ == '__main__':

    csv_path = 'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/scenarios.csv'
    pandas.DataFrame({
        'scenarios': ['baseline', 'alternative', 'classical'],
        'logfiles': [
            'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/sdr_example/InVEST-sdr-log-2025-07-21--14_04_29.txt',
            'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/sdr_example_scenario/InVEST-sdr-log-2025-08-05--15_26_49.txt',
            'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/sdr_example_scenario2/InVEST-sdr-log-2025-08-06--15_50_25.txt']
        }).to_csv(csv_path, index=False)
    workspace = 'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/scenario_compare2'
    args = {
        'scenarios': csv_path,
        'workspace_dir': workspace
    }
    model_id = 'sdr_compare_scenarios'
    with natcap.invest.utils.prepare_workspace(
            args['workspace_dir'],
            model_id=model_id,
            logging_level=logging.INFO):
        LOGGER.log(
            datastack.ARGS_LOG_LEVEL,
            'Starting model with parameters: \n%s',
            datastack.format_args_dict(args, model_id))

        execute(args)
