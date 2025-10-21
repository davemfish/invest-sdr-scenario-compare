from collections import namedtuple
import json
import logging
import os

import geopandas
from osgeo import gdal, ogr
import natcap.invest.utils
from natcap.invest import datastack
from natcap.invest import spec
from natcap.invest import validation
from natcap.invest.sdr import sdr
import numpy
import pandas
import pygeoprocessing

from invest_sdr_scenario_compare import report_jinja


SCENARIO_COL_NAME = 'scenario'
FID_COL_NAME = 'watershed_id'

LOGGER = logging.getLogger(__name__)

# I'd rather not support use of a results suffix in this plugin,
# but invest requires it be in the MODEL_SPEC
RESULTS_SUFFIX = spec.SUFFIX
RESULTS_SUFFIX.hidden = True

MODEL_SPEC = spec.ModelSpec(
    module_name=__name__,
    model_id="sdr_compare_scenarios",
    model_title="SDR Compare Scenarios",
    userguide='',
    input_field_order=[
        ['workspace_dir', 'scenarios']],
    inputs=[
        spec.WORKSPACE,
        RESULTS_SUFFIX,
        spec.N_WORKERS,
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
        spec.TASKGRAPH_CACHE,
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
        ),
        spec.SingleBandRasterOutput(
            id="diff_avoided_erosion_[SCENARIO]",
            path="[SCENARIO]/diff_avoided_erosion_[SCENARIO].tif",
            about="Difference in avoided erosion (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('avoided_erosion').data_type,
            units=sdr.MODEL_SPEC.get_output('avoided_erosion').units
        ),
        spec.SingleBandRasterOutput(
            id="diff_avoided_export_[SCENARIO]",
            path="[SCENARIO]/diff_avoided_export_[SCENARIO].tif",
            about="Difference in avoided export (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('avoided_export').data_type,
            units=sdr.MODEL_SPEC.get_output('avoided_export').units
        ),
        spec.SingleBandRasterOutput(
            id="diff_rkls_[SCENARIO]",
            path="[SCENARIO]/diff_rkls_[SCENARIO].tif",
            about="Difference in RKLS (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('rkls').data_type,
            units=sdr.MODEL_SPEC.get_output('rkls').units
        ),
        spec.SingleBandRasterOutput(
            id="diff_sed_deposition_[SCENARIO]",
            path="[SCENARIO]/diff_sed_deposition_[SCENARIO].tif",
            about="Difference in sediment deposition (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('sed_deposition').data_type,
            units=sdr.MODEL_SPEC.get_output('sed_deposition').units
        ),
        spec.SingleBandRasterOutput(
            id="diff_sed_export_[SCENARIO]",
            path="[SCENARIO]/diff_sed_export_[SCENARIO].tif",
            about="Difference in sediment export (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('sed_export').data_type,
            units=sdr.MODEL_SPEC.get_output('sed_export').units
        ),
        spec.SingleBandRasterOutput(
            id="diff_usle_[SCENARIO]",
            path="[SCENARIO]/diff_usle_[SCENARIO].tif",
            about="Difference in USLE (scenario - baseline).",
            data_type=sdr.MODEL_SPEC.get_output('usle').data_type,
            units=sdr.MODEL_SPEC.get_output('usle').units
        ),
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


def validate_comparable_scenarios(vector_path_list):
    """Ensure that all vectors have the same features and geometry.

    Raises:
        ValueError if they do not.
    """
    base_vector = gdal.OpenEx(vector_path_list[0], gdal.OF_VECTOR | gdal.GA_ReadOnly)
    base_layer = base_vector.GetLayer()
    base_srs_wkt = base_layer.GetSpatialRef().ExportToWkt()
    base_extent = base_layer.GetExtent()
    base_n_features = base_layer.GetFeatureCount()

    try:
        for vector_path in vector_path_list[1:]:
            vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
            layer = vector.GetLayer()
            srs_wkt = layer.GetSpatialRef().ExportToWkt()
            extent = layer.GetExtent()
            n_features = layer.GetFeatureCount()
            if srs_wkt != base_srs_wkt:
                raise ValueError(
                    f'Projection mismatch between {vector_path_list[0]} and'
                    f' {vector_path}')
            if n_features != base_n_features:
                raise ValueError(
                    f'Feature count mismatch between {vector_path_list[0]} and'
                    f' {vector_path}')
            if extent != base_extent:
                raise ValueError(
                    f'Extent mismatch between {vector_path_list[0]} and'
                    f' {vector_path}')
            for feature_a, feature_b in zip(base_layer, layer):
                geom_a = feature_a.GetGeometryRef()
                geom_b = feature_b.GetGeometryRef()
                if not geom_a.Equals(geom_b):
                    raise ValueError(
                        f'Geometry mismatch between {vector_path_list[0]} and'
                        f' {vector_path} for feature id {feature_a.GetFID()}')
    except ValueError as e:
        LOGGER.error(
            'Scenarios cannot be directly compared because they represent'
            ' different geographies.')
        raise e
    finally:
        layer = None
        vector = None
        base_layer = None
        base_vector = None


def execute(args):
    args, file_registry, task_graph = MODEL_SPEC.setup(args)
    scenarios_df = natcap.invest.utils.read_csv_to_dataframe(args['scenarios'])
    scenario_logfiles_map = {
        k: v for k, v in zip(
            list(scenarios_df.scenarios), list(scenarios_df.logfiles))}
    base_scenario_name = list(scenarios_df.scenarios)[0]
    LOGGER.info(scenarios_df)
    LOGGER.info(f'Baseline scenario: {base_scenario_name}')

    # workspace = args['workspace_dir']
    # if not os.path.exists(workspace):
    #     os.mkdir(workspace)

    WorkspaceRegistry = namedtuple(
        'WorkspaceRegistry', ['workspace', 'file_registry'])
    workspace_registries = {}
    for scenario, logfile_path in scenario_logfiles_map.items():
        _, ds_info = datastack.get_datastack_info(logfile_path)
        args_dict = ds_info.args
        ws = args_dict['workspace_dir']
        suffix = args_dict['results_suffix']
        with open(os.path.join(ws, f'file_registry{args["results_suffix"]}.json'), 'r') as file:
            scenario_registry = json.loads(file.read())
        # suffix = natcap.invest.utils.make_suffix_string(
        #     args_dict, 'results_suffix')
        # vector_path = file_registry['watershed_results_sdr']
        workspace_registries[scenario] = WorkspaceRegistry(ws, scenario_registry)
    
    watershed_results_id = 'watershed_results_sdr'
    validate_comparable_scenarios(
        [x.file_registry[watershed_results_id] for x in workspace_registries.values()])

    raster_id_list = [
        'avoided_erosion',
        'avoided_export',
        'sed_deposition',
        'sed_export',
        'rkls',
        'usle']
    baseline_raster_path_list = [
        workspace_registries[base_scenario_name].file_registry[_raster_id]
        for _raster_id in raster_id_list]

    field_list = ["usle_tot", "sed_export", "sed_dep", "avoid_exp", "avoid_eros"]
    results_df = pandas.DataFrame(
        columns=[SCENARIO_COL_NAME, FID_COL_NAME] + field_list)
    for scenario in workspace_registries:
        ws_vector = geopandas.read_file(
            workspace_registries[scenario].file_registry[watershed_results_id])
        df = ws_vector[field_list]
        df.insert(0, SCENARIO_COL_NAME, [scenario])
        df.insert(0, FID_COL_NAME, df.index)
        results_df = pandas.concat([results_df, df])

        if scenario == base_scenario_name:
            continue

        LOGGER.info(f'differencing for scenario {scenario}')

        target_workspace = os.path.join(args['workspace_dir'], scenario)
        if not os.path.exists(target_workspace):
            os.mkdir(target_workspace)

        scenario_raster_path_list = [
            workspace_registries[scenario].file_registry[_raster_id]
            for _raster_id in raster_id_list]
        target_raster_path_list = [
            file_registry[(f'diff_{_raster_id}_[SCENARIO]', scenario)]
            # os.path.join(target_workspace,
            #              f'diff_{_raster_id}_{scenario}.tif')
            for _raster_id in raster_id_list]
        difference_rasters(
            baseline_raster_path_list,
            scenario_raster_path_list,
            target_raster_path_list)

    # target_watersheds_table_path = os.path.join(
    #     workspace, 'watershed_results.csv')
    long_df = pandas.melt(
        results_df, id_vars=[FID_COL_NAME, SCENARIO_COL_NAME])
    long_df.to_csv(file_registry['watershed_results.csv'], index=False)

    report_jinja.report(args, MODEL_SPEC, file_registry, workspace_registries)
    return file_registry.registry


@validation.invest_validator
def validate(args, limit_to=None):
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
