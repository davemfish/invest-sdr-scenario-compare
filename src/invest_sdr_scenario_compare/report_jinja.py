import os

from invest_reports import utils
import pandas
import taskgraph

from jinja2 import Environment, PackageLoader


SCENARIO_COL_NAME = 'scenario'
FID_COL_NAME = 'watershed_id'


def save_figure(target_path, plot_func, plot_func_args):
    fig = plot_func(*plot_func_args)
    fig.savefig(target_path, bbox_inches='tight')


def plot_raster_diffs(workspace_dir, scenario_name):
    _raster_dtype_list = (
        (os.path.join(workspace_dir, scenario_name, f'diff_avoided_erosion_{scenario_name}.tif'), 'divergent', 'linear'),
        (os.path.join(workspace_dir, scenario_name, f'diff_avoided_export_{scenario_name}.tif'), 'divergent', 'linear'),
        (os.path.join(workspace_dir, scenario_name, f'diff_sed_deposition_{scenario_name}.tif'), 'divergent', 'linear'),
        (os.path.join(workspace_dir, scenario_name, f'diff_sed_export_{scenario_name}.tif'), 'divergent', 'linear'),
        (os.path.join(workspace_dir, scenario_name, f'diff_rkls_{scenario_name}.tif'), 'divergent', 'linear'),
        (os.path.join(workspace_dir, scenario_name, f'diff_usle_{scenario_name}.tif'), 'divergent', 'linear')
    )

    _tif_list, _dtype_list, _transform_list = zip(*_raster_dtype_list)

    fig = utils.plot_raster_list(
        _tif_list,
        datatype_list=_dtype_list,
        transform_list=_transform_list)
    return fig


def report(args, model_spec, file_registry, workspace_registries):

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.

    task_graph = taskgraph.TaskGraph(
        os.path.join(args['workspace_dir'], 'taskgraph_cache'),
        n_workers=n_workers)

    images_dir = os.path.join(args['workspace_dir'], '_images')
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    output_html_path = os.path.join(args['workspace_dir'], 'report.html')
    scenarios_df = pandas.read_csv(args['scenarios'])
    scenario_names = list(scenarios_df.name)
    baseline_name = scenario_names.pop(0)
    scenario_table_html = scenarios_df.to_html(table_id='name', index=False)

    results_df = pandas.read_csv(file_registry['watershed_results.csv'])
    wide_df = results_df.pivot(
        index=[FID_COL_NAME, 'variable'],
        columns=SCENARIO_COL_NAME,
        values='value')
    wide_df.columns.name = None
    wide_df.reset_index(inplace=True)
    for _scen in scenario_names:
        wide_df[f'{_scen}_percent_change'] = (
            (wide_df[_scen] - wide_df[baseline_name]) / wide_df[baseline_name]
        ) * 100
    cols = list(wide_df.columns)
    cols.insert(0, cols.pop(cols.index(baseline_name)))
    cols.insert(0, cols.pop(cols.index('variable')))
    cols.insert(0, cols.pop(cols.index(FID_COL_NAME)))
    wide_df = wide_df[cols]
    watersheds_table = wide_df.to_html(table_id="watersheds", index=False)

    raster_dtype_list = (
        ('avoided_erosion', 'continuous', 'linear'),
        ('avoided_export', 'continuous', 'log'),
        ('sed_deposition', 'continuous', 'log'),
        ('sed_export', 'continuous', 'log'),
        ('rkls', 'continuous', 'linear'),
        ('usle', 'continuous', 'log')
    )

    sdr_raster_fig_per_scenario = {}
    for (raster_id, datatype, transform) in raster_dtype_list:
        tif_list = []
        subtitle_list = []
        for _, row in scenarios_df.iterrows():
            scenario_name = row['name']
            subtitle_list.append(scenario_name)
            scenario_file_reg = workspace_registries[scenario_name].file_registry
            tif_list.append(
                scenario_file_reg[raster_id])
        png_path = os.path.join(
            images_dir, f'{raster_id}.png')
        sdr_raster_fig_per_scenario[raster_id] = png_path
        task_graph.add_task(
            func=save_figure,
            kwargs={
                'target_path': png_path,
                'plot_func': utils.plot_raster_facets,
                'plot_func_args': (tif_list, datatype, transform, subtitle_list)
            },
            target_path_list=[png_path],
            task_name=f'save_figure_{raster_id}'
        )

    for key, png_path in sdr_raster_fig_per_scenario.items():
        sdr_raster_fig_per_scenario[key] = utils.base64_encode_file(png_path)

    diff_figure_per_scenario = {}
    for scenario in scenario_names:
        png_name = f'diff_{scenario}.png'
        png_path = os.path.join(images_dir, png_name)
        diff_figure_per_scenario[scenario] = png_path
        task_graph.add_task(
            func=save_figure,
            kwargs={
                'target_path': png_path,
                'plot_func': plot_raster_diffs,
                'plot_func_args': (args['workspace_dir'], scenario)
            },
            target_path_list=[png_path],
            task_name=f'save_figure_{png_name}'
        )

    task_graph.close()
    task_graph.join()

    # Read pngs into memory so they can be embedded in html
    for key, png_path in diff_figure_per_scenario.items():
        diff_figure_per_scenario[key] = utils.base64_encode_file(png_path)

    env = Environment(
        loader=PackageLoader('invest_sdr_scenario_compare', 'templates'))
    template = env.get_template('report.html')
    invest_reports_env = env = Environment(
        loader=PackageLoader('invest_reports', 'jinja_templates'))

    with open(output_html_path, "w", encoding="utf-8") as output_file:
        output_file.write(template.render(
            model_name=model_spec.model_title,
            scenario_table_html=scenario_table_html,
            watersheds_data_description=model_spec.get_output(
                'watershed_results.csv').about,
            watersheds_data=watersheds_table,
            sdr_scenario_rasters=sdr_raster_fig_per_scenario,
            sdr_diff_rasters=diff_figure_per_scenario,
            model_spec_outputs=model_spec.outputs,
            invest_reports_env=invest_reports_env,
        ))
