import os
import sys

from invest_reports import utils
from natcap.invest import datastack
import matplotlib
import matplotlib.pyplot as plt
import pandas
import taskgraph

from jinja2 import Environment, PackageLoader


SCENARIO_COL_NAME = 'scenario'
FID_COL_NAME = 'watershed_id'

env = Environment(
    loader=PackageLoader('invest_sdr_scenario_compare', 'templates'))
TEMPLATE = env.get_template('report.jinja')


def plot_raster_png(raster_path):
    target_path = f'{os.path.splitext(raster_path)[0]}.png'
    fig, ax = plt.subplots()  # Create a figure containing a single Axes.
    arr, resampled = utils.read_masked_array(raster_path, 'bilinear')
    mappable = ax.imshow(arr, cmap='viridis')
    fig.colorbar(mappable, ax=ax)
    ax.set(
        title="FOO\nsubtitle")
    ax.set_axis_off()
    plt.savefig(target_path)
    return target_path


def save_figure(target_path, plot_func, plot_func_args):
    fig = plot_func(*plot_func_args)
    fig.savefig(target_path)


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


def report(args):
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

    jinja_data = {}
    workspace_dir = args['workspace_dir']
    images_dir = os.path.join(workspace_dir, '_images')
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    output_html_path = os.path.join(workspace_dir, 'report.html')
    scenarios_df = pandas.read_csv(args['scenarios'])
    baseline_name = scenarios_df.scenarios[0]

    # png_path = plot_raster_png(os.path.join(
    #     workspace_dir, 'alternative', 'diff_sed_export_alternative.tif'))

    watershed_results_table_path = os.path.join(
        workspace_dir, 'watershed_results.csv')
    results_df = pandas.read_csv(watershed_results_table_path)
    wide_df = results_df.pivot(
        index=[FID_COL_NAME, 'variable'], columns=SCENARIO_COL_NAME, values='value')
    wide_df.columns.name = None
    wide_df.reset_index(inplace=True)
    for _scen in scenarios_df.scenarios:
        if _scen == baseline_name:
            continue
        wide_df[f'{_scen}_percent_change'] = (
            (wide_df[_scen] - wide_df[baseline_name]) / wide_df[baseline_name]
        ) * 100
    cols = list(wide_df.columns)
    cols.insert(0, cols.pop(cols.index(baseline_name)))
    cols.insert(0, cols.pop(cols.index('variable')))
    cols.insert(0, cols.pop(cols.index(FID_COL_NAME)))
    wide_df = wide_df[cols]
    jinja_data['watersheds_data'] = wide_df.to_html(table_id="watersheds", index=False)

    raster_dtype_list = (
        ('avoided_erosion{suffix_str}.tif', 'continuous', 'linear'),
        ('avoided_export{suffix_str}.tif', 'continuous', 'log'),
        ('sed_deposition{suffix_str}.tif', 'continuous', 'log'),
        ('sed_export{suffix_str}.tif', 'continuous', 'log'),
        ('rkls{suffix_str}.tif', 'continuous', 'linear'),
        ('usle{suffix_str}.tif', 'continuous', 'log')
    )

    fig_dict = {}
    for raster_tuple in raster_dtype_list:
        tif_list = []
        title_list = []
        for _, row in scenarios_df.iterrows():
            scenario = row.scenarios
            title_list.append(scenario)
            logfile = row.logfiles
            _, ds_info = datastack.get_datastack_info(logfile)
            ws = ds_info.args['workspace_dir']
            suffix = ds_info.args['results_suffix']
            tif_list.append(
                os.path.join(ws, raster_tuple[0].format(suffix_str=suffix)))
            datatype = raster_tuple[1]
            transform = raster_tuple[2]
        png_path = os.path.join(
            images_dir, f'{os.path.splitext(raster_tuple[0])[0]}.png')
        fig_dict[raster_tuple[0]] = png_path
        task_graph.add_task(
            func=save_figure,
            kwargs={
                'target_path': png_path,
                'plot_func': utils.plot_raster_facets,
                'plot_func_args': (tif_list, datatype, transform, title_list)
            },
            target_path_list=[png_path],
            task_name=f'save_figure_{raster_tuple[0]}'
        )
        # fig = utils.plot_raster_facets(
        #     tif_list, datatype, transform=transform, subtitle_list=title_list)
        # fig.savefig(png_path)

    jinja_data["sdr_scenario_rasters"] = fig_dict

    diff_fig_dict = {}
    for _scenario in scenarios_df.scenarios:
        if _scenario != baseline_name:
            png_name = f'diff_{_scenario}.png'
            png_path = os.path.join(images_dir, png_name)
            diff_fig_dict[_scenario] = png_path
            task_graph.add_task(
                func=save_figure,
                kwargs={
                    'target_path': png_path,
                    'plot_func': plot_raster_diffs,
                    'plot_func_args': (workspace_dir, _scenario)
                },
                target_path_list=[png_path],
                task_name=f'save_figure_{png_name}'
            )
            # fig = plot_raster_diffs(workspace_dir, _scenario)
            # fig.savefig(png_path)
    jinja_data["sdr_diff_rasters"] = diff_fig_dict

    task_graph.close()
    task_graph.join()

    with open(output_html_path, "w", encoding="utf-8") as output_file:
        output_file.write(TEMPLATE.render(jinja_data))

