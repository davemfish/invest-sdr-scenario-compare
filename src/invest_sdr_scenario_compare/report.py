import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium", app_title="SDR scenario comparison")


@app.cell
def _():
    import os

    import geopandas
    import marimo as mo
    from natcap.invest import datastack
    import pandas
    from pandas.api.types import is_float_dtype
    import pygeoprocessing

    from invest_reports import utils
    return datastack, is_float_dtype, mo, os, pandas, utils


@app.cell
def _(datastack, mo):
    logfile_path = mo.cli_args().get('logfile')
    logfile_path = 'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/scenario_compare/InVEST-sdr_compare_scenarios-log-2025-08-22--16_25_31.txt'
    # logfile_path = 'C:/Users/dmf/projects/forum/sdr/sample_report3/InVEST-sdr-log-2025-07-18--14_34_11.txt'
    _, ds_info = datastack.get_datastack_info(logfile_path)
    args_dict = ds_info.args
    workspace = args_dict['workspace_dir']
    mo.accordion({'SDR model arguments': args_dict})
    return args_dict, workspace


@app.cell
def _(args_dict, pandas):
    scenarios_df = pandas.read_csv(args_dict['scenarios'])
    baseline_name = scenarios_df.scenarios[0]
    scenarios_df
    return baseline_name, scenarios_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results by Watershed""")
    return


@app.cell
def _(baseline_name, is_float_dtype, mo, os, pandas, scenarios_df, workspace):
    watershed_results_table_path = os.path.join(
        workspace,
        'watershed_results.csv')
    results_df = pandas.read_csv(watershed_results_table_path)

    vars = ["usle_tot", "sed_export", "sed_dep", "avoid_exp", "avoid_eros"]
    _df_dict = {}
    for var in vars:
        _df = results_df.loc[results_df.variable == var].pivot(index='id', columns='scenario', values='value')
        for _scen in scenarios_df.scenarios:
            if _scen == baseline_name:
                continue
            _df[f'{_scen}_percent_change'] = ((_df[_scen] - _df[baseline_name]) / _df[baseline_name]) * 100
        cols = list(_df.columns)
        cols.insert(0, cols.pop(cols.index(baseline_name)))
        _df = _df[cols]
        _df_dict[var] = mo.ui.table(
            _df,
            format_mapping={col: "{:.2f}".format for col in _df.columns if is_float_dtype(_df[col])})
    mo.accordion(_df_dict)
    # pandas.melt(df, id_vars=['id', 'scenario'])
    # No reason for choropleth when there is only one feature
    # if len(ws_vector) > 1:
    #     _fields = ["usle_tot", "sed_export", "sed_dep", "avoid_exp", "avoid_eros"]
    #     mo.output.replace(utils.plot_choropleth(ws_vector, _fields))
    return


@app.cell
def _(os, workspace):
    watershed_results_vector_path = os.path.join(workspace, _scenario, f'diff_watershed_results_sdr_{_scenario}.gpkg')
    return


@app.cell
def _(mo):
    mo.md(r"""## Raster Results: Across Scenarios""")
    return


@app.cell(disabled=True)
def _(datastack, mo, os, scenarios_df, utils):
    _raster_dtype_list = (
        ('avoided_erosion{suffix_str}.tif', 'continuous', 'linear'),
        ('avoided_export{suffix_str}.tif', 'continuous', 'log'),
        ('sed_deposition{suffix_str}.tif', 'continuous', 'log'),
        ('sed_export{suffix_str}.tif', 'continuous', 'log'),
        ('rkls{suffix_str}.tif', 'continuous', 'linear'),
        ('usle{suffix_str}.tif', 'continuous', 'log')
    )

    _fig_dict = {}
    for _raster_tuple in _raster_dtype_list:
        _tif_list = []
        _title_list = []
        for _, _row in scenarios_df.iterrows():
            _scenario = _row.scenarios
            _title_list.append(_scenario)
            _logfile = _row.logfiles
            _, _ds_info = datastack.get_datastack_info(_logfile)
            _ws = _ds_info.args['workspace_dir']
            _suffix = _ds_info.args['results_suffix']
            _tif_list.append(
                os.path.join(_ws, _raster_tuple[0].format(suffix_str=_suffix)))
            datatype = _raster_tuple[1]
            transform = _raster_tuple[2]
        # _tif_list, _dtype_list, _transform_list = zip(*_raster_list)
        _fig_dict[_raster_tuple[0]] = utils.plot_raster_facets(
            _tif_list, datatype, transform, _title_list)


    # mo.vstack(_fig_list)

    mo.accordion(_fig_dict)
    return


@app.cell
def _(mo):
    mo.md(r"""## Raster Results: Differences from base scenario""")
    return


@app.cell(disabled=True)
def _(baseline_name, mo, os, scenarios_df, utils, workspace):
    def _plot_raster_diffs(scenario_name):
        _raster_dtype_list = (
            (os.path.join(workspace, scenario_name, f'diff_avoided_erosion_{scenario_name}.tif'), 'divergent', 'linear'),
            (os.path.join(workspace, scenario_name, f'diff_avoided_export_{scenario_name}.tif'), 'divergent', 'linear'),
            (os.path.join(workspace, scenario_name, f'diff_sed_deposition_{scenario_name}.tif'), 'divergent', 'linear'),
            (os.path.join(workspace, scenario_name, f'diff_sed_export_{scenario_name}.tif'), 'divergent', 'linear'),
            (os.path.join(workspace, scenario_name, f'diff_rkls_{scenario_name}.tif'), 'divergent', 'linear'),
            (os.path.join(workspace, scenario_name, f'diff_usle_{scenario_name}.tif'), 'divergent', 'linear')
        )

        _tif_list, _dtype_list, _transform_list = zip(*_raster_dtype_list)

        _figure = utils.plot_raster_list(
            _tif_list,
            datatype_list=_dtype_list,
            transform_list=_transform_list,
        )
        return _figure

    _fig_dict = {}
    for _scenario in scenarios_df.scenarios:
        if _scenario != baseline_name:
            _fig_dict[_scenario] = _plot_raster_diffs(_scenario)

    # mo.vstack(_fig_list)

    mo.accordion(_fig_dict)
    return


@app.cell
def _():
    ## Raster Stats
    return


@app.cell
def _(mo, os, scenarios_df, utils, workspace):
    _stats_dict = {}
    for _scenario in scenarios_df.scenarios:
        _raster_stats = utils.raster_workspace_summary(os.path.join(workspace, _scenario))
        _stats_dict[_scenario] = _raster_stats

    mo.accordion(_stats_dict, multiple=True)
    return


@app.cell
def _():
    # import matplotlib
    # import matplotlib.pyplot as plt
    # # _tif_path = os.path.join(workspace, 'alternative', f'diff_sed_export_alternative.tif')
    # # raster_info = pygeoprocessing.get_raster_info(_tif_path)
    # # print(raster_info['overviews'][-1])
    # # raster_info['overviews']

    # fig, ax = plt.subplots()             # Create a figure containing a single Axes.
    # arr, resampled = utils.read_masked_array(os.path.join(workspace, 'alternative', f'diff_sed_export_alternative.tif'), 'linear')
    # mappable = ax.imshow(arr, cmap='BrBG', norm=matplotlib.colors.CenteredNorm())
    # fig.colorbar(mappable, ax=ax)
    # ax.set(
    #     title="FOO\nbaseline")
    # ax.set_axis_off()
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## *
    Raster plots with an __*__ were resampled to lower resolution for plotting. Full resolution rasters are available in the output workspace.
    """
    )
    return


if __name__ == "__main__":
    app.run()
