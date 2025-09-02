import os
import sys

from invest_reports import utils
from natcap.invest import datastack
import matplotlib
import matplotlib.pyplot as plt
import pandas

from jinja2 import Environment, PackageLoader

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


def report(args):
    workspace_dir = args['workspace_dir']
    # mo.accordion({'SDR model arguments': args})
    output_html_path = os.path.join(workspace_dir, 'report.html')
    scenarios_df = pandas.read_csv(args['scenarios'])
    baseline_name = scenarios_df.scenarios[0]

    # png_path = plot_raster_png(os.path.join(
    #     workspace_dir, 'alternative', 'diff_sed_export_alternative.tif'))

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
        fig = utils.plot_raster_facets(
            tif_list, datatype, transform=transform, subtitle_list=title_list)
        png_path = os.path.join(
            workspace_dir, f'{os.path.splitext(raster_tuple[0])[0]}.png')
        fig.savefig(png_path)

        fig_dict[raster_tuple[0]] = png_path

    jinja_data = {"sdr_scenario_rasters": fig_dict}

    with open(output_html_path, "w", encoding="utf-8") as output_file:
        # with open(input_template_path) as template_file:
        #     j2_template = Template(template_file.read())
        #     output_file.write(j2_template.render(jinja_data))
        output_file.write(TEMPLATE.render(jinja_data))


if __name__ == '__main__':
    logfile_path = 'C:/Users/dmf/projects/forum/sdr_ndr_swy_luzon/scenario_compare2/InVEST-sdr_compare_scenarios-log-2025-08-29--10_22_56.txt'
    _, ds_info = datastack.get_datastack_info(logfile_path)
    report(ds_info.args)
