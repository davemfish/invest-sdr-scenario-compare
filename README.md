## SDR Scenario Comparison: an InVEST Plugin

Use this InVEST plugin to calculate the difference between SDR scenarios.

Before using this plugin, run the SDR model any number of times _for the same geographic region_ 
to generate scenarios. Designate one of those scenarios as the "baseline", and use this plugin 
to compare each other scenario to the baseline scenario.

Scenarios are compared by creating difference maps ("scenario - baseline")
for the main raster outputs of SDR. Differences at the watershed-scale are also
calculated and presented as a "percent change" from the baseline scenario.

_In order for scenarios to be compared, they must all use the same `Watersheds` input 
and the same `Digital Elevation Model` input._

### Setup:

Make a CSV with three columns: 'name', 'description', and 'logfile'. Example,
|name              |description                               |logfile                                                                                              |
|------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------|
|baseline          |the SDR sample data                       |C:/Users/dmf/projects/sdr/InVEST-sdr-log-2025-10-06--15_36_34.txt                                    |
|no_erosion_control|all LULC categories have 0 erosion control|C:/Users/dmf/projects/sdr_no_erosion_control/InVEST-sdr-log-2025-10-06--15_37_44.txt                 |

Each row represents a scenario.
* 'name' column should be a simple label to describe that scenario, for example, 'baseline'.
* 'description' should be a text description of the scenario and how it differs from the baseline scenario.
* 'logfile' should be the path to the InVEST logfile (.txt file) that was generated when the
SDR model was run.

Include as many rows as desired. **The first row will be treated as the baseline scenario** 
to which all other scenarios are compared.

### Interpreting Results:

After running the plugin, find the `report.html` file in the output workspace.
Open this file in a web browser to see a summary of the scenario comparison.

The output workspace also includes a folder for each scenario (other than the baseline), 
which includes the differenced rasters and watershed results table.

### How to install InVEST plugins?

InVEST Workbench users can install this plugin using the "Manage Plugins" menu
within the Workbench, and this plugin's URL: https://github.com/davemfish/invest-sdr-scenario-compare/

Python users can install this repository from source:  
`pip install invest-sdr-scenario-compare@git+https://github.com/davemfish/invest-sdr-scenario-compare.git@main`  
See `pyproject.toml` for dependencies.
