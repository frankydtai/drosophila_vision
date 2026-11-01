
# Plotting data from Figure 4

This directory includes files to simulate model responses to different stimuli
 from "Non-preferred contrast responses in the Drosophila motion pathways reveal
a receptive field structure that explains a common visual illusion "

Authors
* [Eyal Gruntman](https://orcid.org/0000-0003-1383-7347)
* [Pablo Reimers](https://orcid.org/0000-0003-0547-1349)
* [Sandro Romani](https://orcid.org/0000-0002-4727-4207)
* [Michael B. Reiser](https://orcid.org/0000-0002-4108-4517)

To generate the figure run modelFigureCode from the source directory

## Files and variables

The files were generated with matlab r2020b

### optTables.mat

This file includes 4 matlab tables containing the optimization results for the
T4 and T5 averaged cells (cardinal and diagonal cells averaged separately)

Each table contains the model parameters along with quality control statistics

### modelFigureCode

Matlab script for generating model simulated responses to single bar flashes,
moving bars, moving edges and minimal reverse-phi stimuli.

 - To explore different parameter sets change the iteration number on line 6
 - To explore diagonal T5 results or T4 results change the optTab in line 9
 - Note! when loading T4 parameters t5_simple_wrap should be changed to
   t4_simple_wrap (lines: 60, 122, 123, 185, 186, 259, 260)


 ## Authors

 * [Pablo Reimers](https://orcid.org/0000-0003-0547-1349)
 * [Sandro Romani](mailto:romanis@janelia.hhmi.org)
