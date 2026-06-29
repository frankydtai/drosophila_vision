
# Plotting Figure 2

This directory includes files to generate Figure 2 from
"Non-preferred contrast responses in the Drosophila motion pathways reveal
a receptive field structure that explains a common visual illusion "

Authors
* [Eyal Gruntman](https://orcid.org/0000-0003-1383-7347)
* [Pablo Reimers](https://orcid.org/0000-0003-0547-1349)
* [Sandro Romani](https://orcid.org/0000-0002-4727-4207)
* [Michael B. Reiser](https://orcid.org/0000-0002-4108-4517)

To generate the figure run Figure2Code from the directory after adding the
"supportingFunctions" folder to your path.

## Files and variables

The files were generated with matlab r2020b

### singleBarStT4.mat

This file contains a structure with all T4 cells responses to single bar flashes

Each field in the main structure contains the results from one cell organized supportingFunctions
5 fields:
.result - contains raw and processed recordings in a 4-D array
(position, duration, width, value)

.positions - the positions that were assayed for the cell (before alignment)
.durations - flash durations that were assayed (in secs)
.width -     bar widths that were assayed (in LED pixels)
.vals -      bar contrast that was assayed

These can be used as a reference to the .result field.
Note! the first and second dimensions for the result array are larger by one
to include summary statistics

### singleBarStT5

Same as above only for T5 cells

### Figure2Code

Matlab script for generating figure 2, with all the necessary calculations

## Authors
* [Eyal Gruntman](mailto:gruntmane@janelia.hhmi.org)
