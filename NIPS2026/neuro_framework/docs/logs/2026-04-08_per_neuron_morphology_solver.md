# FAFBMCNetwork Per-Neuron Morphology Solver

## Date
2026-04-08

## Goal
Replace the optic-lobe `FAFBMCNetwork` shared-morphology solver with a true per-neuron morphology solver while keeping the existing neuron-parameter and synapse-parameter initialization flow intact.

## What Changed
- `MorphologyGraph` now supports direct construction from in-memory SWC rows via `from_swc_rows(...)` and `from_nodes(...)`.
- `FAFBMCNetwork.from_preprocessed_with_morphology_packages(...)` no longer stops at morphology metadata attachment.
- The factory now:
  - builds the usual FAFB neuron/synapse parameter tables
  - loads each neuron's packed simplified SWC
  - converts each neuron into its own `MorphologyGraph`
  - concatenates all neuron morphologies into one global sparse axial graph
  - switches the network to a ragged per-neuron morphology mode

## Solver Design
- The existing `_solve_implicit(...)` routine is reused directly.
- Instead of solving `B * N` copies of one shared morphology, the new path solves `B` copies of one large disconnected sparse graph containing all neuron compartments.
- Each compartment stores an owning neuron index through `net.comp_owner`.
- HH parameters remain neuron-level, but are expanded onto compartments at runtime.
- Input current is injected into `net.soma_comp_idx`.
- Synapses currently use:
  - presynaptic compartment: soma/root compartment
  - postsynaptic compartment: last compartment in that neuron's compartment list

## Fallback Behavior
- Some `root_id`s in `neurons.csv` may not have a packed morphology.
- In that case, the builder falls back to a representative packed morphology instead of failing the whole optic-lobe build.
- Metadata exposed on the network:
  - `net.morphology_fallback_root_ids`
  - `net.morphology_fallback_count`
  - `net.morphology_graph_mode = "per_neuron_global_sparse"`
  - `net.morphology_compartment_total`
  - `net.morphology_compartment_max`

## Current Limits
- The target-compartment heuristic is still simple (`last_compartment`), not yet branch-class aware.
- This log only covers the geometry solver change.
- Full-scale optic-lobe benchmarking, memory profiling, and differentiability verification remain separate follow-up tasks.

## Minimal Verification
- `py_compile` passed for:
  - `neuro_framework/models/fafb_mc_network.py`
  - `neuro_framework/models/morphology.py`
- Small real-data subset build passed:
  - ragged mode enabled
  - per-neuron state shape became `(batch, total_comp)`
- Small synthetic-edge forward pass passed:
  - network built in ragged mode
  - synapse path executed
  - output shape remained `(B, T, N)`

## Status
Implemented for the optic-lobe builder; large-scale validation still in progress.
