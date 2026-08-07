# FAFBMCNetwork Parameter Hierarchy Log

## Date
2026-04-13

## Goal
Add a 3-level ion-parameter initialization hierarchy to `FAFBMCNetwork`:

1. neurotransmitter default
2. neuron type default
3. root_id override

This keeps current type-level behavior as the default, but allows:
- fallback initialization for neuron types without direct rules
- future literature-backed neurotransmitter priors
- per-neuron overrides for selected root IDs

## TODO
- [x] Inspect available columns in `neurons.csv`, `connections.csv`, and ion-rule CSVs
- [x] Confirm neurotransmitter must be inferred from `connections.csv::nt_type`
- [x] Add neurotransmitter default ion-rule support in `FAFBMCNetwork`
- [x] Add root-id override ion-rule support in `FAFBMCNetwork`
- [ ] Update `notebooks_mcHH/01_build_optic_lobe_model.ipynb` to show the new interface
- [ ] Run a minimal build test and lint check

## Design Notes
- `neurons.csv` currently has no neurotransmitter column
- neurotransmitter is inferred per presynaptic neuron from the dominant outgoing `nt_type`, weighted by `syn_count`
- type rule still has higher priority than neurotransmitter default
- `root_id` override has highest priority
- current morphology solver remains shared; this log only tracks Step 1 (parameter hierarchy)

## Code Changes
### `neuro_framework/models/fafb_mc_network.py`
Added:
- `_DEFAULT_ION_BY_NT`
- `_normalize_nt()`
- `_load_nt_ion_rules()`
- `_load_root_ion_overrides()`
- `_infer_root_nt_types()`

Updated:
- `FAFBMCNetwork.from_preprocessed(...)`
  - new optional args:
    - `nt_ion_rules_path`
    - `neuron_ion_overrides_path`
  - now builds ion parameters via:
    - nt default -> type rule -> root override
  - attaches metadata:
    - `net.root_ids`
    - `net.neuron_nt_types`
    - `net.neuron_param_source`
    - `net.root_ion_overrides_count`

## Expected CSV Schemas
### Neurotransmitter default ion rules
Columns:
- `nt_type` or `neurotransmitter`
- `gNa`
- `gK`
- `gLeak`
- `eNa`
- `eK`
- `eLeak`

### Root override ion rules
Columns:
- `root_id`
- optional subset of:
  - `gNa`, `gK`, `gLeak`, `eNa`, `eK`, `eLeak`

## Status
In progress.
