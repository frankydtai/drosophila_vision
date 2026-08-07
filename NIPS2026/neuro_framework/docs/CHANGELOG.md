# Changelog

All notable changes to `neuro_framework` are recorded here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-04-04

### Added

#### `models/`
- **`synapses.py`** ⭐ NEW — Synapse models inspired by Jaxley:
  - `BaseSynapse` — Abstract base class for all synapse models
  - `TanhRateSynapse` — Tanh-based rate synapse (no state variables)
  - `TanhConductanceSynapse` — Tanh synapse with conductance-based current
  - `IonotropicSynapse` — Biophysical ionotropic synapse with state variable `s`
  - All models support gradient-based learning
  - Reference: Jaxley (https://jaxley.readthedocs.io/), Abbott & Marder (1998)

#### `models/network_torch.py`
- **Synapse model integration**:
  - New parameter: `synapse_model` (str or BaseSynapse instance)
  - New parameter: `learn_synapse_params` (bool)
  - Automatic synapse state initialization and updates in forward pass
  - Support for voltage-dependent synaptic currents
  - Backward compatible (default is 'simple' weight-based synapses)

#### `notebooks/`
- **`05_synapse_models.ipynb`** ⭐ NEW — Comprehensive synapse models demonstration:
  - Load connectome data
  - Create networks with each synapse model
  - Compare activity patterns across models
  - Visualize differences (heatmaps, time traces)
  - Training example with ionotropic synapses
  - Summary table and comparison
- **`test_synapse_models.py`** — Automated testing script for all synapse models

#### `docs/`
- **`synapse_models_integration.md`** — Complete documentation:
  - Overview of all synapse models
  - Mathematical formulations
  - Usage examples
  - Comparison table
  - Implementation details
  - Testing results
  - Future work

### Changed

#### `notebooks/`
- **`00_quick_start.ipynb`** — Added reference to new synapse models notebook
- **`README.md`** — Updated with synapse models section and new figure count (32 total)

#### `docs/`
- **`README.md`** — Updated with synapse models in overview and quick start
- **`CHANGELOG.md`** — This file

### Fixed
- None

---

## [0.1.0] — 2026-03-30

### Added

#### `connectome/`
- `loader.py` — `ConnectomeLoader` class with factory methods `from_banc()` and
  `from_fafb()`. Normalises column names across BANC and FAFB datasets into a
  common schema (`root_id`, `cell_type`, `nt_type`, `pre_root_id`, `post_root_id`,
  `syn_count`). Exposes `get_adjacency_tensors()` and `nt_sign()` helpers.

#### `models/`
- `dynamics.py` — Three neuron dynamics models sharing the `BaseDynamics`
  interface (`write_initial_state`, `state_velocity`, `step`):
  - `VoltageModel` — DMN / FlyVis leaky-integrator (Lappalainen et al. 2024)
  - `LIFModel` — Leaky Integrate-and-Fire with straight-through surrogate gradient
  - `HHModel` — Single-compartment Hodgkin-Huxley (m/h/n gating variables)
  - `DYNAMICS_REGISTRY` + `build_dynamics()` factory
- `network_torch.py` — `ConnectomeNetwork(nn.Module)`: connectome-constrained
  recurrent network. Learnable log-space synapse weights, time constants, biases.
  `from_loader()` factory. `simulate()` / `forward()` return `(B, T, N)` tensor.
- `network_jax.py` — `JaxleyNetwork`: wraps `jx.Network` (Jaxley library).
  Builds point-neuron network from connectome, adds HH / Leak channels,
  connects via `IonotropicSynapse`. `simulate()`, `simulate_batch()`,
  `make_loss_and_grad()` for JAX differentiable training.

#### `stimulus/`
- `visual.py` — Visual stimulus generators:
  - `FlashStimulus` — full-field on/off flash
  - `MovingBarStimulus` — oriented bar sweeping at configurable speed/direction
  - `MovingEdgeStimulus` — luminance step-edge
  - `SinusoidalGrating` — spatiotemporal sinusoidal grating
  - `hex_grid_coords(radius)` — hexagonal grid coordinate utility
  - `build_stimulus_tensor()` — one-line factory returning `(B, T, N)` tensor

#### `training/`
- `losses.py` — Loss function library:
  - `mse_loss` — mean squared error
  - `correlation_loss` — 1 − Pearson correlation
  - `spike_rate_loss` — MSE on mean firing rates
  - `direction_selectivity_loss` — DSI loss for T4/T5 direction tuning
  - `knockout_consistency_loss` — DMN Method A silencing constraint
  - `combined_loss` — weighted multi-objective sum
  - `LossRegistry` — name → function lookup dict
- `trainer.py` — Training loops:
  - `TorchTrainer` — PyTorch autograd trainer with DataLoader, validation,
    checkpointing, Method A (`step_with_knockout()`), Method B
    (`layerwise_train()`)
  - `JaxTrainer` — JAX + Optax gradient-descent loop for `JaxleyNetwork`
  - `TrainingHistory` — lightweight metrics container

#### `utils/`
- `logging.py` — Centralised logging configuration. Routes all `neuro_framework`
  logs to both console and `logs/neuro_framework.log`.

#### `docs/`
- `CHANGELOG.md` — this file
- `architecture.md` — module descriptions, design decisions, data-flow diagram
- `todo.md` — remaining work and NeurIPS 2026 milestones

#### `logs/`
- `.gitkeep` — placeholder; runtime log files written here, excluded from git

---

## Planned

### [0.2.0] — target ~2026-04-07
- Load real FAFB LC connectome (LC4, LC6, LC9, LC11, LC15 subtypes)
- Calcium imaging ground-truth loader (`data/calcium/`)
- FAFB connectome integration test
- Method A training demo notebook

### [0.3.0] — target ~2026-04-21
- Method B layer-wise training demo and benchmark vs DMN baseline
- Jaxley HH simulation benchmark on optic-lobe-scale network
- NeuroMechFly DN interface (demo only)
- Ablation: Voltage vs LIF vs HH on direction selectivity task

### [0.4.0] — target ~2026-05-01 (paper freeze)
- Final results, figures, paper-ready plots
- Code cleanup and open-source preparation
