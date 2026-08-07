# Architecture & Design Document

> `neuro_framework` — Connectome-constrained neural network modelling  
> of the Drosophila visual system  
> NeurIPS 2026

---

## 1. Project Goal

Replace the simplified LC (Lobula Columnar) layer in the DMN paper  
(Lappalainen et al., *Nature* 2024) with real synaptic connectivity from  
the FlyWire FAFB v783 connectome, train the resulting network to match  
calcium imaging recordings from LC neurons, and show improved performance  
over the DMN and RF (Shiu et al.) baselines.

**Visual pathway modelled:**
```
Retina / Visual Scene
      ↓
Photoreceptors  (R1–R8)
      ↓
T4 / T5 cells   (motion detection, 4 subtypes each)
      ↓
Tm cells        (Transmedullary neurons)
      ↓
LC cells        (Lobula Columnar — real FAFB connectome)  ← core contribution
      ↓
DN              (Descending Neurons — visual output)
      ↓
[Demo] NeuroMechFly body model (MuJoCo)
```

---

## 2. Repository Layout

```
NIPS2026/
├── neuro_framework/          ← THIS package (model + training)
│   ├── connectome/           ← data loading
│   ├── models/               ← neuron & network implementations
│   ├── stimulus/             ← visual stimulus generators
│   ├── training/             ← loss functions & trainers
│   ├── utils/                ← logging, common helpers
│   ├── docs/                 ← project documentation (this folder)
│   ├── logs/                 ← runtime log files (git-ignored)
│   └── README.md
│
├── flyvis/                   ← FlyVis / DMN reference code (read-only)
├── Jaxley_notebook/          ← Jaxley tutorial & optic-lobe example
├── Shiu/                     ← Shiu RF-model notebook (Brian2)
├── Connectome Dataset/       ← connectome data + utilities
└── docs/                     ← project-level meeting notes & plans
    └── zoom meeting/
```

---

## 3. Module-by-Module Description

### 3.1 `connectome/loader.py`

**Purpose:** Unified I/O layer for connectome data.

| Class / Function | Description |
|---|---|
| `ConnectomeLoader` | Dataclass holding paths, filter settings, and cached DataFrames |
| `.from_banc(**kwargs)` | Factory for BANC dataset (neurons.csv.gz + connections_princeton.csv.gz) |
| `.from_fafb(**kwargs)` | Factory for FAFB/FlyWire dataset (`consolidated_cell_types.csv.gz`, `classification.csv.gz`, `neurons.csv.gz`, `connections_princeton.csv.gz`) |
| `.load()` | Returns `(nodes_df, edges_df)` after applying cell-type and neuropil filters |
| `.get_adjacency_tensors()` | Returns `(pre_idx, post_idx, syn_count)` numpy arrays for tensor construction |
| `.nt_sign()` | Returns `(n_nodes,)` float array: +1 excitatory, −1 inhibitory, 0 unknown |

**Normalised schema:**

*nodes_df columns:* `root_id`, `cell_type`, `nt_type`, `nt_type_verified`, `body_part`, `node_idx`  
*edges_df columns:* `pre_root_id`, `post_root_id`, `syn_count`, `neuropil`, `pre_idx`, `post_idx`

**Design note:** Both BANC and FAFB use different column naming conventions.  
All mapping is handled inside `_normalise_node_columns()` and  
`_normalise_edge_columns()` so downstream code is dataset-agnostic.

---

### 3.2 `models/dynamics.py`

**Purpose:** Neuron dynamics models as composable strategy objects.

All models implement the `BaseDynamics` interface:

```python
class BaseDynamics:
    def write_initial_state(batch, n_nodes, dt, device) -> dict[str, Tensor]
    def state_velocity(state, params, x_t, dt, target_sum) -> dict[str, Tensor]
    def step(state, params, x_t, dt, target_sum) -> dict[str, Tensor]  # Euler
```

| Model | State keys | Biologically motivated for |
|---|---|---|
| `VoltageModel` | `activity` | Rate-coded retina → LC (DMN-style) |
| `LIFModel` | `v`, `z`, `ref` | Spiking neurons, gradient-based training |
| `HHModel` | `v`, `m`, `h`, `n` | High-fidelity single-compartment biophysics |

**Key design choices:**
- `LIFModel` uses a **straight-through estimator** (STE) for spike backprop,  
  enabling gradient flow through the threshold non-linearity.
- `HHModel` uses vectorised gating kinetics; all operations are broadcastable  
  over `(batch, n_nodes)` without Python loops.
- `build_dynamics(name, **kwargs)` + `DYNAMICS_REGISTRY` allow swapping models  
  via config string at construction time.

---

### 3.3 `models/network_torch.py`

**Purpose:** PyTorch recurrent network constrained by connectome topology.

```
ConnectomeNetwork(nn.Module)
├── Buffers (non-trainable):
│   ├── pre_idx    (n_edges,)   source neuron indices
│   ├── post_idx   (n_edges,)   target neuron indices
│   ├── syn_count  (n_edges,)   synapse counts from connectome
│   └── nt_sign    (n_nodes,)   neurotransmitter sign
│
├── Parameters (trainable):
│   ├── log_weight_abs  (n_edges,)   log |synaptic weight|
│   ├── log_tau         (n_nodes,)   log time constant
│   └── bias            (n_nodes,)   resting potential offset
│
└── dynamics:  BaseDynamics subclass
```

**Forward pass:**
1. For each timestep `t`: extract `x_t = x[:, t, :]`
2. Compute signed weights: `sign(nt_sign[pre]) * exp(log_weight_abs)`
3. Compute synaptic input via `scatter_add` over `post_idx`
4. Call `dynamics.step(state, params, x_t, dt, target_sum)`
5. Collect primary state variable → stack to `(B, T, N)`

**Parameterisation:** Weights are in log-space to enforce positivity of  
`|w|`; sign is determined by NT type, not learned, matching biological  
constraints (Dale's law).

---

### 3.4 `models/network_jax.py`

**Purpose:** Biophysical multi-compartment network via Jaxley (JAX backend).

```
JaxleyNetwork
├── Builds jx.Network of point-neuron jx.Cell objects
├── Assigns channels: Leak | Na+K+Leak (HH) per config
├── Groups cells by NT type and cell_type
├── Connects with IonotropicSynapse (from edge table)
└── Exposes:
    ├── simulate(params, t_max, stimuli, dt)  → (v_rec, state)
    ├── simulate_batch(...)                   → jax.vmap over stimuli
    └── make_loss_and_grad(loss_fn, ...)      → value_and_grad function
```

**When to use Jaxley vs PyTorch:**

| Criterion | PyTorch (`ConnectomeNetwork`) | Jaxley (`JaxleyNetwork`) |
|---|---|---|
| Speed on large networks | Faster (vectorised, GPU) | Slower (JAX compile overhead) |
| Biological realism | Medium (DMN/LIF) | High (HH, multi-compartment) |
| Training ecosystem | PyTorch optimisers | Optax |
| Gradient support | Full autograd | JAX value_and_grad |
| Recommended for | Method A/B training | Parameter fitting, HH experiments |

---

### 3.5 `stimulus/visual.py`

**Purpose:** Generate spatiotemporal visual stimuli mapped to input neurons.

All stimulus classes inherit `BaseStimulus` and implement `_generate(t_array, coords)`  
returning `(T, n_nodes)` float32 numpy array.

| Class | Key parameters | Use case |
|---|---|---|
| `FlashStimulus` | `flash_on`, `flash_off`, `amplitude` | ON/OFF flash responses |
| `MovingBarStimulus` | `direction_deg`, `speed`, `bar_width` | T4/T5 direction tuning |
| `MovingEdgeStimulus` | `direction_deg`, `speed` | Edge-motion responses |
| `SinusoidalGrating` | `spatial_freq`, `temporal_freq`, `contrast` | Frequency tuning |

**Coordinate system:** Stimuli are computed in the 2-D plane defined by  
`hex_grid_coords(radius)`. For real photoreceptor positions, pass the  
actual (x, y) coordinates from the connectome node table.

---

### 3.6 `training/losses.py`

**Purpose:** Neuroscience-motivated loss functions.

| Loss | Formula | Purpose |
|---|---|---|
| `mse_loss` | `mean((pred - target)²)` | General supervised matching |
| `correlation_loss` | `1 - Pearson(pred, target)` | Match temporal dynamics |
| `spike_rate_loss` | `MSE(mean_t(pred), target_rates)` | Firing rate targets |
| `direction_selectivity_loss` | `MSE(DSI, target_DSI)` or `-mean(DSI)` | T4/T5 tuning |
| `knockout_consistency_loss` | `cosine_sim(full, silenced)` | DMN Method A constraint |
| `combined_loss` | `Σ wᵢ · lossᵢ` | Multi-objective training |

**DMN Method A** uses `mse_loss` (supervised) + `knockout_consistency_loss`  
(silencing constraint) in a weighted sum. The silencing term penalises  
networks where removing a neuron type has no effect on output.

---

### 3.7 `training/trainer.py`

**Purpose:** High-level training loops orchestrating network, stimulus, loss.

#### `TorchTrainer`

```
train()                  ← standard supervised loop
step_with_knockout()     ← Method A: supervised + silencing loss
layerwise_train()        ← Method B: progressive layer unfreezing
save() / load()          ← checkpoint to logs/ or custom path
```

**Method A flow:**
```
x_full     → network → pred_full ─┐
x_knockout → network → pred_ko   ─┤→ combined_loss → backward()
                         y_gt   ──┘
```

**Method B flow:**
```
Freeze all params
For each layer group (upstream → downstream):
    Unfreeze this group
    Rebuild optimizer with active params
    Run train() for n_epochs_per_layer
```

#### `JaxTrainer`

Wraps a Jaxley `make_loss_and_grad()` function with an Optax optimiser loop.  
Compatible with `optax.adam`, `optax.adamw`.

---

### 3.8 `utils/logging.py`

**Purpose:** Centralised logging setup so all modules write to a consistent  
location without each file needing to configure handlers.

- Configures the `neuro_framework` root logger.
- Writes to **both** `logs/neuro_framework.log` (file, rotating) and stdout.
- Call `setup_logging(level='INFO')` once at the start of a script / notebook.
- Log files are written to `neuro_framework/logs/` which is **git-ignored**.

---

### 3.9 `logs/` directory

| File | Description |
|---|---|
| `.gitkeep` | Keeps the empty directory in git |
| `neuro_framework.log` | Runtime log (auto-created, git-ignored) |
| `training_<run_id>.log` | Per-run training log (auto-created by trainer) |

> **Rule:** All log output goes to `logs/`. Never write log files into  
> `docs/`, `models/`, or the project root.

---

## 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS                                  │
│                                                                 │
│  BANC / FAFB CSVs ──→ ConnectomeLoader ──→ nodes_df, edges_df  │
│                                                                 │
│  hex_grid_coords()  ──→ BaseStimulus ──→ (B, T, N) Tensor      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   ConnectomeNetwork       │  (PyTorch)
                    │   or JaxleyNetwork        │  (Jaxley)
                    │                           │
                    │  params:                  │
                    │  • log_weight_abs (edges) │
                    │  • log_tau        (nodes) │
                    │  • bias           (nodes) │
                    │                           │
                    │  dynamics:                │
                    │  • VoltageModel           │
                    │  • LIFModel               │
                    │  • HHModel                │
                    └────────────┬─────────────┘
                                 │  (B, T, N) activity
                    ┌────────────▼─────────────┐
                    │      Loss Functions       │
                    │  • mse / correlation      │
                    │  • direction_selectivity  │
                    │  • knockout_consistency   │
                    └────────────┬─────────────┘
                                 │  scalar loss
                    ┌────────────▼─────────────┐
                    │        Trainer            │
                    │  TorchTrainer / JaxTrainer│
                    │  Method A / Method B      │
                    │  → checkpoint to logs/    │
                    └───────────────────────────┘
```

---

## 5. Configuration Conventions

| Setting | Where to change | Example |
|---|---|---|
| Dataset source | `ConnectomeLoader.from_banc()` / `from_fafb()` | `cell_types=['LC4','LC6']` |
| Neuron model | `ConnectomeNetwork(dynamics='lif')` | `'voltage'`, `'lif'`, `'hh'` |
| Learning rate | `TorchTrainer(optimizer_cfg={'lr': 1e-3})` | `1e-3` |
| Stimulus type | `build_stimulus_tensor('moving_bar', ...)` | see `stimulus/visual.py` |
| Loss function | `trainer.train(loss_name='correlation')` | `'mse'`, `'correlation'` |
| Log directory | `utils/logging.py` `LOG_DIR` constant | `neuro_framework/logs/` |

---

## 6. Extension Points

### Adding a new dynamics model
1. Subclass `BaseDynamics` in `models/dynamics.py`
2. Implement `write_initial_state()` and `state_velocity()`
3. Register in `DYNAMICS_REGISTRY`

### Adding a new connectome source
1. Add a `from_xxx()` classmethod to `ConnectomeLoader`
2. Implement `_normalise_node_columns()` and `_normalise_edge_columns()` branches
3. No changes needed in the network or training code

### Adding a new loss function
1. Add a function to `training/losses.py`
2. Register in `LossRegistry`
3. Usable immediately via `trainer.train(loss_name='my_loss')`

### Adding a new stimulus type
1. Subclass `BaseStimulus` in `stimulus/visual.py`
2. Implement `_generate(t_array, coords)`
3. Register in `build_stimulus_tensor()` registry dict

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| Log-space weights | Enforces `\|w\| > 0`; sign fixed by NT type (Dale's law) |
| Separate PyTorch / Jaxley backends | PyTorch for fast GPU training; Jaxley for HH biophysics |
| `BaseDynamics` interface | Easy swap of dynamics without changing network or trainer code |
| Normalised connectome schema | Dataset-agnostic downstream code; easy to add BANC, FAFB, maleCNS |
| `logs/` separate from `docs/` | Runtime artefacts never mixed with documentation |
| `scatter_add` for synaptic input | O(E) time, works on GPU, no dense adjacency matrix needed |

---

## 8. References

| Paper | Role in this project |
|---|---|
| Lappalainen et al. *Nature* 2024 (FlyVis/DMN) | Direct baseline; `VoltageModel` derived from this |
| Shiu et al. *Nature* 2024 (RF model) | Second baseline |
| Jaxley paper (Deistler et al.) | Jaxley backend |
| FlyWire FAFB v783 (Dorkenwald et al.) | Primary connectome data |
| BANC dataset | Secondary connectome data |
| Hodgkin & Huxley 1952 | `HHModel` kinetics |
