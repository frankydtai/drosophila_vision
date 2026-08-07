# neuro_framework

> Connectome-constrained neural network framework for the Drosophila visual system  
> Target: **NeurIPS 2026**

---

## Overview

This framework integrates:

| Component | Description |
|-----------|-------------|
| **Connectome loader** | Unified I/O for BANC and FAFB (FlyWire) datasets |
| **Neuron dynamics** | Voltage/DMN, LIF, Hodgkin-Huxley (PyTorch) |
| **Synapse models** | Simple, TanhRate, TanhConductance, Ionotropic (Jaxley-inspired) ⭐ NEW |
| **Network (PyTorch)** | Connectome-constrained recurrent network, fully differentiable |
| **Network (Jaxley)** | Biophysical multi-compartment network via JAX |
| **Stimuli** | Flash, moving bar, moving edge, sinusoidal grating |
| **Training** | Supervised, DMN-style knockout (Method A), layer-wise progressive (Method B) |

---

## Architecture

```
Visual Input (Retina / Visual Scene)
        ↓
  [stimulus/visual.py]  →  (batch, T, n_nodes) tensor
        ↓
  [connectome/loader.py]  →  nodes_df, edges_df  (BANC or FAFB)
        ↓
  [models/network_torch.py OR models/network_jax.py]
  ┌──────────────────────────────────────────────┐
  │  ConnectomeNetwork / JaxleyNetwork           │
  │  Photoreceptors → T4/T5 → Tm → LC → DN      │
  │  Dynamics: VoltageModel | LIFModel | HHModel │
  └──────────────────────────────────────────────┘
        ↓
  Neural activity  (batch, T, n_nodes)
        ↓
  [training/losses.py]  →  MSE / correlation / DSI / knockout loss
        ↓
  [training/trainer.py]  →  TorchTrainer | JaxTrainer
        ↓
  Back-propagation  (autograd / JAX value_and_grad)
```

---

## Directory Structure

```
neuro_framework/
├── __init__.py
├── connectome/
│   ├── __init__.py
│   └── loader.py          # ConnectomeLoader (BANC / FAFB)
├── models/
│   ├── __init__.py
│   ├── dynamics.py        # VoltageModel, LIFModel, HHModel
│   ├── synapses.py        # TanhRate, TanhConductance, Ionotropic ⭐ NEW
│   ├── network_torch.py   # ConnectomeNetwork (PyTorch)
│   └── network_jax.py     # JaxleyNetwork (JAX/Jaxley)
├── stimulus/
│   ├── __init__.py
│   └── visual.py          # Flash, MovingBar, MovingEdge, Grating
├── training/
│   ├── __init__.py
│   ├── losses.py          # MSE, correlation, DSI, knockout losses
│   └── trainer.py         # TorchTrainer, JaxTrainer
└── utils/
    └── __init__.py
```

---

## Quick Start

### 1. Load connectome data

```python
from neuro_framework.connectome import ConnectomeLoader

# Load ALL neurons (no filter)
loader = ConnectomeLoader.from_banc(min_syn_count=5)
nodes, edges = loader.load()
# Result: 115,151 neurons, 1,373,303 edges

# FAFB whole-brain
loader = ConnectomeLoader.from_fafb(
    data_dir="/path/to/fafb/data",
    min_syn_count=5
)
nodes, edges = loader.load()
# Result: 138,043 neurons, 2,699,071 edges

# Filter by super_class (recommended)
loader = ConnectomeLoader.from_fafb(
    data_dir="/path/to/fafb/data",
    super_classes=['optic'],  # Visual system only
    min_syn_count=5
)

# Filter by cell types (for specific pathways)
loader = ConnectomeLoader.from_banc(
    cell_types=['T4a', 'T4b', 'T4c', 'T4d', 'LC4', 'LC6'],
    min_syn_count=5
)
nodes, edges = loader.load()
print(f"{len(nodes)} neurons, {len(edges)} synapses")
```

### 2. Build a network (PyTorch backend)

```python
from neuro_framework.models import ConnectomeNetwork

# Voltage / DMN-style model (default, matches FlyVis paper)
net = ConnectomeNetwork.from_loader(loader, dynamics='voltage')

# LIF spiking network
net = ConnectomeNetwork.from_loader(loader, dynamics='lif')

# Hodgkin-Huxley
net = ConnectomeNetwork.from_loader(loader, dynamics='hh')

# With advanced synapse models ⭐ NEW
net = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',  # or 'tanh_rate', 'tanh_conductance', 'simple'
    learn_synapse_params=True
)

print(net)  # shows n_nodes, n_edges, synapse model, trainable_params
```

### 3. Build a network (Jaxley backend)

```python
from neuro_framework.models import JaxleyNetwork

# Requires: pip install jaxley optax
net_jax = JaxleyNetwork.from_loader(loader, channel='hh', dt=0.025)
params  = net_jax.get_parameters()
```

### 4. Create visual stimuli

```python
from neuro_framework.stimulus import hex_grid_coords, build_stimulus_tensor

coords = hex_grid_coords(radius=5)   # 61-node hex grid

# Moving bar: (batch=4, T=500, n_nodes=61)
x = build_stimulus_tensor(
    'moving_bar',
    coords=coords,
    batch_size=4,
    t_end=500.0, dt=1.0,
    direction_deg=0.0,
    speed=0.05,
)

# Flash
x_flash = build_stimulus_tensor(
    'flash', coords=coords, batch_size=4,
    flash_on=100.0, flash_off=200.0, t_end=500.0,
)
```

### 5. Train (Method A – DMN-style, PyTorch)

```python
from neuro_framework.training import TorchTrainer
import torch

trainer = TorchTrainer(net, optimizer_cfg={'lr': 1e-3})

# y_train: ground-truth calcium / voltage traces  (batch, T, n_output)
history = trainer.train(
    x_train, y_train,
    n_epochs=200,
    batch_size=4,
    loss_name='correlation',
    log_every=20,
)
```

### 6. Train (Method A – with knockout constraint)

```python
import torch

# Zero out T4 neurons in the knockout run
ko_mask = torch.zeros(net.n_nodes, dtype=torch.bool)
ko_mask[t4_indices] = True
x_knockout = x_train.clone()
x_knockout[:, :, ko_mask] = 0.0

loss, breakdown = trainer.step_with_knockout(
    x_full=x_train[0:4],
    x_knockout=x_knockout[0:4],
    y=y_train[0:4],
    knockout_node_mask=ko_mask,
    supervised_weight=1.0,
    knockout_weight=0.5,
)
```

### 7. Train (Method B – layer-wise progressive)

```python
# Define upstream → downstream parameter groups
layer_groups = [
    [net.log_weight_abs],    # all synapses (first pass)
    [net.bias],              # biases
    [net.log_tau],           # time constants
]

history = trainer.layerwise_train(
    layer_groups,
    x_train, y_train,
    n_epochs_per_layer=100,
    loss_name='mse',
)
```

### 8. Train with Jaxley (JAX + Optax)

```python
from neuro_framework.training import JaxTrainer
import jax.numpy as jnp

trainer_jax = JaxTrainer(net_jax, lr=1e-3)

stimuli = [net_jax.make_current_clamp(
    node_indices=photoreceptor_idx,
    i_amp=0.1, t_start=50.0, t_end=200.0
)]

params, history = trainer_jax.train(
    stimuli=stimuli,
    targets=jnp.array(y_target),
    t_max=500.0,
    n_steps=1000,
    log_every=100,
)
```

---

## Neuron Dynamics Models

| Model | Class | State variables | Key parameters | Reference |
|-------|-------|-----------------|----------------|-----------|
| **Voltage / DMN** | `VoltageModel` | `activity` | `time_const`, `bias`, `weight` | FlyVis / DMN (Lappalainen et al. 2024) |
| **LIF** | `LIFModel` | `v`, `z`, `ref` | `tau_m`, `v_thresh`, `v_rest` | Standard LIF |
| **Hodgkin-Huxley** | `HHModel` | `v`, `m`, `h`, `n` | `g_Na`, `g_K`, `g_L`, `C_m` | Hodgkin & Huxley 1952 |

---

## Synapse Models ⭐ NEW

| Model | State Variables | Parameters | Biological Realism | Use Case |
|-------|----------------|------------|-------------------|----------|
| **Simple** | None | `weight` | Low | Fast prototyping, large networks |
| **TanhRate** | None | `gS`, `x_offset`, `slope` | Medium | Rate-based models |
| **TanhConductance** | None | `gS`, `e_syn`, `x_offset`, `slope` | Medium-High | Conductance-based models |
| **Ionotropic** | `s` (open prob) | `gS`, `e_syn`, `k_minus`, `v_th`, `delta` | High | Biophysical modeling |

**Reference**: Jaxley (https://jaxley.readthedocs.io/), Abbott & Marder (1998)

---

## Loss Functions

| Loss | Function | Use case |
|------|----------|----------|
| `mse` | Mean squared error | General supervised |
| `correlation` | 1 − Pearson r | Match activity traces |
| `spike_rate` | MSE on mean rates | Firing rate targets |
| `direction_selectivity` | DSI loss | T4/T5 direction tuning |
| `knockout` | Cosine similarity penalty | DMN method A silencing |

---

## Dependencies

```
# Core (always required)
torch >= 2.0
numpy
pandas
pyarrow          # for .parquet FAFB files

# Jaxley backend (optional)
jax
jaxlib
jaxley           # pip install git+https://github.com/jaxleyverse/jaxley.git
optax            # pip install optax

# FlyVis / DMN reference (optional)
# see flyvis/ subdirectory
```

---

## Documentation

### Core Features
- **[All Neurons Support](docs/all_neurons_support.md)** ⭐ — How to load and model all neurons (BANC: 115k, FAFB: 138k)
- **[Synapse Models Integration](docs/synapse_models_integration.md)** ⭐ NEW — TanhRate, TanhConductance, Ionotropic synapses
- [Implementation Summary](docs/implementation_summary.md) — Technical implementation details
- [Architecture](docs/architecture.md) — Project architecture and design
- [TODO](docs/todo.md) — Development roadmap

### Visualization Reports
- [Full Optic Lobe Visualization](docs/full_connectome_visualization_report.md) — 925 neurons, 25 types
- [FAFB Full Brain Visualization](docs/fafb_full_brain_report.md) — 138k neurons, 8.7k types
- [Network Visualization Report](docs/network_visualization_report.md) — Type-to-type connectivity graphs
- [All Visualizations Summary](docs/ALL_VISUALIZATION_COMPLETE.md) — Comprehensive report

### Testing & Debugging
- [Notebook Debug Report](docs/notebook_debug_report.md) — LIF model fixes
- [Phase 1 Complete](docs/PHASE1_COMPLETE.md) — Framework foundation status

---

## References

- Lappalainen et al. *Nature* 2024 — FlyVis / DMN (connectome-constrained networks)
- Shiu et al. *Nature* 2024 — RF Model baseline
- Jaxley paper — differentiable biophysical simulation
- FlyWire FAFB v783 — connectome data source
- BANC dataset — Bristle-Associated Nerve Cord connectome

---

*Last updated: 2026-03-30*  
*Project: NeurIPS 2026 — Drosophila Visual System Modelling*
