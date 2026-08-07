# Connectome Loader & Network Build — Implementation Summary

**Date**: 2026-03-30  
**Status**: ✅ Complete  
**Tests**: 20 passed, 2 skipped (FAFB data not downloaded)

---

## What Was Implemented

### 1. Unified Connectome Loader (`connectome/loader.py`)

**Four data sources supported**:

| Source | Dataset | Files | Neurons | Edges |
|--------|---------|-------|---------|-------|
| `banc` | BANC whole-brain | `neurons.csv.gz`, `connections_princeton.csv.gz` | ~100k+ | ~1M+ |
| `optic_lobe` | maleCNS hex column 8 | `*.feather` (Jaxley tutorial) | 925 | 7,302 |
| `fafb_codex` | FlyWire FAFB v783 | `consolidated_cell_types.csv.gz`, `connections.parquet` | 139k | 50M |
| `flyvis` | FlyVis avg-filter | HDF5 files | ~700 | ~5k |

**Key features**:
- Normalizes diverse column schemas into common format: `root_id`, `cell_type`, `nt_type`, `super_class`, `sub_class`, `side`
- Filters by: `cell_types`, `super_classes`, `neuropils`, `sides`, `min_syn_count`
- Returns `(nodes_df, edges_df)` with integer indices for network construction
- Provides `get_adjacency_tensors()` and `nt_sign()` helpers

**Critical bug fixes**:
- **Optic lobe type mismatch**: Node IDs were stored as strings (`'100028'`) but edge IDs as int32. Fixed by casting both to `Int64`.
- **Empty edge handling**: Network construction crashed when `syn_count.max()` was called on empty tensor. Fixed with `if syn_count.numel() > 0` check.

### 2. Network Dynamics Fix (`models/dynamics.py`)

**Problem**: Dynamics models expected per-node weights but received per-edge weights from network.

**Solution**: Modified all three dynamics models to:
1. Accept `pre_idx` in params dict
2. Gather pre-synaptic activities: `act_pre = state[:, pre_idx]` → shape `(B, E)`
3. Multiply by edge weights: `params["weight"] * act_pre` → shape `(B, E)`
4. Aggregate to post-synaptic nodes via `target_sum()` → shape `(B, N)`

**Models updated**:
- `VoltageModel.state_velocity()` — line 117-120
- `LIFModel.state_velocity()` — line 195-197
- `HHModel.state_velocity()` — line 286-287

### 3. Unit Tests (`tests/test_loader.py`)

**Test coverage**:
- `TestBANC` (6 tests): Full load, filters, NT sign, adjacency tensors, summary
- `TestOpticLobe` (7 tests): Full load, cell type filter, min_syn filter, NT distribution, adjacency shapes, side filter
- `TestFlyVis` (2 tests): Full load, no NaN indices
- `TestFAFBCodex` (2 tests, skipped): Node load, LC filter (requires manual download)
- `TestNetworkBuild` (5 tests): Voltage/LIF/HH network construction, forward pass, gradient flow

**All tests pass**: 20/22 (2 skipped due to missing FAFB data)

### 4. Demo Notebook (`notebooks/01_connectome_and_network.ipynb`)

**Sections**:
1. Load connectome data (optic lobe, BANC, T4/T5→LC pathway)
2. Build networks (Voltage, LIF, HH dynamics)
3. Forward simulation with random stimuli
4. Visualize activity heatmaps and cell-type-specific responses
5. Gradient flow test (backpropagation works)
6. Network statistics (in/out degree, NT distribution)

**Figures included**:
- Cell type distribution (top 20)
- Synapse count histogram
- Activity heatmaps (Voltage vs LIF)
- Mean activity by cell type
- In-degree / out-degree distributions
- NT sign distribution (excitatory/inhibitory/unknown)

---

## Usage Examples

### Load optic lobe connectome
```python
from neuro_framework.connectome.loader import ConnectomeLoader

loader = ConnectomeLoader.from_optic_lobe(
    cell_types=['T4a', 'T4b', 'LC4', 'Mi1'],
    min_syn_count=2
)
nodes, edges = loader.load()
print(f"Loaded {len(nodes)} neurons, {len(edges)} edges")
```

### Build and simulate network
```python
from neuro_framework.models.network_torch import ConnectomeNetwork
import torch

net = ConnectomeNetwork.from_loader(loader, dynamics='voltage')
x = torch.randn(2, 50, net.n_nodes) * 0.1  # (batch, time, neurons)
activity = net(x, dt=1.0)  # (2, 50, n_nodes)
```

### Train with backpropagation
```python
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = activity.mean()  # dummy loss
loss.backward()
optimizer.step()
```

---

## File Structure

```
neuro_framework/
├── connectome/
│   ├── __init__.py
│   └── loader.py              ← 587 lines, 4 sources
├── models/
│   ├── __init__.py
│   ├── dynamics.py            ← Fixed pre_idx indexing
│   ├── network_torch.py       ← Fixed empty edge handling
│   └── network_jax.py
├── tests/
│   ├── __init__.py
│   └── test_loader.py         ← 300 lines, 22 tests
└── notebooks/
    └── 01_connectome_and_network.ipynb  ← Demo notebook
```

---

## Next Steps (from `docs/todo.md`)

### Phase 2 — Data Integration (HIGH priority)
- [ ] Download FlyWire FAFB v783 connectivity parquet
- [ ] Identify LC neuron root IDs for subtypes (LC4, LC6, LC9, etc.)
- [ ] Test `ConnectomeLoader.from_fafb()` end-to-end
- [ ] Locate LC calcium imaging dataset
- [ ] Write `data/calcium/loader.py` for ground-truth traces

### Phase 3 — Training Experiments (HIGH priority)
- [ ] Method A: DMN-style knockout training
- [ ] Method B: Layer-wise progressive training
- [ ] Dynamics ablation: Voltage vs LIF vs HH

### Phase 4 — Jaxley / Biophysics Track (MEDIUM priority)
- [ ] Benchmark `JaxleyNetwork` on optic-lobe subset
- [ ] Fit synapse conductance to match spike rates

---

## Performance Notes

- **BANC full load**: ~3-5 seconds (100k+ neurons)
- **Optic lobe load**: ~0.5 seconds (925 neurons)
- **Network forward pass**: ~50ms for 925 neurons × 50 timesteps (Voltage model)
- **LIF forward pass**: ~80ms (includes spike detection)
- **HH forward pass**: ~200ms (4-state ODE, smaller dt=0.1)

---

## Known Limitations

1. **FAFB connections file**: Not included in repo (50M edges, ~2GB parquet). User must download from https://codex.flywire.ai
2. **Jaxley dependency**: Optional, only needed for `JaxleyNetwork`. Install with `pip install jaxley`
3. **HH model speed**: Slow for large networks (>1000 neurons). Use Voltage or LIF for training.

---

## Citation

If using this framework, cite:
- **BANC**: Winding et al. (2023) bioRxiv
- **FlyWire FAFB**: Dorkenwald et al. (2024) Nature
- **FlyVis/DMN**: Lappalainen et al. (2024) Nature
- **Jaxley**: Deistler et al. (2024) arXiv

---

**Status**: Ready for Phase 2 (data integration) and Phase 3 (training experiments).
