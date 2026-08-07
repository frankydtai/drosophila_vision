# Synapse Models Integration

**Date**: 2026-04-04  
**Status**: ✅ Completed

## Overview

We have successfully integrated multiple synapse models inspired by Jaxley into our PyTorch-based connectome network framework. These models provide different levels of biological realism and computational complexity.

## Available Synapse Models

### 1. Simple (Default)
**File**: Built into `network_torch.py`

Simple weight-based synapses:
```
I_syn = Σ w_ji * a_j
```

**Features**:
- No state variables
- Minimal parameters (weights only)
- Fastest computation
- Good for large-scale networks

**Use when**: Fast prototyping, large networks, synaptic dynamics not critical

---

### 2. TanhRateSynapse
**File**: `models/synapses.py`

Tanh-based rate synapse (no state):
```
I = -gS * tanh((V_pre - x_offset) * slope)
```

**Parameters**:
- `gS`: Maximal synaptic conductance (default: 1e-4)
- `x_offset`: Voltage offset for tanh (default: -70.0 mV)
- `slope`: Slope of tanh activation (default: 1.0)

**Features**:
- No state variables
- Smooth activation function
- Voltage-dependent transmission
- All parameters learnable

**Use when**: Rate-based models, smooth activation needed

---

### 3. TanhConductanceSynapse
**File**: `models/synapses.py`

Tanh synapse with conductance-based current:
```
I = tanh((V_pre - x_offset) * slope) * gS * (V_post - e_syn)
```

**Parameters**:
- `gS`: Maximal synaptic conductance (default: 1e-4 uS)
- `e_syn`: Reversal potential (default: 0.0 mV)
- `x_offset`: Voltage offset (default: -70.0 mV)
- `slope`: Slope of activation (default: 1.0)

**Features**:
- No state variables
- Includes driving force (V_post - e_syn)
- More biologically realistic than TanhRate
- All parameters learnable

**Use when**: Driving force is important, medium biological realism needed

---

### 4. IonotropicSynapse
**File**: `models/synapses.py`

Biophysical ionotropic synapse with state variable:

**State dynamics**:
```
s_inf = 1 / (1 + exp((v_th - V_pre) / delta))
tau_s = (1 - s_inf) / k_minus
ds/dt = (s_inf - s) / tau_s
```

**Current**:
```
I = gS * s * (V_post - e_syn)
```

**Parameters**:
- `gS`: Maximal conductance (default: 1e-4 uS)
- `e_syn`: Reversal potential (default: 0.0 mV)
- `k_minus`: Unbinding rate constant (default: 0.025 s^-1)
- `v_th`: Voltage threshold (default: -35.0 mV)
- `delta`: Voltage sensitivity (default: 10.0 mV)
- `s_init`: Initial state (default: 0.2)

**Features**:
- Has state variable `s` (open probability)
- Most biologically realistic
- Captures synaptic dynamics
- All parameters learnable

**Use when**: Biophysical modeling, synaptic dynamics matter, detailed simulations

**Reference**: Abbott & Marder (1998), "Modeling Small Networks"

---

## Usage

### Basic Usage

```python
from neuro_framework.connectome.loader import ConnectomeLoader
from neuro_framework.models.network_torch import ConnectomeNetwork

# Load data
loader = ConnectomeLoader.from_optic_lobe(cell_types=['T4a', 'T4b'])

# Create network with different synapse models
net_simple = ConnectomeNetwork.from_loader(
    loader, 
    dynamics='voltage',
    synapse_model='simple'  # or None
)

net_tanh = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='tanh_rate',
    learn_synapse_params=True
)

net_conductance = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='tanh_conductance',
    learn_synapse_params=True
)

net_ionotropic = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',
    learn_synapse_params=True
)
```

### Training

All synapse models support gradient-based learning:

```python
import torch

# Create network
net = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',
    learn_weights=True,
    learn_synapse_params=True
)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    activity = net(stimulus, dt=1.0)
    
    # Loss
    loss = compute_loss(activity, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

## Comparison

| Model | State Variables | Parameters per Edge | Biological Realism | Computational Cost |
|-------|----------------|---------------------|-------------------|-------------------|
| **Simple** | 0 | 1 (weight) | Low | Lowest |
| **TanhRate** | 0 | 3 (gS, x_offset, slope) | Medium | Low |
| **TanhConductance** | 0 | 4 (gS, e_syn, x_offset, slope) | Medium-High | Low |
| **Ionotropic** | 1 (s) | 5 (gS, e_syn, k_minus, v_th, delta) | High | Medium |

### Parameter Counts (Example: 77 edges)

- **Simple**: 263 parameters (93 neurons + 77 weights + 93 biases)
- **TanhRate**: 494 parameters (+231 synapse params)
- **TanhConductance**: 571 parameters (+308 synapse params)
- **Ionotropic**: 648 parameters (+385 synapse params)

## Implementation Details

### Architecture

```
neuro_framework/
├── models/
│   ├── synapses.py          # NEW: Synapse models
│   ├── network_torch.py     # UPDATED: Integrated synapse models
│   └── dynamics.py          # Neuron dynamics (unchanged)
└── notebooks/
    ├── 05_synapse_models.ipynb    # NEW: Synapse models demo
    └── test_synapse_models.py     # NEW: Test script
```

### Key Classes

**`BaseSynapse`** (Abstract base class):
- `compute_current(states, pre_voltage, post_voltage, params) -> current`
- `update_states(states, delta_t, pre_voltage, post_voltage, params) -> new_states`

**`ConnectomeNetwork`** (Updated):
- New parameter: `synapse_model` (str or BaseSynapse instance)
- New parameter: `learn_synapse_params` (bool)
- Automatically handles synapse state initialization and updates
- Backward compatible (default is 'simple')

### Integration with Network

The synapse models are integrated into the forward pass:

1. **Gather voltages**: Extract pre- and post-synaptic voltages from neuron states
2. **Update states**: If synapse has state variables, update them
3. **Compute currents**: Calculate synaptic currents based on voltages
4. **Apply weights**: Multiply by learnable weights
5. **Aggregate**: Sum currents onto post-synaptic neurons

## Testing

All synapse models have been tested:

```bash
cd /Users/lengyuner/Desktop/NIPS2026
python3 neuro_framework/notebooks/test_synapse_models.py
```

**Test Results**:
- ✅ All models forward pass correctly
- ✅ All models support gradient flow
- ✅ All models produce reasonable activity patterns
- ✅ Parameter counts match expectations

## Notebooks

### 05_synapse_models.ipynb
**Purpose**: Comprehensive demonstration of all synapse models

**Contents**:
1. Load connectome data
2. Create networks with each synapse model
3. Compare activity patterns
4. Visualize differences
5. Training example
6. Summary table

**Outputs**:
- `figures/synapse_models_comparison.png`: Activity heatmaps
- `figures/synapse_models_traces.png`: Time traces
- `figures/synapse_training_curve.png`: Training curve

### Updated Notebooks

- **00_quick_start.ipynb**: Added reference to synapse models
- **01_connectome_and_network.ipynb**: Can be updated to show synapse options
- **test_synapse_models.py**: Automated testing script

## References

1. **Jaxley Documentation**: https://jaxley.readthedocs.io/
   - TanhRateSynapse: https://jaxley.readthedocs.io/en/latest/reference/_autosummary/jaxley.synapses.TanhRateSynapse.html
   - TanhConductanceSynapse: https://jaxley.readthedocs.io/en/latest/reference/_autosummary/jaxley.synapses.TanhConductanceSynapse.html
   - IonotropicSynapse: https://jaxley.readthedocs.io/en/latest/reference/_autosummary/jaxley.synapses.IonotropicSynapse.html

2. **Abbott & Marder (1998)**: "Modeling Small Networks" in *Methods in Neuronal Modeling*, MIT Press.

## Future Work

Potential extensions:

1. **Additional synapse models**:
   - NMDA receptors (voltage-dependent Mg2+ block)
   - Short-term plasticity (facilitation/depression)
   - Gap junctions (electrical synapses)

2. **Synapse-specific parameters**:
   - Different parameters for excitatory vs inhibitory synapses
   - Cell-type-specific synapse properties

3. **Learning rules**:
   - Spike-timing-dependent plasticity (STDP)
   - Hebbian learning
   - Homeostatic plasticity

4. **Performance optimization**:
   - Sparse synapse state storage
   - Batched synapse updates
   - GPU acceleration

## Summary

✅ **Completed**:
- Implemented 4 synapse models (Simple, TanhRate, TanhConductance, Ionotropic)
- Integrated into ConnectomeNetwork
- Full gradient support for all models
- Comprehensive testing and validation
- Detailed documentation and examples
- Jupyter notebook demonstration

🎯 **Key Benefits**:
- Easy to switch between models
- All models are differentiable
- Compatible with all neuron dynamics
- Flexible parameter learning
- Inspired by established Jaxley framework
