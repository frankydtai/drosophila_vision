# %% [markdown]
# # Synapse Models in Connectome Networks
# 
# **Date**: 2026-04-04  
# **Purpose**: Demonstrate different synapse models inspired by Jaxley
# 
# This notebook shows how to use different synapse models in connectome networks:
# 1. **Simple**: Weight-based synapses (default)
# 2. **TanhRateSynapse**: Tanh-based rate synapse (no state)
# 3. **TanhConductanceSynapse**: Tanh with conductance-based current
# 4. **IonotropicSynapse**: Biophysical synapse with state variable
# 
# ## References
# - Jaxley: https://jaxley.readthedocs.io/
# - Abbott & Marder (1998): "Modeling Small Networks"

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.insert(0, '/Users/lengyuner/Desktop/NIPS2026')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from neuro_framework.connectome.loader import ConnectomeLoader
from neuro_framework.models.network_torch import ConnectomeNetwork

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## Load Connectome Data

# %%
print("Loading Optic Lobe data...")
loader = ConnectomeLoader.from_optic_lobe(
    cell_types=['T4a', 'T4b', 'T5a', 'T5b', 'Mi1', 'Tm3', 'L1', 'L2'],
    min_syn_count=5
)
nodes, edges = loader.load()

print(f"Loaded {len(nodes)} neurons, {len(edges)} edges")
print(f"Cell types: {nodes['cell_type'].unique()}")

# %% [markdown]
# ## 1. Simple Weight-Based Synapses (Default)
# 
# The default synapse model uses simple weighted connections:
# 
# $$I_{syn} = \sum_{j \to i} w_{ji} \cdot a_j$$
# 
# where $w_{ji}$ is the synaptic weight and $a_j$ is the pre-synaptic activity.

# %%
print("\n" + "="*70)
print("1. Simple Weight-Based Synapses")
print("="*70)

net_simple = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='simple',  # or None
)

print(f"\nNetwork: {net_simple}")
print(f"Parameters: {net_simple.n_parameters():,}")

# Create stimulus
B, T = 1, 100
x = torch.zeros(B, T, net_simple.n_nodes)
x[:, 10:30, :5] = torch.randn(B, 20, 5) * 0.2  # Pulse to first 5 neurons

# Simulate
with torch.no_grad():
    activity_simple = net_simple(x, dt=1.0)

print(f"Output shape: {activity_simple.shape}")
print(f"Activity range: [{activity_simple.min():.4f}, {activity_simple.max():.4f}]")

# %% [markdown]
# ## 2. TanhRateSynapse
# 
# Tanh-based rate synapse (no state variables):
# 
# $$I_{syn} = -g_S \cdot \tanh\left(\frac{V_{pre} - x_{offset}}{slope}\right)$$
# 
# **Parameters**:
# - $g_S$: Maximal synaptic conductance
# - $x_{offset}$: Voltage offset for tanh
# - $slope$: Slope of tanh activation

# %%
print("\n" + "="*70)
print("2. TanhRateSynapse")
print("="*70)

net_tanh_rate = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='tanh_rate',
    learn_synapse_params=True,
)

print(f"\nNetwork: {net_tanh_rate}")
print(f"Parameters: {net_tanh_rate.n_parameters():,}")

# Simulate
with torch.no_grad():
    activity_tanh_rate = net_tanh_rate(x, dt=1.0)

print(f"Output shape: {activity_tanh_rate.shape}")
print(f"Activity range: [{activity_tanh_rate.min():.4f}, {activity_tanh_rate.max():.4f}]")

# %% [markdown]
# ## 3. TanhConductanceSynapse
# 
# Tanh synapse with conductance-based current:
# 
# $$I_{syn} = \tanh\left(\frac{V_{pre} - x_{offset}}{slope}\right) \cdot g_S \cdot (V_{post} - E_{syn})$$
# 
# **Parameters**:
# - $g_S$: Maximal synaptic conductance
# - $E_{syn}$: Reversal potential
# - $x_{offset}$: Voltage offset
# - $slope$: Slope of activation

# %%
print("\n" + "="*70)
print("3. TanhConductanceSynapse")
print("="*70)

net_tanh_cond = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='tanh_conductance',
    learn_synapse_params=True,
)

print(f"\nNetwork: {net_tanh_cond}")
print(f"Parameters: {net_tanh_cond.n_parameters():,}")

# Simulate
with torch.no_grad():
    activity_tanh_cond = net_tanh_cond(x, dt=1.0)

print(f"Output shape: {activity_tanh_cond.shape}")
print(f"Activity range: [{activity_tanh_cond.min():.4f}, {activity_tanh_cond.max():.4f}]")

# %% [markdown]
# ## 4. IonotropicSynapse
# 
# Biophysical ionotropic synapse with state variable $s$ (open probability):
# 
# **State dynamics**:
# $$s_{\infty} = \frac{1}{1 + \exp\left(\frac{V_{th} - V_{pre}}{\delta}\right)}$$
# $$\tau_s = \frac{1 - s_{\infty}}{k_{minus}}$$
# $$\frac{ds}{dt} = \frac{s_{\infty} - s}{\tau_s}$$
# 
# **Current**:
# $$I_{syn} = g_S \cdot s \cdot (V_{post} - E_{syn})$$
# 
# **Parameters**:
# - $g_S$: Maximal conductance
# - $E_{syn}$: Reversal potential
# - $k_{minus}$: Unbinding rate constant
# - $V_{th}$: Voltage threshold
# - $\delta$: Voltage sensitivity

# %%
print("\n" + "="*70)
print("4. IonotropicSynapse")
print("="*70)

net_ionotropic = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',
    learn_synapse_params=True,
)

print(f"\nNetwork: {net_ionotropic}")
print(f"Parameters: {net_ionotropic.n_parameters():,}")

# Simulate
with torch.no_grad():
    activity_ionotropic = net_ionotropic(x, dt=1.0)

print(f"Output shape: {activity_ionotropic.shape}")
print(f"Activity range: [{activity_ionotropic.min():.4f}, {activity_ionotropic.max():.4f}]")

# %% [markdown]
# ## Comparison of Synapse Models

# %%
print("\n" + "="*70)
print("Comparison of Synapse Models")
print("="*70)

models = {
    'Simple': (net_simple, activity_simple),
    'TanhRate': (net_tanh_rate, activity_tanh_rate),
    'TanhConductance': (net_tanh_cond, activity_tanh_cond),
    'Ionotropic': (net_ionotropic, activity_ionotropic),
}

print(f"\n{'Model':<20} {'Parameters':>12} {'Mean Activity':>15} {'Std Activity':>15}")
print("-"*65)
for name, (net, act) in models.items():
    print(f"{name:<20} {net.n_parameters():>12,} {act.mean():>15.6f} {act.std():>15.6f}")

# %% [markdown]
# ## Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, (net, act)) in enumerate(models.items()):
    ax = axes[idx]
    
    # Plot activity heatmap
    activity_np = act[0].cpu().numpy()  # First batch
    im = ax.imshow(activity_np.T, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest', vmin=-0.5, vmax=0.5)
    
    ax.set_title(f'{name} Synapse Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Stimulus start')
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Stimulus end')
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Activity')

plt.tight_layout()
plt.savefig('figures/synapse_models_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: figures/synapse_models_comparison.png")
plt.show()

# %% [markdown]
# ## Activity Traces for Selected Neurons

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Select a few neurons to plot
neuron_indices = [0, 5, 10, 20]

for idx, (name, (net, act)) in enumerate(models.items()):
    ax = axes[idx]
    
    activity_np = act[0].cpu().numpy()  # First batch
    
    for neuron_idx in neuron_indices:
        ax.plot(activity_np[:, neuron_idx], label=f'Neuron {neuron_idx}', alpha=0.7)
    
    ax.set_title(f'{name} Synapse Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Activity', fontsize=12)
    ax.axvspan(10, 30, alpha=0.2, color='green', label='Stimulus')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/synapse_models_traces.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/synapse_models_traces.png")
plt.show()

# %% [markdown]
# ## Training with Different Synapse Models
# 
# All synapse models support gradient-based learning.

# %%
print("\n" + "="*70)
print("Training Example with Ionotropic Synapse")
print("="*70)

# Create network
net_train = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',
    learn_weights=True,
    learn_synapse_params=True,
)

# Create target activity pattern
target = torch.randn(B, T, net_train.n_nodes) * 0.1

# Optimizer
optimizer = torch.optim.Adam(net_train.parameters(), lr=1e-3)

# Training loop
losses = []
for epoch in range(50):
    optimizer.zero_grad()
    
    # Forward pass
    activity = net_train(x, dt=1.0)
    
    # Loss
    loss = ((activity - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

# Plot training curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training with Ionotropic Synapse Model', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('figures/synapse_training_curve.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: figures/synapse_training_curve.png")
plt.show()

# %% [markdown]
# ## Summary
# 
# ### Synapse Models Comparison
# 
# | Model | State Variables | Parameters | Biological Realism | Computational Cost |
# |-------|----------------|------------|-------------------|-------------------|
# | **Simple** | None | Weights only | Low | Lowest |
# | **TanhRate** | None | gS, x_offset, slope | Medium | Low |
# | **TanhConductance** | None | gS, e_syn, x_offset, slope | Medium-High | Low |
# | **Ionotropic** | s (open prob) | gS, e_syn, k_minus, v_th, delta | High | Medium |
# 
# ### When to Use Each Model
# 
# - **Simple**: Fast prototyping, large-scale networks, when synaptic dynamics are not critical
# - **TanhRate**: Rate-based models, when you need smooth activation functions
# - **TanhConductance**: When driving force (V_post - E_syn) is important
# - **Ionotropic**: Biophysical modeling, when synaptic dynamics matter, detailed simulations
# 
# ### Key Features
# 
# - ✅ All models support gradient-based learning
# - ✅ All models are fully differentiable
# - ✅ Easy to switch between models
# - ✅ Compatible with all neuron dynamics (Voltage, LIF, HH)
# - ✅ Inspired by Jaxley's synapse models

# %%
