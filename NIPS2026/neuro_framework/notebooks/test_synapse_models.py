"""
Test Synapse Models Integration
================================
Test the new synapse models in ConnectomeNetwork.

Usage:
    cd /Users/lengyuner/Desktop/NIPS2026
    /Users/lengyuner/anaconda3/bin/python3 neuro_framework/notebooks/test_synapse_models.py
"""

import sys
sys.path.insert(0, '/Users/lengyuner/Desktop/NIPS2026')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from neuro_framework.connectome.loader import ConnectomeLoader
from neuro_framework.models.network_torch import ConnectomeNetwork

print("="*70)
print("Testing Synapse Models Integration")
print("="*70)

# Load small dataset for testing
print("\n1. Loading Optic Lobe data...")
loader = ConnectomeLoader.from_optic_lobe(
    cell_types=['T4a', 'T4b', 'T5a', 'T5b', 'Mi1', 'Tm3'],
    min_syn_count=5
)
nodes, edges = loader.load()
print(f"   Loaded {len(nodes)} neurons, {len(edges)} edges")

# Test different synapse models
synapse_models = ['simple', 'tanh_rate', 'tanh_conductance', 'ionotropic']
results = {}

for syn_model in synapse_models:
    print(f"\n2. Testing synapse model: {syn_model}")
    
    # Build network
    net = ConnectomeNetwork.from_loader(
        loader,
        dynamics='voltage',
        synapse_model=syn_model,
        learn_synapse_params=True
    )
    
    print(f"   Network: {net}")
    print(f"   Parameters: {net.n_parameters():,}")
    
    # Create stimulus
    B, T = 2, 50
    x = torch.zeros(B, T, net.n_nodes)
    x[:, :20, :5] = torch.randn(B, 20, 5) * 0.1  # Input to first 5 neurons
    
    # Forward pass
    print(f"   Running forward pass...")
    with torch.no_grad():
        activity = net(x, dt=1.0)
    
    print(f"   Output shape: {activity.shape}")
    print(f"   Activity range: [{activity.min():.4f}, {activity.max():.4f}]")
    print(f"   Mean activity: {activity.mean():.4f}")
    
    # Test gradient flow
    print(f"   Testing gradient flow...")
    activity_grad = net(x, dt=1.0)
    loss = activity_grad.mean()
    loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in net.parameters() if p.grad is not None)
    total_params = sum(1 for p in net.parameters())
    print(f"   Gradients: {has_grad}/{total_params} parameters")
    
    results[syn_model] = {
        'activity': activity.detach().cpu().numpy(),
        'n_params': net.n_parameters(),
        'mean_activity': activity.mean().item(),
    }
    
    print(f"   ✓ {syn_model} passed!")

# Compare results
print("\n" + "="*70)
print("Comparison of Synapse Models")
print("="*70)

print(f"\n{'Model':<20} {'Parameters':>12} {'Mean Activity':>15}")
print("-"*50)
for model, res in results.items():
    print(f"{model:<20} {res['n_params']:>12,} {res['mean_activity']:>15.6f}")

# Visualize activities
print("\n3. Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (model, res) in enumerate(results.items()):
    ax = axes[idx]
    activity = res['activity'][0]  # First batch
    
    im = ax.imshow(activity.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'{model.replace("_", " ").title()} Synapse', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron Index')
    plt.colorbar(im, ax=ax, label='Activity')

plt.tight_layout()
output_path = '/Users/lengyuner/Desktop/NIPS2026/neuro_framework/notebooks/figures/synapse_models_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {output_path}")

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
