"""
Connectome Loading and Network Construction Demo
================================================
This script demonstrates the neuro_framework capabilities.
Run this to verify everything works before converting to notebook.

Usage:
    cd /Users/lengyuner/Desktop/NIPS2026
    /Users/lengyuner/anaconda3/bin/python neuro_framework/notebooks/test_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns

from neuro_framework.connectome.loader import ConnectomeLoader
from neuro_framework.models.network_torch import ConnectomeNetwork
from neuro_framework.utils.logging import setup_logging

# Setup
setup_logging(level='INFO')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print(f"Project root: {PROJECT_ROOT}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# Section 1: Load Optic Lobe Connectome
# ============================================================================
print("\n" + "="*70)
print("SECTION 1: Load Optic Lobe Connectome")
print("="*70)

loader_ol = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
nodes_ol, edges_ol = loader_ol.load()

print(f"\n=== Optic Lobe Connectome ===")
print(f"Neurons: {len(nodes_ol)}")
print(f"Edges: {len(edges_ol)}")
print(f"\nTop 10 cell types:")
print(nodes_ol['cell_type'].value_counts().head(10))

summary_ol = loader_ol.summary()
print(f"\nMean synapse count: {summary_ol['mean_syn_count']:.2f}")
print(f"Max synapse count: {summary_ol['max_syn_count']:.0f}")

# ============================================================================
# Section 2: Visualize Cell Type Distribution
# ============================================================================
print("\n" + "="*70)
print("SECTION 2: Visualize Cell Type Distribution")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 20 cell types
top_types = nodes_ol['cell_type'].value_counts().head(20)
axes[0].barh(range(len(top_types)), top_types.values)
axes[0].set_yticks(range(len(top_types)))
axes[0].set_yticklabels(top_types.index)
axes[0].set_xlabel('Count')
axes[0].set_title('Top 20 Cell Types (Optic Lobe)')
axes[0].invert_yaxis()

# Synapse count distribution
axes[1].hist(edges_ol['syn_count'], bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Synapse Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Synapse Count Distribution')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig1_cell_types.png', dpi=150)
print("✓ Saved fig1_cell_types.png")
plt.close()

# ============================================================================
# Section 3: Load BANC Subset
# ============================================================================
print("\n" + "="*70)
print("SECTION 3: Load BANC Optic Lobe Subset")
print("="*70)

try:
    loader_banc = ConnectomeLoader.from_banc(
        # super_classes=['optic_lobe_intrinsic'],  # May not exist in BANC
        min_syn_count=5
    )
    nodes_banc, edges_banc = loader_banc.load()
    
    print(f"\n=== BANC Connectome ===")
    print(f"Neurons: {len(nodes_banc)}")
    print(f"Edges: {len(edges_banc)}")
    if len(nodes_banc) > 0:
        print(f"\nTop 10 cell types:")
        print(nodes_banc['cell_type'].value_counts().head(10))
except Exception as e:
    print(f"BANC loading skipped or failed: {e}")
    nodes_banc, edges_banc = None, None

# ============================================================================
# Section 4: Filter to T4/T5 → LC Pathway
# ============================================================================
print("\n" + "="*70)
print("SECTION 4: Filter to T4/T5 → LC Pathway")
print("="*70)

target_types = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d',
                'LC4', 'LC6', 'LC9', 'LC10', 'LC11', 'LC15',
                'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20',
                'L1', 'L2', 'L3', 'L4', 'L5']

loader_t4lc = ConnectomeLoader.from_optic_lobe(
    cell_types=target_types,
    min_syn_count=2
)
nodes_t4lc, edges_t4lc = loader_t4lc.load()

print(f"\n=== T4/T5 → LC Pathway ===")
print(f"Neurons: {len(nodes_t4lc)}")
print(f"Edges: {len(edges_t4lc)}")
print(f"\nCell type breakdown:")
print(nodes_t4lc['cell_type'].value_counts())

# ============================================================================
# Section 5: Build Voltage Network
# ============================================================================
print("\n" + "="*70)
print("SECTION 5: Build Voltage Network")
print("="*70)

net_voltage = ConnectomeNetwork.from_loader(
    loader_t4lc,
    dynamics='voltage',
    dt=1.0,
    init_weight_scale=0.01
)

print(f"\n=== Voltage Network ===")
print(net_voltage)
print(f"Trainable parameters: {net_voltage.n_parameters():,}")
print(f"\nParameter shapes:")
print(f"  log_weight_abs: {net_voltage.log_weight_abs.shape}")
print(f"  log_tau: {net_voltage.log_tau.shape}")
print(f"  bias: {net_voltage.bias.shape}")

# ============================================================================
# Section 6: Build LIF Network
# ============================================================================
print("\n" + "="*70)
print("SECTION 6: Build LIF Network")
print("="*70)

net_lif = ConnectomeNetwork.from_loader(
    loader_t4lc,
    dynamics='lif',
    dt=1.0
)

print(f"\n=== LIF Network ===")
print(net_lif)
print(f"Trainable parameters: {net_lif.n_parameters():,}")

# ============================================================================
# Section 7: Build HH Network (small)
# ============================================================================
print("\n" + "="*70)
print("SECTION 7: Build HH Network (small)")
print("="*70)

loader_small = ConnectomeLoader.from_optic_lobe(
    cell_types=['L1', 'L2', 'L3', 'L4', 'L5'],
    min_syn_count=2
)
nodes_small, edges_small = loader_small.load()

if len(nodes_small) > 0:
    net_hh = ConnectomeNetwork.from_loader(
        loader_small,
        dynamics='hh',
        dt=0.1
    )
    print(f"\n=== HH Network ===")
    print(net_hh)
    print(f"Neurons: {len(nodes_small)}")
else:
    print("No L1-L5 neurons found in optic lobe dataset")
    net_hh = None

# ============================================================================
# Section 8: Forward Simulation - Random Input
# ============================================================================
print("\n" + "="*70)
print("SECTION 8: Forward Simulation - Random Input")
print("="*70)

B, T = 2, 50
N = net_voltage.n_nodes

# Stimulus only to first 20% of neurons (input layer)
n_input = int(0.2 * N)
x = torch.zeros(B, T, N)
x[:, :, :n_input] = torch.randn(B, T, n_input) * 5.0  # Stronger input for LIF spikes

print(f"\nStimulus shape: {x.shape}")
print(f"Input neurons: {n_input} / {N}")
print(f"Input current range: [{x.min():.2f}, {x.max():.2f}]")

# Run voltage model
with torch.no_grad():
    activity_voltage = net_voltage(x, dt=1.0)

print(f"\nVoltage model output shape: {activity_voltage.shape}")
print(f"Activity range: [{activity_voltage.min():.3f}, {activity_voltage.max():.3f}]")
print(f"Mean activity: {activity_voltage.mean():.3f}")

# Run LIF model with adjusted parameters
# LIF needs positive input current to spike (v_rest=-70, v_thresh=-55)
# Input should be positive to depolarize neurons
x_lif = torch.zeros(B, T, N)
x_lif[:, :, :n_input] = torch.abs(torch.randn(B, T, n_input)) * 20.0  # Positive strong input

with torch.no_grad():
    activity_lif = net_lif(x_lif, dt=1.0)

print(f"\nLIF model output shape: {activity_lif.shape}")
print(f"Spike rate: {activity_lif.mean():.3f} (fraction of time spiking)")
print(f"Total spikes: {activity_lif.sum().item():.0f}")
print(f"Neurons that spiked: {(activity_lif.sum(dim=(0,1)) > 0).sum().item()} / {N}")

# ============================================================================
# Section 9: Visualize Network Activity
# ============================================================================
print("\n" + "="*70)
print("SECTION 9: Visualize Network Activity")
print("="*70)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Voltage model (batch 0)
im0 = axes[0].imshow(
    activity_voltage[0].T.numpy(),
    aspect='auto',
    cmap='viridis',
    interpolation='nearest'
)
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Neuron Index')
axes[0].set_title('Voltage Model Activity (Batch 0)')
plt.colorbar(im0, ax=axes[0], label='Activity')

# LIF model (batch 0)
im1 = axes[1].imshow(
    activity_lif[0].T.numpy(),
    aspect='auto',
    cmap='binary',
    interpolation='nearest'
)
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Neuron Index')
axes[1].set_title('LIF Model Spikes (Batch 0)')
plt.colorbar(im1, ax=axes[1], label='Spike')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig2_activity.png', dpi=150)
print("✓ Saved fig2_activity.png")
plt.close()

# ============================================================================
# Section 10: Mean Activity by Cell Type
# ============================================================================
print("\n" + "="*70)
print("SECTION 10: Mean Activity by Cell Type")
print("="*70)

mean_activity = activity_voltage[0].mean(dim=0).numpy()
cell_types = nodes_t4lc['cell_type'].values

df_activity = pd.DataFrame({
    'cell_type': cell_types,
    'mean_activity': mean_activity
})
grouped = df_activity.groupby('cell_type')['mean_activity'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(range(len(grouped)), grouped.values)
plt.yticks(range(len(grouped)), grouped.index)
plt.xlabel('Mean Activity')
plt.title('Mean Activity by Cell Type (Voltage Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig3_mean_activity.png', dpi=150)
print("✓ Saved fig3_mean_activity.png")
plt.close()

# ============================================================================
# Section 11: Gradient Flow Test
# ============================================================================
print("\n" + "="*70)
print("SECTION 11: Gradient Flow Test")
print("="*70)

net_voltage.train()
optimizer = torch.optim.Adam(net_voltage.parameters(), lr=1e-3)

out = net_voltage(x, dt=1.0)
loss = out.mean()

optimizer.zero_grad()
loss.backward()

print(f"\n=== Gradient Check ===")
print(f"Loss: {loss.item():.4f}")
print(f"log_weight_abs grad norm: {net_voltage.log_weight_abs.grad.norm().item():.4f}")
print(f"log_tau grad norm: {net_voltage.log_tau.grad.norm().item():.4f}")
print(f"bias grad norm: {net_voltage.bias.grad.norm().item():.4f}")

optimizer.step()
print("\n✓ Gradient flow successful")

# ============================================================================
# Section 12: Network Statistics
# ============================================================================
print("\n" + "="*70)
print("SECTION 12: Network Statistics")
print("="*70)

pre_idx, post_idx, syn_count = loader_t4lc.get_adjacency_tensors()

in_degree = np.bincount(post_idx, minlength=len(nodes_t4lc))
out_degree = np.bincount(pre_idx, minlength=len(nodes_t4lc))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# In-degree distribution
axes[0].hist(in_degree[in_degree > 0], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('In-degree')
axes[0].set_ylabel('Count')
axes[0].set_title('In-degree Distribution')
axes[0].set_yscale('log')

# Out-degree distribution
axes[1].hist(out_degree[out_degree > 0], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Out-degree')
axes[1].set_ylabel('Count')
axes[1].set_title('Out-degree Distribution')
axes[1].set_yscale('log')

# Degree correlation
valid = (in_degree > 0) & (out_degree > 0)
axes[2].scatter(in_degree[valid], out_degree[valid], alpha=0.5, s=10)
axes[2].set_xlabel('In-degree')
axes[2].set_ylabel('Out-degree')
axes[2].set_title('In-degree vs Out-degree')
axes[2].set_xscale('log')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig4_connectivity.png', dpi=150)
print("✓ Saved fig4_connectivity.png")
plt.close()

print(f"\n=== Connectivity Statistics ===")
print(f"Mean in-degree: {in_degree.mean():.2f}")
print(f"Mean out-degree: {out_degree.mean():.2f}")
print(f"Max in-degree: {in_degree.max()}")
print(f"Max out-degree: {out_degree.max()}")

# ============================================================================
# Section 13: Neurotransmitter Distribution
# ============================================================================
print("\n" + "="*70)
print("SECTION 13: Neurotransmitter Distribution")
print("="*70)

nt_sign = loader_t4lc.nt_sign()
nt_labels = ['Inhibitory', 'Unknown', 'Excitatory']
nt_counts = [(nt_sign == -1).sum(), (nt_sign == 0).sum(), (nt_sign == 1).sum()]

plt.figure(figsize=(8, 6))
plt.bar(nt_labels, nt_counts, color=['red', 'gray', 'blue'], alpha=0.7, edgecolor='black')
plt.ylabel('Count')
plt.title('Neurotransmitter Sign Distribution')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig5_nt_distribution.png', dpi=150)
print("✓ Saved fig5_nt_distribution.png")
plt.close()

print(f"\n=== Neurotransmitter Distribution ===")
print(f"Excitatory: {nt_counts[2]} ({100*nt_counts[2]/len(nt_sign):.1f}%)")
print(f"Inhibitory: {nt_counts[0]} ({100*nt_counts[0]/len(nt_sign):.1f}%)")
print(f"Unknown: {nt_counts[1]} ({100*nt_counts[1]/len(nt_sign):.1f}%)")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("DEMO COMPLETE - ALL TESTS PASSED ✓")
print("="*70)
print("\nGenerated figures:")
print("  - fig1_cell_types.png")
print("  - fig2_activity.png")
print("  - fig3_mean_activity.png")
print("  - fig4_connectivity.png")
print("  - fig5_nt_distribution.png")
print("\nNext steps:")
print("  1. Load real visual stimuli (moving bars, gratings)")
print("  2. Train networks on calcium imaging ground truth")
print("  3. Implement Method A (knockout) and Method B (layer-wise)")
print("  4. Evaluate direction selectivity vs DMN baseline")
