# %% [markdown]
# # Connectome Data Loading - Quick Start Guide
# 
# **Purpose**: Simple examples of loading and exploring connectome data  
# **Date**: 2026-04-04
# 
# This notebook demonstrates:
# 1. Loading data from different sources (BANC, FAFB, Optic Lobe)
# 2. Basic data exploration with print statements
# 3. Simple statistics and summaries

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.insert(0, '/Users/lengyuner/Desktop/NIPS2026')

import pandas as pd
import numpy as np
from neuro_framework.connectome.loader import ConnectomeLoader

# Set pandas display options for better readability
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

# %% [markdown]
# ## Example 1: Load BANC Whole-Brain Connectome
# 
# BANC (Bristle-Associated Nerve Cord) contains ~115k neurons from the whole brain.

# %%
print("="*70)
print("Example 1: BANC Whole-Brain Connectome")
print("="*70)
print()

# Load BANC data with minimum synapse threshold
loader_banc = ConnectomeLoader.from_banc(min_syn_count=5)
nodes_banc, edges_banc = loader_banc.load()

# Basic statistics
print(f"📊 Dataset Statistics:")
print(f"  Neurons: {len(nodes_banc):,}")
print(f"  Edges: {len(edges_banc):,}")
print(f"  Cell types: {nodes_banc['cell_type'].nunique():,}")
print(f"  Super classes: {nodes_banc['super_class'].nunique()}")
print()

# Show sample neurons
print(f"📋 Sample Neurons (first 5):")
print(nodes_banc[['node_idx', 'cell_type', 'nt_type', 'super_class', 'side']].head())
print()

# Show sample connections
print(f"🔗 Sample Connections (first 5):")
print(edges_banc[['pre_idx', 'post_idx', 'syn_count', 'neuropil']].head())
print()

# Top cell types by count
print(f"🏆 Top 10 Cell Types by Neuron Count:")
top_types = nodes_banc['cell_type'].value_counts().head(10)
for i, (cell_type, count) in enumerate(top_types.items(), 1):
    print(f"  {i:2d}. {cell_type:30s} {count:6,} neurons")
print()

# Neurotransmitter distribution
print(f"💊 Neurotransmitter Distribution:")
nt_dist = nodes_banc['nt_type'].value_counts()
for nt, count in nt_dist.items():
    pct = 100 * count / len(nodes_banc)
    print(f"  {str(nt):20s} {count:6,} neurons ({pct:5.1f}%)")
print()

# %% [markdown]
# ## Example 2: Load FAFB Visual System
# 
# FAFB (FlyWire Full Adult Fly Brain) v783 contains ~138k neurons.  
# Here we load only the visual system (optic super-class).

# %%
print("="*70)
print("Example 2: FAFB Visual System")
print("="*70)
print()

# Load FAFB visual system
loader_fafb = ConnectomeLoader.from_fafb(
    data_dir="/Users/lengyuner/Desktop/data/flywire/Jun2025",
    super_classes=['optic'],  # Filter to visual system only
    min_syn_count=5
)
nodes_fafb, edges_fafb = loader_fafb.load()

# Basic statistics
print(f"📊 Dataset Statistics:")
print(f"  Neurons: {len(nodes_fafb):,}")
print(f"  Edges: {len(edges_fafb):,}")
print(f"  Cell types: {nodes_fafb['cell_type'].nunique():,}")
print(f"  Super classes: {nodes_fafb['super_class'].nunique()}")
print()

# Show sample neurons
print(f"📋 Sample Neurons (first 5):")
print(nodes_fafb[['node_idx', 'cell_type', 'nt_type', 'super_class', 'side']].head())
print()

# Show sample connections
print(f"🔗 Sample Connections (first 5):")
print(edges_fafb[['pre_idx', 'post_idx', 'syn_count', 'neuropil']].head())
print()

# Top cell types
print(f"🏆 Top 10 Cell Types by Neuron Count:")
top_types_fafb = nodes_fafb['cell_type'].value_counts().head(10)
for i, (cell_type, count) in enumerate(top_types_fafb.items(), 1):
    print(f"  {i:2d}. {cell_type:30s} {count:6,} neurons")
print()

# Connection statistics
print(f"📈 Connection Statistics:")
print(f"  Mean synapses per edge: {edges_fafb['syn_count'].mean():.1f}")
print(f"  Median synapses per edge: {edges_fafb['syn_count'].median():.1f}")
print(f"  Max synapses: {edges_fafb['syn_count'].max():,}")
print(f"  Total synapses: {edges_fafb['syn_count'].sum():,}")
print()

# %% [markdown]
# ## Example 3: Load Optic Lobe Subset
# 
# Optic lobe dataset from maleCNS (925 neurons, 25 cell types).  
# This is a smaller, well-characterized dataset good for testing.

# %%
print("="*70)
print("Example 3: Optic Lobe Subset")
print("="*70)
print()

# Load optic lobe data
loader_ol = ConnectomeLoader.from_optic_lobe()
nodes_ol, edges_ol = loader_ol.load()

# Basic statistics
print(f"📊 Dataset Statistics:")
print(f"  Neurons: {len(nodes_ol):,}")
print(f"  Edges: {len(edges_ol):,}")
print(f"  Cell types: {nodes_ol['cell_type'].nunique()}")
print(f"  Super classes: {nodes_ol['super_class'].nunique()}")
print()

# Show all cell types (small dataset)
print(f"📋 All Cell Types:")
cell_type_counts = nodes_ol['cell_type'].value_counts()
for i, (cell_type, count) in enumerate(cell_type_counts.items(), 1):
    print(f"  {i:2d}. {cell_type:20s} {count:4,} neurons")
print()

# Show sample neurons
print(f"📋 Sample Neurons (first 5):")
print(nodes_ol[['node_idx', 'cell_type', 'nt_type', 'super_class', 'side']].head())
print()

# Show sample connections
print(f"🔗 Sample Connections (first 5):")
print(edges_ol[['pre_idx', 'post_idx', 'syn_count']].head())
print()

# %% [markdown]
# ## Example 4: Filter by Specific Cell Types
# 
# Load only T4/T5 direction-selective neurons and their inputs.

# %%
print("="*70)
print("Example 4: T4/T5 Pathway (Filtered)")
print("="*70)
print()

# Load specific cell types
loader_t4t5 = ConnectomeLoader.from_optic_lobe(
    cell_types=['T4a', 'T4b', 'T4c', 'T4d',
                'T5a', 'T5b', 'T5c', 'T5d',
                'Mi1', 'Mi4', 'Mi9',
                'Tm1', 'Tm2', 'Tm3',
                'L1', 'L2', 'L3', 'L4', 'L5'],
    min_syn_count=5
)
nodes_t4t5, edges_t4t5 = loader_t4t5.load()

# Basic statistics
print(f"📊 Dataset Statistics:")
print(f"  Neurons: {len(nodes_t4t5):,}")
print(f"  Edges: {len(edges_t4t5):,}")
print(f"  Cell types: {nodes_t4t5['cell_type'].nunique()}")
print()

# Cell type breakdown
print(f"📋 Cell Type Breakdown:")
cell_type_counts = nodes_t4t5['cell_type'].value_counts()
for cell_type, count in cell_type_counts.items():
    print(f"  {cell_type:10s} {count:4,} neurons")
print()

# Connectivity matrix (type-to-type)
print(f"🔗 Type-to-Type Connectivity (top 10 connections):")
# Merge edges with cell types
edges_with_types = edges_t4t5.merge(
    nodes_t4t5[['node_idx', 'cell_type']].rename(columns={'cell_type': 'pre_type'}),
    left_on='pre_idx', right_on='node_idx', how='left'
).merge(
    nodes_t4t5[['node_idx', 'cell_type']].rename(columns={'cell_type': 'post_type'}),
    left_on='post_idx', right_on='node_idx', how='left'
)

# Aggregate by type-to-type
type_to_type = edges_with_types.groupby(['pre_type', 'post_type'])['syn_count'].sum().reset_index()
type_to_type = type_to_type.sort_values('syn_count', ascending=False).head(10)

for i, row in enumerate(type_to_type.itertuples(), 1):
    print(f"  {i:2d}. {row.pre_type:10s} → {row.post_type:10s}  {row.syn_count:6,} synapses")
print()

# %% [markdown]
# ## Example 5: Compare Datasets
# 
# Quick comparison of all loaded datasets.

# %%
print("="*70)
print("Example 5: Dataset Comparison")
print("="*70)
print()

# Create comparison table
datasets = [
    ("BANC (whole-brain)", nodes_banc, edges_banc),
    ("FAFB (visual system)", nodes_fafb, edges_fafb),
    ("Optic Lobe (all)", nodes_ol, edges_ol),
    ("T4/T5 Pathway", nodes_t4t5, edges_t4t5),
]

print(f"{'Dataset':<25} {'Neurons':>10} {'Edges':>10} {'Cell Types':>12} {'Avg Syn/Edge':>14}")
print("-" * 75)

for name, nodes, edges in datasets:
    n_neurons = len(nodes)
    n_edges = len(edges)
    n_types = nodes['cell_type'].nunique()
    avg_syn = edges['syn_count'].mean()
    print(f"{name:<25} {n_neurons:>10,} {n_edges:>10,} {n_types:>12,} {avg_syn:>14.1f}")

print()

# %% [markdown]
# ## Summary
# 
# This notebook demonstrated:
# 
# 1. **Loading different datasets**:
#    - BANC whole-brain (115k neurons)
#    - FAFB visual system (77k neurons)
#    - Optic lobe subset (925 neurons)
#    - Filtered T4/T5 pathway
# 
# 2. **Basic exploration**:
#    - Neuron and edge counts
#    - Cell type distributions
#    - Neurotransmitter distributions
#    - Connection statistics
#    - Type-to-type connectivity
# 
# 3. **Key features**:
#    - Unified `ConnectomeLoader` interface
#    - Flexible filtering (cell types, super classes, min synapses)
#    - Consistent data format across sources
# 
# ### Next Steps
# 
# - **Build networks**: See `01_connectome_and_network.ipynb`
# - **Synapse models**: See `05_synapse_models.ipynb` ⭐ NEW
# - **Visualize networks**: See `04_network_visualization_display.ipynb`
# - **Full analysis**: See `02_full_connectome_visualization.ipynb` and `03_fafb_full_brain.ipynb`

# %%
