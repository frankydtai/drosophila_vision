"""
Full Connectome Visualization
==============================
Comprehensive visualization of all neuron types in the optic lobe and BANC datasets.

This notebook provides:
1. Overview statistics for all neuron types
2. Hierarchical clustering of cell types by connectivity
3. Network graph visualization
4. Connectivity matrices
5. Functional grouping analysis
6. Comparative analysis across datasets

Usage:
    cd /Users/lengyuner/Desktop/NIPS2026
    /Users/lengyuner/anaconda3/bin/python neuro_framework/notebooks/02_full_connectome_visualization.py
"""

import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from neuro_framework.connectome.loader import ConnectomeLoader
from neuro_framework.utils.logging import setup_logging

# Setup
setup_logging(level='INFO')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

print(f"Project root: {PROJECT_ROOT}")
print("\n" + "="*80)
print("FULL CONNECTOME VISUALIZATION")
print("="*80)

# ============================================================================
# Section 1: Load Full Optic Lobe Connectome (All Neuron Types)
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: Load Full Optic Lobe Connectome")
print("="*80)

loader_full = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
nodes_full, edges_full = loader_full.load()

print(f"\n=== Full Optic Lobe Connectome ===")
print(f"Total neurons: {len(nodes_full)}")
print(f"Total edges: {len(edges_full)}")
print(f"Unique cell types: {nodes_full['cell_type'].nunique()}")
print(f"Mean synapse count: {edges_full['syn_count'].mean():.2f}")
print(f"Total synapses: {edges_full['syn_count'].sum():.0f}")

# Cell type statistics
cell_type_counts = nodes_full['cell_type'].value_counts()
print(f"\nTop 20 cell types:")
print(cell_type_counts.head(20))

# ============================================================================
# Section 2: Cell Type Overview - Comprehensive Statistics
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: Cell Type Statistics")
print("="*80)

# Calculate per-cell-type statistics
pre_idx, post_idx, syn_count = loader_full.get_adjacency_tensors()
in_degree = np.bincount(post_idx, minlength=len(nodes_full))
out_degree = np.bincount(pre_idx, minlength=len(nodes_full))

# Add degree info to nodes
nodes_full['in_degree'] = in_degree
nodes_full['out_degree'] = out_degree
nodes_full['total_degree'] = in_degree + out_degree

# Group by cell type
cell_type_stats = nodes_full.groupby('cell_type').agg({
    'root_id': 'count',
    'in_degree': ['mean', 'std', 'max'],
    'out_degree': ['mean', 'std', 'max'],
    'total_degree': ['mean', 'std', 'max'],
    'nt_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
}).round(2)

cell_type_stats.columns = ['count', 'in_mean', 'in_std', 'in_max', 
                            'out_mean', 'out_std', 'out_max',
                            'total_mean', 'total_std', 'total_max', 'nt_type']
cell_type_stats = cell_type_stats.sort_values('count', ascending=False)

print("\nCell type statistics (top 30):")
print(cell_type_stats.head(30).to_string())

# Save to CSV
csv_path = PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'cell_type_stats.csv'
cell_type_stats.to_csv(csv_path)
print(f"\n✓ Saved cell_type_stats.csv")

# ============================================================================
# Section 3: Visualization 1 - Cell Type Distribution
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: Cell Type Distribution Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 3.1 Top 30 cell types by count
top30 = cell_type_counts.head(30)
axes[0, 0].barh(range(len(top30)), top30.values, color='steelblue', alpha=0.8)
axes[0, 0].set_yticks(range(len(top30)))
axes[0, 0].set_yticklabels(top30.index, fontsize=9)
axes[0, 0].set_xlabel('Neuron Count', fontsize=11)
axes[0, 0].set_title('Top 30 Cell Types by Count', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 3.2 Cell type count distribution (log scale)
count_dist = cell_type_counts.values
axes[0, 1].hist(count_dist, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Neuron Count per Cell Type', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Cell Type Count Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# 3.3 Mean connectivity by cell type (top 30)
top30_conn = cell_type_stats.head(30)
x_pos = np.arange(len(top30_conn))
axes[1, 0].bar(x_pos - 0.2, top30_conn['in_mean'], 0.4, label='In-degree', 
               color='green', alpha=0.7)
axes[1, 0].bar(x_pos + 0.2, top30_conn['out_mean'], 0.4, label='Out-degree', 
               color='orange', alpha=0.7)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(top30_conn.index, rotation=90, fontsize=8)
axes[1, 0].set_ylabel('Mean Degree', fontsize=11)
axes[1, 0].set_title('Mean Connectivity by Cell Type (Top 30)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 3.4 Neurotransmitter distribution by cell type
nt_by_type = nodes_full.groupby(['cell_type', 'nt_type']).size().unstack(fill_value=0)
top20_types = cell_type_counts.head(20).index
nt_subset = nt_by_type.loc[top20_types]
nt_subset.plot(kind='barh', stacked=True, ax=axes[1, 1], 
               color=['red', 'gray', 'blue', 'green', 'purple'])
axes[1, 1].set_xlabel('Neuron Count', fontsize=11)
axes[1, 1].set_title('Neurotransmitter Distribution (Top 20 Cell Types)', 
                      fontsize=12, fontweight='bold')
axes[1, 1].legend(title='NT Type', fontsize=8, loc='lower right')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_full_01_overview.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_full_01_overview.png")
plt.close()

# ============================================================================
# Section 4: Connectivity Matrix Visualization
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: Connectivity Matrix")
print("="*80)

# Build connectivity matrix for top 40 cell types
top_types = cell_type_counts.head(40).index.tolist()
type_to_idx = {t: i for i, t in enumerate(top_types)}

# Filter nodes and edges to top types
nodes_top = nodes_full[nodes_full['cell_type'].isin(top_types)].copy()
nodes_top['type_idx'] = nodes_top['cell_type'].map(type_to_idx)

# Build type-to-type connectivity matrix
conn_matrix = np.zeros((len(top_types), len(top_types)))

for _, edge in edges_full.iterrows():
    pre_id = edge['pre_root_id']
    post_id = edge['post_root_id']
    
    pre_node = nodes_full[nodes_full['root_id'] == pre_id]
    post_node = nodes_full[nodes_full['root_id'] == post_id]
    
    if len(pre_node) > 0 and len(post_node) > 0:
        pre_type = pre_node.iloc[0]['cell_type']
        post_type = post_node.iloc[0]['cell_type']
        
        if pre_type in type_to_idx and post_type in type_to_idx:
            i = type_to_idx[pre_type]
            j = type_to_idx[post_type]
            conn_matrix[i, j] += edge['syn_count']

print(f"Connectivity matrix shape: {conn_matrix.shape}")
print(f"Total connections: {conn_matrix.sum():.0f}")
print(f"Non-zero entries: {(conn_matrix > 0).sum()}")

# Visualize connectivity matrix
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# 4.1 Raw connectivity matrix (log scale)
im1 = axes[0].imshow(np.log10(conn_matrix + 1), cmap='viridis', aspect='auto')
axes[0].set_xticks(range(len(top_types)))
axes[0].set_yticks(range(len(top_types)))
axes[0].set_xticklabels(top_types, rotation=90, fontsize=7)
axes[0].set_yticklabels(top_types, fontsize=7)
axes[0].set_xlabel('Post-synaptic Cell Type', fontsize=11)
axes[0].set_ylabel('Pre-synaptic Cell Type', fontsize=11)
axes[0].set_title('Connectivity Matrix (log10 synapse count)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='log10(synapse count + 1)')

# 4.2 Binary connectivity (presence/absence)
binary_conn = (conn_matrix > 0).astype(float)
im2 = axes[1].imshow(binary_conn, cmap='binary', aspect='auto')
axes[1].set_xticks(range(len(top_types)))
axes[1].set_yticks(range(len(top_types)))
axes[1].set_xticklabels(top_types, rotation=90, fontsize=7)
axes[1].set_yticklabels(top_types, fontsize=7)
axes[1].set_xlabel('Post-synaptic Cell Type', fontsize=11)
axes[1].set_ylabel('Pre-synaptic Cell Type', fontsize=11)
axes[1].set_title('Binary Connectivity Matrix', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='Connection exists')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_full_02_connectivity_matrix.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_full_02_connectivity_matrix.png")
plt.close()

# ============================================================================
# Section 5: Hierarchical Clustering of Cell Types
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: Hierarchical Clustering")
print("="*80)

# Use connectivity patterns to cluster cell types
# Distance metric: 1 - correlation of connectivity profiles
conn_profiles = conn_matrix + conn_matrix.T  # Symmetrize
conn_profiles_norm = conn_profiles / (conn_profiles.sum(axis=1, keepdims=True) + 1e-8)

# Compute pairwise distances
distances = pdist(conn_profiles_norm, metric='correlation')
linkage_matrix = linkage(distances, method='ward')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(16, 10))
dendrogram(linkage_matrix, labels=top_types, ax=ax, leaf_font_size=8)
ax.set_xlabel('Cell Type', fontsize=12)
ax.set_ylabel('Distance', fontsize=12)
ax.set_title('Hierarchical Clustering of Cell Types by Connectivity Pattern', 
             fontsize=14, fontweight='bold')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_full_03_clustering.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_full_03_clustering.png")
plt.close()

# ============================================================================
# Section 6: Functional Grouping Analysis
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: Functional Grouping")
print("="*80)

# Define functional groups based on known cell types
functional_groups = {
    'Photoreceptors': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'],
    'Lamina': ['L1', 'L2', 'L3', 'L4', 'L5', 'C2', 'C3'],
    'Medulla_input': ['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20'],
    'Direction_selective': ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'],
    'Lobula_columnar': ['LC4', 'LC6', 'LC9', 'LC10', 'LC11', 'LC15', 'LC16', 'LC17', 'LC18', 'LC21'],
    'Lobula_plate': ['LPi', 'LPTc', 'LPT', 'VS', 'HS'],
    'Centrifugal': ['CT1', 'CT2'],
}

# Assign functional groups
def assign_group(cell_type):
    for group, types in functional_groups.items():
        if cell_type in types:
            return group
    return 'Other'

nodes_full['functional_group'] = nodes_full['cell_type'].apply(assign_group)

# Group statistics
group_stats = nodes_full.groupby('functional_group').agg({
    'root_id': 'count',
    'in_degree': 'mean',
    'out_degree': 'mean',
    'total_degree': 'mean'
}).round(2)
group_stats.columns = ['count', 'mean_in', 'mean_out', 'mean_total']
group_stats = group_stats.sort_values('count', ascending=False)

print("\nFunctional group statistics:")
print(group_stats.to_string())

# Visualize functional groups
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 Neuron count by functional group
group_counts = nodes_full['functional_group'].value_counts()
axes[0, 0].bar(range(len(group_counts)), group_counts.values, color='teal', alpha=0.7)
axes[0, 0].set_xticks(range(len(group_counts)))
axes[0, 0].set_xticklabels(group_counts.index, rotation=45, ha='right')
axes[0, 0].set_ylabel('Neuron Count', fontsize=11)
axes[0, 0].set_title('Neuron Count by Functional Group', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 6.2 Mean connectivity by functional group
x_pos = np.arange(len(group_stats))
axes[0, 1].bar(x_pos - 0.2, group_stats['mean_in'], 0.4, label='In-degree', 
               color='green', alpha=0.7)
axes[0, 1].bar(x_pos + 0.2, group_stats['mean_out'], 0.4, label='Out-degree', 
               color='orange', alpha=0.7)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(group_stats.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Mean Degree', fontsize=11)
axes[0, 1].set_title('Mean Connectivity by Functional Group', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 6.3 In-degree vs out-degree by functional group
for group in group_counts.index:
    group_nodes = nodes_full[nodes_full['functional_group'] == group]
    axes[1, 0].scatter(group_nodes['in_degree'], group_nodes['out_degree'], 
                       label=group, alpha=0.6, s=20)
axes[1, 0].set_xlabel('In-degree', fontsize=11)
axes[1, 0].set_ylabel('Out-degree', fontsize=11)
axes[1, 0].set_title('In-degree vs Out-degree by Functional Group', 
                      fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=8, loc='upper right')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# 6.4 Neurotransmitter distribution by functional group
nt_by_group = nodes_full.groupby(['functional_group', 'nt_type']).size().unstack(fill_value=0)
nt_by_group.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                 color=['red', 'gray', 'blue', 'green', 'purple'])
axes[1, 1].set_xlabel('Functional Group', fontsize=11)
axes[1, 1].set_ylabel('Neuron Count', fontsize=11)
axes[1, 1].set_title('Neurotransmitter Distribution by Functional Group', 
                      fontsize=12, fontweight='bold')
axes[1, 1].legend(title='NT Type', fontsize=8)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_full_04_functional_groups.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_full_04_functional_groups.png")
plt.close()

# ============================================================================
# Section 7: Network Topology Analysis
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: Network Topology")
print("="*80)

# Calculate network-level statistics
total_neurons = len(nodes_full)
total_edges = len(edges_full)
total_synapses = edges_full['syn_count'].sum()

# Density
max_possible_edges = total_neurons * (total_neurons - 1)
density = total_edges / max_possible_edges

# Degree distribution
all_degrees = nodes_full['total_degree'].values
mean_degree = all_degrees.mean()
median_degree = np.median(all_degrees)

# Reciprocity (bidirectional connections)
edge_pairs = set()
reciprocal_count = 0
for _, edge in edges_full.iterrows():
    pair = tuple(sorted([edge['pre_root_id'], edge['post_root_id']]))
    if pair in edge_pairs:
        reciprocal_count += 1
    edge_pairs.add(pair)
reciprocity = reciprocal_count / len(edge_pairs) if len(edge_pairs) > 0 else 0

print(f"\n=== Network Topology Statistics ===")
print(f"Total neurons: {total_neurons}")
print(f"Total edges: {total_edges}")
print(f"Total synapses: {total_synapses:.0f}")
print(f"Network density: {density:.6f}")
print(f"Mean degree: {mean_degree:.2f}")
print(f"Median degree: {median_degree:.2f}")
print(f"Reciprocity: {reciprocity:.4f}")

# Visualize topology
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7.1 Degree distribution (log-log)
degree_counts = np.bincount(all_degrees.astype(int))
degrees = np.arange(len(degree_counts))
axes[0, 0].scatter(degrees[degree_counts > 0], degree_counts[degree_counts > 0], 
                   alpha=0.6, s=30)
axes[0, 0].set_xlabel('Degree', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].set_title('Degree Distribution (log-log)', fontsize=12, fontweight='bold')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(alpha=0.3)

# 7.2 Synapse count distribution
axes[0, 1].hist(edges_full['syn_count'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[0, 1].set_xlabel('Synapse Count per Edge', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Synapse Count Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# 7.3 In-degree vs out-degree (all neurons)
axes[1, 0].hexbin(nodes_full['in_degree'], nodes_full['out_degree'], 
                  gridsize=30, cmap='YlOrRd', mincnt=1)
axes[1, 0].set_xlabel('In-degree', fontsize=11)
axes[1, 0].set_ylabel('Out-degree', fontsize=11)
axes[1, 0].set_title('In-degree vs Out-degree (All Neurons)', fontsize=12, fontweight='bold')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# 7.4 Cumulative degree distribution
sorted_degrees = np.sort(all_degrees)[::-1]
cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
axes[1, 1].plot(sorted_degrees, cumulative, linewidth=2)
axes[1, 1].set_xlabel('Degree', fontsize=11)
axes[1, 1].set_ylabel('Cumulative Fraction', fontsize=11)
axes[1, 1].set_title('Cumulative Degree Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xscale('log')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_full_05_topology.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_full_05_topology.png")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FULL CONNECTOME VISUALIZATION COMPLETE ✓")
print("="*80)

print("\nGenerated files:")
print("  - cell_type_stats.csv")
print("  - fig_full_01_overview.png")
print("  - fig_full_02_connectivity_matrix.png")
print("  - fig_full_03_clustering.png")
print("  - fig_full_04_functional_groups.png")
print("  - fig_full_05_topology.png")

print("\nKey findings:")
print(f"  • {nodes_full['cell_type'].nunique()} unique cell types")
print(f"  • {total_neurons} neurons, {total_edges} edges")
print(f"  • Network density: {density:.6f}")
print(f"  • Mean degree: {mean_degree:.2f}")
print(f"  • Reciprocity: {reciprocity:.4f}")
print(f"  • Top 3 cell types: {', '.join(cell_type_counts.head(3).index.tolist())}")

print("\nNext steps:")
print("  1. Compare with BANC whole-brain connectome")
print("  2. Analyze specific pathways (e.g., T4/T5 → LC → DN)")
print("  3. Build network models for different functional groups")
print("  4. Validate connectivity patterns with literature")
