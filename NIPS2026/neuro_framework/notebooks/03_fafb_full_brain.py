"""
FAFB Full Brain Connectome Visualization
=========================================
Complete visualization of FAFB v783 dataset with ALL 139,255 neurons.

This script provides:
1. Full brain statistics (all neuron types)
2. Super-class and class-level analysis
3. Visual system detailed analysis (95k neurons)
4. Connectivity patterns across brain regions
5. Comparison with optic lobe subset

Data source: /Users/lengyuner/Desktop/data/flywire/Jun2025/
- consolidated_cell_types.csv.gz (138,327 neurons, 8,772 types)
- classification.csv.gz (139,255 neurons, hierarchical)
- connections_princeton.csv.gz (5.3M edges, filtered ≥5 synapses)
- visual_neuron_types.csv.gz (95,079 visual neurons)

Usage:
    cd /Users/lengyuner/Desktop/NIPS2026
    /Users/lengyuner/anaconda3/bin/python neuro_framework/notebooks/03_fafb_full_brain.py
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from neuro_framework.utils.logging import setup_logging

# Setup
setup_logging(level='INFO')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)

# FAFB data path
FAFB_DIR = Path('/Users/lengyuner/Desktop/data/flywire/Jun2025')

print(f"Project root: {PROJECT_ROOT}")
print(f"FAFB data dir: {FAFB_DIR}")
print("\n" + "="*80)
print("FAFB FULL BRAIN CONNECTOME VISUALIZATION")
print("="*80)

# ============================================================================
# Section 1: Load FAFB Full Brain Data
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: Load FAFB Full Brain Data")
print("="*80)

print("Loading consolidated_cell_types.csv.gz...")
cell_types = pd.read_csv(FAFB_DIR / 'consolidated_cell_types.csv.gz', compression='gzip')
print(f"✓ Loaded {len(cell_types)} neurons with cell types")

print("Loading classification.csv.gz...")
classification = pd.read_csv(FAFB_DIR / 'classification.csv.gz', compression='gzip')
print(f"✓ Loaded {len(classification)} neurons with classification")

print("Loading connections_princeton.csv.gz (this may take a minute)...")
connections = pd.read_csv(FAFB_DIR / 'connections_princeton.csv.gz', compression='gzip')
print(f"✓ Loaded {len(connections)} connections (≥5 synapses)")

# Merge cell types and classification
print("\nMerging datasets...")
neurons = classification.merge(cell_types, on='root_id', how='left')
print(f"✓ Merged dataset: {len(neurons)} neurons")

print(f"\n=== FAFB Full Brain Statistics ===")
print(f"Total neurons: {len(neurons):,}")
print(f"Total connections: {len(connections):,}")
print(f"Total synapses: {connections['syn_count'].sum():,.0f}")
print(f"Unique cell types: {neurons['primary_type'].nunique():,}")
print(f"Unique super classes: {neurons['super_class'].nunique()}")
print(f"Unique classes: {neurons['class'].nunique()}")

# ============================================================================
# Section 2: Super-Class Analysis
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: Super-Class Analysis")
print("="*80)

super_class_counts = neurons['super_class'].value_counts()
print("\nSuper-class distribution:")
print(super_class_counts)

# Calculate connectivity per super-class
print("\nCalculating connectivity statistics...")
# This is memory-intensive, so we'll sample if needed
if len(connections) > 10000000:
    print("  (Using sampled connections for speed)")
    conn_sample = connections.sample(n=min(5000000, len(connections)), random_state=42)
else:
    conn_sample = connections

# Map root_id to super_class
root_to_super = neurons.set_index('root_id')['super_class'].to_dict()

conn_sample['pre_super'] = conn_sample['pre_root_id'].map(root_to_super)
conn_sample['post_super'] = conn_sample['post_root_id'].map(root_to_super)

# Super-class connectivity matrix
super_classes = neurons['super_class'].dropna().unique()
super_to_idx = {s: i for i, s in enumerate(sorted(super_classes))}

super_conn_matrix = np.zeros((len(super_classes), len(super_classes)))
for _, row in conn_sample.iterrows():
    pre_s = row['pre_super']
    post_s = row['post_super']
    if pd.notna(pre_s) and pd.notna(post_s) and pre_s in super_to_idx and post_s in super_to_idx:
        i = super_to_idx[pre_s]
        j = super_to_idx[post_s]
        super_conn_matrix[i, j] += row['syn_count']

print(f"✓ Super-class connectivity matrix: {super_conn_matrix.shape}")

# Visualize super-class analysis
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 2.1 Super-class neuron counts
axes[0, 0].barh(range(len(super_class_counts)), super_class_counts.values, color='steelblue', alpha=0.8)
axes[0, 0].set_yticks(range(len(super_class_counts)))
axes[0, 0].set_yticklabels(super_class_counts.index, fontsize=10)
axes[0, 0].set_xlabel('Neuron Count', fontsize=12)
axes[0, 0].set_title('Neuron Count by Super-Class', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(super_class_counts.values):
    axes[0, 0].text(v, i, f' {v:,}', va='center', fontsize=9)

# 2.2 Super-class connectivity matrix
sorted_super = sorted(super_classes)
im = axes[0, 1].imshow(np.log10(super_conn_matrix + 1), cmap='viridis', aspect='auto')
axes[0, 1].set_xticks(range(len(sorted_super)))
axes[0, 1].set_yticks(range(len(sorted_super)))
axes[0, 1].set_xticklabels(sorted_super, rotation=45, ha='right', fontsize=9)
axes[0, 1].set_yticklabels(sorted_super, fontsize=9)
axes[0, 1].set_xlabel('Post-synaptic Super-Class', fontsize=11)
axes[0, 1].set_ylabel('Pre-synaptic Super-Class', fontsize=11)
axes[0, 1].set_title('Super-Class Connectivity Matrix (log10)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[0, 1], label='log10(synapse count + 1)')

# 2.3 Flow distribution
flow_counts = neurons['flow'].value_counts()
axes[1, 0].pie(flow_counts.values, labels=flow_counts.index, autopct='%1.1f%%', 
               colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
axes[1, 0].set_title('Neuron Flow Distribution', fontsize=14, fontweight='bold')

# 2.4 Side distribution
side_counts = neurons['side'].value_counts()
axes[1, 1].bar(range(len(side_counts)), side_counts.values, 
               color=['red', 'blue', 'gray'], alpha=0.7)
axes[1, 1].set_xticks(range(len(side_counts)))
axes[1, 1].set_xticklabels(side_counts.index, fontsize=11)
axes[1, 1].set_ylabel('Neuron Count', fontsize=12)
axes[1, 1].set_title('Neuron Side Distribution', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(side_counts.values):
    axes[1, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_fafb_01_super_class.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_fafb_01_super_class.png")
plt.close()

# ============================================================================
# Section 3: Cell Type Analysis (Top Types)
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: Cell Type Analysis")
print("="*80)

cell_type_counts = neurons['primary_type'].value_counts()
print(f"\nTotal unique cell types: {len(cell_type_counts):,}")
print(f"\nTop 50 cell types:")
print(cell_type_counts.head(50))

# Save top cell types to CSV
top_types_df = pd.DataFrame({
    'cell_type': cell_type_counts.head(100).index,
    'count': cell_type_counts.head(100).values
})

# Add super_class info
type_to_super = neurons.groupby('primary_type')['super_class'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown')
top_types_df['super_class'] = top_types_df['cell_type'].map(type_to_super)

csv_path = PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'fafb_top100_cell_types.csv'
top_types_df.to_csv(csv_path, index=False)
print(f"\n✓ Saved fafb_top100_cell_types.csv")

# Visualize top cell types
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 3.1 Top 40 cell types
top40 = cell_type_counts.head(40)
axes[0, 0].barh(range(len(top40)), top40.values, color='coral', alpha=0.8)
axes[0, 0].set_yticks(range(len(top40)))
axes[0, 0].set_yticklabels(top40.index, fontsize=8)
axes[0, 0].set_xlabel('Neuron Count', fontsize=11)
axes[0, 0].set_title('Top 40 Cell Types', fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 3.2 Cell type count distribution
count_dist = cell_type_counts.values
axes[0, 1].hist(count_dist, bins=100, edgecolor='black', alpha=0.7, color='purple')
axes[0, 1].set_xlabel('Neuron Count per Cell Type', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Cell Type Count Distribution', fontsize=13, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)

# 3.3 Cell types per super-class
super_type_counts = neurons.groupby('super_class')['primary_type'].nunique().sort_values(ascending=False)
axes[1, 0].bar(range(len(super_type_counts)), super_type_counts.values, color='teal', alpha=0.7)
axes[1, 0].set_xticks(range(len(super_type_counts)))
axes[1, 0].set_xticklabels(super_type_counts.index, rotation=45, ha='right', fontsize=9)
axes[1, 0].set_ylabel('Number of Cell Types', fontsize=11)
axes[1, 0].set_title('Cell Type Diversity by Super-Class', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 3.4 Cumulative cell type distribution
sorted_counts = np.sort(cell_type_counts.values)[::-1]
cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
axes[1, 1].plot(range(len(cumulative)), cumulative * 100, linewidth=2, color='darkblue')
axes[1, 1].axhline(y=50, color='red', linestyle='--', label='50% of neurons')
axes[1, 1].axhline(y=80, color='orange', linestyle='--', label='80% of neurons')
axes[1, 1].set_xlabel('Number of Cell Types (ranked)', fontsize=11)
axes[1, 1].set_ylabel('Cumulative % of Neurons', fontsize=11)
axes[1, 1].set_title('Cumulative Cell Type Distribution', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xscale('log')

# Find how many types cover 50% and 80%
idx_50 = np.where(cumulative >= 0.5)[0][0]
idx_80 = np.where(cumulative >= 0.8)[0][0]
axes[1, 1].text(idx_50, 55, f'{idx_50} types', ha='center', fontsize=9, color='red')
axes[1, 1].text(idx_80, 85, f'{idx_80} types', ha='center', fontsize=9, color='orange')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_fafb_02_cell_types.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_fafb_02_cell_types.png")
plt.close()

print(f"\n50% of neurons covered by top {idx_50} cell types")
print(f"80% of neurons covered by top {idx_80} cell types")

# ============================================================================
# Section 4: Visual System Analysis
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: Visual System Analysis")
print("="*80)

print("Loading visual_neuron_types.csv.gz...")
visual_neurons = pd.read_csv(FAFB_DIR / 'visual_neuron_types.csv.gz', compression='gzip')
print(f"✓ Loaded {len(visual_neurons)} visual neurons")

# Merge with main neuron data
visual_full = visual_neurons.merge(neurons[['root_id', 'super_class', 'flow']], 
                                    on='root_id', how='left')

print(f"\nVisual system statistics:")
print(f"Total visual neurons: {len(visual_full):,}")
print(f"Unique visual types: {visual_full['type'].nunique():,}")
print(f"Unique visual families: {visual_full['family'].nunique()}")
print(f"Unique subsystems: {visual_full['subsystem'].nunique()}")

# Visual neuron type distribution
visual_type_counts = visual_full['type'].value_counts()
visual_family_counts = visual_full['family'].value_counts()
visual_subsystem_counts = visual_full['subsystem'].value_counts()

print(f"\nTop 20 visual neuron types:")
print(visual_type_counts.head(20))

print(f"\nTop 10 visual families:")
print(visual_family_counts.head(10))

print(f"\nSubsystem distribution:")
print(visual_subsystem_counts)

# Visualize visual system
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 4.1 Top 30 visual types
top30_visual = visual_type_counts.head(30)
axes[0, 0].barh(range(len(top30_visual)), top30_visual.values, color='green', alpha=0.7)
axes[0, 0].set_yticks(range(len(top30_visual)))
axes[0, 0].set_yticklabels(top30_visual.index, fontsize=8)
axes[0, 0].set_xlabel('Neuron Count', fontsize=11)
axes[0, 0].set_title('Top 30 Visual Neuron Types', fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 4.2 Visual families
top20_families = visual_family_counts.head(20)
axes[0, 1].bar(range(len(top20_families)), top20_families.values, color='orange', alpha=0.7)
axes[0, 1].set_xticks(range(len(top20_families)))
axes[0, 1].set_xticklabels(top20_families.index, rotation=90, fontsize=8)
axes[0, 1].set_ylabel('Neuron Count', fontsize=11)
axes[0, 1].set_title('Top 20 Visual Neuron Families', fontsize=13, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# 4.3 Subsystem distribution
axes[1, 0].bar(range(len(visual_subsystem_counts)), visual_subsystem_counts.values, 
               color='purple', alpha=0.7)
axes[1, 0].set_xticks(range(len(visual_subsystem_counts)))
axes[1, 0].set_xticklabels(visual_subsystem_counts.index, rotation=45, ha='right', fontsize=9)
axes[1, 0].set_ylabel('Neuron Count', fontsize=11)
axes[1, 0].set_title('Visual Subsystem Distribution', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(visual_subsystem_counts.values):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

# 4.4 Category distribution
category_counts = visual_full['category'].value_counts()
axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               colors=['#ff6b6b', '#4ecdc4'], startangle=90)
axes[1, 1].set_title('Visual Neuron Category', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_fafb_03_visual_system.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_fafb_03_visual_system.png")
plt.close()

# ============================================================================
# Section 5: Connectivity Analysis
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: Connectivity Analysis")
print("="*80)

# Neuropil distribution
neuropil_counts = connections['neuropil'].value_counts()
neuropil_synapses = connections.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)

print(f"\nTop 20 neuropils by connection count:")
print(neuropil_counts.head(20))

print(f"\nTop 20 neuropils by synapse count:")
print(neuropil_synapses.head(20))

# NT type distribution
nt_counts = connections['nt_type'].value_counts()
nt_synapses = connections.groupby('nt_type')['syn_count'].sum()

print(f"\nNeurotransmitter distribution (connections):")
print(nt_counts)

print(f"\nNeurotransmitter distribution (synapses):")
print(nt_synapses)

# Visualize connectivity
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 5.1 Top 30 neuropils by connection count
top30_neuro = neuropil_counts.head(30)
axes[0, 0].barh(range(len(top30_neuro)), top30_neuro.values, color='steelblue', alpha=0.7)
axes[0, 0].set_yticks(range(len(top30_neuro)))
axes[0, 0].set_yticklabels(top30_neuro.index, fontsize=8)
axes[0, 0].set_xlabel('Connection Count', fontsize=11)
axes[0, 0].set_title('Top 30 Neuropils by Connection Count', fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# 5.2 Synapse count distribution
axes[0, 1].hist(connections['syn_count'], bins=100, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Synapse Count per Connection', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Synapse Count Distribution (≥5 threshold)', fontsize=13, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# 5.3 NT type distribution
axes[1, 0].bar(range(len(nt_counts)), nt_counts.values, 
               color=['red', 'blue', 'green', 'orange', 'purple', 'gray'][:len(nt_counts)], 
               alpha=0.7)
axes[1, 0].set_xticks(range(len(nt_counts)))
axes[1, 0].set_xticklabels(nt_counts.index, rotation=45, ha='right', fontsize=10)
axes[1, 0].set_ylabel('Connection Count', fontsize=11)
axes[1, 0].set_title('Neurotransmitter Type Distribution', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(nt_counts.values):
    axes[1, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

# 5.4 NT type by synapses
axes[1, 1].bar(range(len(nt_synapses)), nt_synapses.values,
               color=['red', 'blue', 'green', 'orange', 'purple', 'gray'][:len(nt_synapses)],
               alpha=0.7)
axes[1, 1].set_xticks(range(len(nt_synapses)))
axes[1, 1].set_xticklabels(nt_synapses.index, rotation=45, ha='right', fontsize=10)
axes[1, 1].set_ylabel('Total Synapse Count', fontsize=11)
axes[1, 1].set_title('Neurotransmitter Type by Synapse Count', fontsize=13, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'neuro_framework' / 'notebooks' / 'figures/fig_fafb_04_connectivity.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved fig_fafb_04_connectivity.png")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FAFB FULL BRAIN VISUALIZATION COMPLETE ✓")
print("="*80)

print("\nGenerated files:")
print("  - fig_fafb_01_super_class.png")
print("  - fig_fafb_02_cell_types.png")
print("  - fig_fafb_03_visual_system.png")
print("  - fig_fafb_04_connectivity.png")
print("  - fafb_top100_cell_types.csv")

print("\nKey findings:")
print(f"  • {len(neurons):,} total neurons")
print(f"  • {cell_type_counts.nunique():,} unique cell types")
print(f"  • {len(connections):,} connections (≥5 synapses)")
print(f"  • {connections['syn_count'].sum():,.0f} total synapses")
print(f"  • {len(visual_full):,} visual neurons ({len(visual_full)/len(neurons)*100:.1f}%)")
print(f"  • Top 3 super-classes: {', '.join(super_class_counts.head(3).index.tolist())}")
print(f"  • Top 3 cell types: {', '.join(cell_type_counts.head(3).index.tolist())}")
print(f"  • {idx_50} cell types cover 50% of neurons")
print(f"  • {idx_80} cell types cover 80% of neurons")

print("\nNext steps:")
print("  1. Compare FAFB with optic lobe subset")
print("  2. Build network models for specific pathways")
print("  3. Analyze LC neurons and their connectivity")
print("  4. Integrate with calcium imaging data")
