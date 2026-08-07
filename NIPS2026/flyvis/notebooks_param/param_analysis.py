#!/usr/bin/env python3
"""
FlyVis Model Parameter Analysis
Generates plots for: tau, bias, syn_strength, syn_count, sign
"""
import sys, warnings, os
warnings.filterwarnings('ignore')

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm as sp_norm
from pathlib import Path

sns.set_theme(style='whitegrid', context='notebook', palette='deep')
plt.rcParams['figure.dpi'] = 120

out_dir = Path(__file__).parent / 'outputs'
out_dir.mkdir(exist_ok=True)

print('Importing flyvis...')
import flyvis
from flyvis.network.network import Network

print('Building network...')
net = Network()

# ─── Extract node params ──────────────────────────────────────────────────────
tc_param   = net.node_params.time_const
bias_param = net.node_params.bias

node_types = tc_param.keys
tau_vals   = tc_param.semantic_values.detach().cpu().numpy()
bias_vals  = bias_param.semantic_values.detach().cpu().numpy()

df_nodes = pd.DataFrame({
    'neuron_type': node_types,
    'tau':         tau_vals,
    'bias':        bias_vals,
})

# ─── Extract edge params ──────────────────────────────────────────────────────
ss_param = net.edge_params.syn_strength
sc_param = net.edge_params.syn_count
sg_param = net.edge_params.sign

ss_keys = ss_param.keys
df_edges = pd.DataFrame({
    'source_type':  [k[0] for k in ss_keys],
    'target_type':  [k[1] for k in ss_keys],
    'syn_strength': ss_param.semantic_values.detach().cpu().numpy(),
    'sign':         sg_param.semantic_values.detach().cpu().numpy(),
})
df_edges['signed_strength'] = df_edges['sign'] * df_edges['syn_strength']

sc_keys = sc_param.keys
df_syn_count = pd.DataFrame({
    'source_type': [k[0] for k in sc_keys],
    'target_type': [k[1] for k in sc_keys],
    'du':          [k[2] for k in sc_keys],
    'dv':          [k[3] for k in sc_keys],
    'syn_count':   sc_param.semantic_values.detach().cpu().numpy(),
})

print(f'Neuron types: {len(df_nodes)}')
print(f'Edge type pairs: {len(df_edges)}')
print(f'Syn count entries: {len(df_syn_count)}')
print(f'Exc/Inh: {(df_edges.sign>0).sum()}/{(df_edges.sign<0).sum()}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Time Constant tau
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bar chart
ax = axes[0]
df_s = df_nodes.sort_values('tau').reset_index(drop=True)
cmap = plt.cm.coolwarm
norm = plt.Normalize(df_s['tau'].min() - 1e-6, df_s['tau'].max() + 1e-6)
colors = [cmap(norm(v)) for v in df_s['tau']]
bars = ax.barh(df_s['neuron_type'], df_s['tau'], color=colors, edgecolor='white', linewidth=0.4)
ax.axvline(0.05, color='red', linestyle='--', lw=1.8, label='init = 0.05 s', alpha=0.8)
ax.set_xlabel('tau (s)', fontsize=12)
ax.set_title('Time Constant tau per Neuron Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, df_s['tau']):
    ax.text(bar.get_width() + df_s['tau'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='left', fontsize=6)

# Histogram
ax2 = axes[1]
ax2.hist(df_nodes['tau'], bins=25, color='steelblue', edgecolor='white', alpha=0.75, density=True)
tau_std = df_nodes['tau'].std()
if tau_std > 1e-8:
    kde = gaussian_kde(df_nodes['tau'])
    xr = np.linspace(df_nodes['tau'].min() - 3*tau_std, df_nodes['tau'].max() + 3*tau_std, 400)
    ax2.plot(xr, kde(xr), color='navy', lw=2.5, label='KDE')
ax2.axvline(df_nodes['tau'].mean(), color='red', linestyle='--', lw=1.8,
            label=f'mean = {df_nodes["tau"].mean():.5f}')
ax2.axvline(0.05, color='orange', linestyle=':', lw=1.8, label='init = 0.05')
ax2.set_xlabel('tau (s)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Distribution of tau (all neuron types)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

tau_all_same = df_nodes['tau'].nunique() == 1
plt.suptitle(
    f'tau: min={df_nodes["tau"].min():.5f}, max={df_nodes["tau"].max():.5f}, '
    f'std={tau_std:.6f}  |  All same value: {tau_all_same}',
    fontsize=10, y=1.01
)
plt.tight_layout()
fig.savefig(out_dir / '01_tau_distribution.png', dpi=150, bbox_inches='tight')
print('Saved: 01_tau_distribution.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Bias (Resting Potential)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
df_sb = df_nodes.sort_values('bias').reset_index(drop=True)
cmap2 = plt.cm.RdYlGn
norm2 = plt.Normalize(df_sb['bias'].min(), df_sb['bias'].max())
colors_b = [cmap2(norm2(v)) for v in df_sb['bias']]
bars_b = ax.barh(df_sb['neuron_type'], df_sb['bias'], color=colors_b, edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color='navy', linestyle='--', lw=1.8, label='init mean = 0.5', alpha=0.8)
ax.set_xlabel('Bias (resting potential)', fontsize=12)
ax.set_title('Bias per Neuron Type', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars_b, df_sb['bias']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=7)

ax2 = axes[1]
mu_b, sig_b = df_nodes['bias'].mean(), df_nodes['bias'].std()
ax2.hist(df_nodes['bias'], bins=20, color='mediumseagreen', edgecolor='white', alpha=0.75, density=True)
xr = np.linspace(df_nodes['bias'].min() - 0.05, df_nodes['bias'].max() + 0.05, 300)
ax2.plot(xr, sp_norm.pdf(xr, mu_b, sig_b), color='darkgreen', lw=2.5, label=f'N({mu_b:.3f},{sig_b:.3f})')
ax2.axvline(0.5, color='navy', linestyle='--', lw=1.8, label='init mean = 0.5')
ax2.axvline(mu_b, color='red', linestyle='-', lw=1.5, label=f'current mean = {mu_b:.3f}')
ax2.set_xlabel('Bias', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Distribution of Bias (all neuron types)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.suptitle(f'Bias: mean={mu_b:.4f}, std={sig_b:.4f}, min={df_nodes["bias"].min():.4f}, max={df_nodes["bias"].max():.4f}',
             fontsize=10, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / '02_bias_distribution.png', dpi=150, bbox_inches='tight')
print('Saved: 02_bias_distribution.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: tau vs Bias scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(df_nodes['tau'], df_nodes['bias'], s=90, alpha=0.85,
                c=range(len(df_nodes)), cmap='tab20', edgecolors='gray', linewidth=0.5)
for _, row in df_nodes.iterrows():
    ax.annotate(row['neuron_type'], (row['tau'], row['bias']),
                fontsize=7, ha='center', va='bottom', alpha=0.85)
ax.axvline(0.05, color='red', linestyle='--', alpha=0.5, lw=1.2, label='tau init=0.05')
ax.axhline(0.5, color='blue', linestyle='--', alpha=0.5, lw=1.2, label='bias init=0.5')
ax.set_xlabel('Time Constant tau (s)', fontsize=12)
ax.set_ylabel('Bias (resting potential)', fontsize=12)
ax.set_title('tau vs Bias for Each Neuron Type', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(out_dir / '03_tau_vs_bias_scatter.png', dpi=150, bbox_inches='tight')
print('Saved: 03_tau_vs_bias_scatter.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Synapse Strength distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.hist(df_edges['syn_strength'], bins=60, color='coral', edgecolor='white', alpha=0.8, density=True)
mean_ss = df_edges['syn_strength'].mean()
median_ss = df_edges['syn_strength'].median()
ax.axvline(mean_ss, color='red', linestyle='--', lw=1.8, label=f'mean={mean_ss:.5f}')
ax.axvline(median_ss, color='purple', linestyle=':', lw=1.8, label=f'median={median_ss:.5f}')
ax.set_xlabel('Synapse Strength', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Synapse Strength\n(all source->target type pairs)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax2 = axes[1]
exc_s = df_edges.loc[df_edges['sign'] > 0, 'signed_strength'].values
inh_s = df_edges.loc[df_edges['sign'] < 0, 'signed_strength'].values
ax2.hist(exc_s, bins=40, color='#2196F3', alpha=0.7, density=True, label=f'Excitatory (n={len(exc_s)})')
ax2.hist(inh_s, bins=40, color='#F44336', alpha=0.7, density=True, label=f'Inhibitory (n={len(inh_s)})')
ax2.axvline(0, color='black', lw=1.2, alpha=0.5)
ax2.set_xlabel('Signed Synapse Strength', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Excitatory vs Inhibitory Synapse Strength', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / '04_syn_strength_distribution.png', dpi=150, bbox_inches='tight')
print('Saved: 04_syn_strength_distribution.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Syn strength per source neuron type (boxplot)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
order_src = (df_edges.groupby('source_type')['syn_strength']
             .median().sort_values().index.tolist())
sns.boxplot(data=df_edges, y='source_type', x='syn_strength', order=order_src,
            palette='Blues', ax=ax, fliersize=2)
ax.set_title('Syn Strength by Source Neuron Type', fontsize=12, fontweight='bold')
ax.set_xlabel('syn_strength', fontsize=11)
ax.set_ylabel('Source Type', fontsize=11)
ax.grid(axis='x', alpha=0.3)

ax2 = axes[1]
order_tgt = (df_edges.groupby('target_type')['syn_strength']
             .median().sort_values().index.tolist())
sns.boxplot(data=df_edges, y='target_type', x='syn_strength', order=order_tgt,
            palette='Oranges', ax=ax2, fliersize=2)
ax2.set_title('Syn Strength by Target Neuron Type', fontsize=12, fontweight='bold')
ax2.set_xlabel('syn_strength', fontsize=11)
ax2.set_ylabel('Target Type', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / '05_syn_strength_by_type.png', dpi=150, bbox_inches='tight')
print('Saved: 05_syn_strength_by_type.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Syn Count distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.hist(df_syn_count['syn_count'], bins=60, color='mediumpurple', edgecolor='white', alpha=0.8)
ax.set_xlabel('Synapse Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Synapse Count\n(src, tgt, du, dv)', fontsize=12, fontweight='bold')
ax.axvline(df_syn_count['syn_count'].mean(), color='red', linestyle='--', lw=1.8,
           label=f'mean={df_syn_count["syn_count"].mean():.1f}')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.hist(np.log1p(df_syn_count['syn_count']), bins=60, color='mediumorchid', edgecolor='white', alpha=0.8)
ax2.set_xlabel('log(1 + syn_count)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Log-scale Synapse Count Distribution', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / '06_syn_count_distribution.png', dpi=150, bbox_inches='tight')
print('Saved: 06_syn_count_distribution.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Syn count by source type
# ─────────────────────────────────────────────────────────────────────────────
df_sc_agg = df_syn_count.groupby(['source_type','target_type'])['syn_count'].sum().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax = axes[0]
order_sc_src = (df_sc_agg.groupby('source_type')['syn_count']
                .sum().sort_values().index.tolist())
sns.boxplot(data=df_sc_agg, y='source_type', x='syn_count', order=order_sc_src,
            palette='Purples', ax=ax, fliersize=2)
ax.set_title('Total Syn Count by Source Type', fontsize=12, fontweight='bold')
ax.set_xlabel('Syn Count (summed over spatial offsets)', fontsize=10)
ax.set_ylabel('Source Type', fontsize=11)
ax.grid(axis='x', alpha=0.3)

ax2 = axes[1]
order_sc_tgt = (df_sc_agg.groupby('target_type')['syn_count']
                .sum().sort_values().index.tolist())
sns.boxplot(data=df_sc_agg, y='target_type', x='syn_count', order=order_sc_tgt,
            palette='Greens', ax=ax2, fliersize=2)
ax2.set_title('Total Syn Count by Target Type', fontsize=12, fontweight='bold')
ax2.set_xlabel('Syn Count (summed over spatial offsets)', fontsize=10)
ax2.set_ylabel('Target Type', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / '07_syn_count_by_type.png', dpi=150, bbox_inches='tight')
print('Saved: 07_syn_count_by_type.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Heatmap — mean syn_strength (source_type x target_type)
# ─────────────────────────────────────────────────────────────────────────────
pivot_ss = (df_edges.pivot_table(index='source_type', columns='target_type',
                                  values='signed_strength', aggfunc='mean'))
# restrict to types that appear in both axes
all_types = sorted(set(df_edges['source_type']) | set(df_edges['target_type']))
pivot_ss = pivot_ss.reindex(index=all_types, columns=all_types)

fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(pivot_ss, ax=ax, cmap='RdBu_r', center=0,
            linewidths=0.3, linecolor='lightgray',
            cbar_kws={'label': 'mean signed syn_strength', 'shrink': 0.6})
ax.set_title('Mean Signed Synapse Strength (source -> target)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target Neuron Type', fontsize=12)
ax.set_ylabel('Source Neuron Type', fontsize=12)
ax.tick_params(axis='x', rotation=90, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)
plt.tight_layout()
fig.savefig(out_dir / '08_syn_strength_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved: 08_syn_strength_heatmap.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Summary — all node params side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# tau bar
ax = axes[0, 0]
df_s2 = df_nodes.sort_values('tau').reset_index(drop=True)
cmap = plt.cm.coolwarm
norm = plt.Normalize(df_s2['tau'].min() - 1e-8, df_s2['tau'].max() + 1e-8)
ax.barh(df_s2['neuron_type'], df_s2['tau'],
        color=[cmap(norm(v)) for v in df_s2['tau']], edgecolor='white', linewidth=0.4)
ax.axvline(0.05, color='red', linestyle='--', lw=1.5, label='init=0.05')
ax.set_title('Time Constant tau per Type', fontsize=11, fontweight='bold')
ax.set_xlabel('tau (s)')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

# bias bar
ax = axes[0, 1]
df_sb2 = df_nodes.sort_values('bias').reset_index(drop=True)
cmap2 = plt.cm.RdYlGn
norm2 = plt.Normalize(df_sb2['bias'].min(), df_sb2['bias'].max())
ax.barh(df_sb2['neuron_type'], df_sb2['bias'],
        color=[cmap2(norm2(v)) for v in df_sb2['bias']], edgecolor='white', linewidth=0.4)
ax.axvline(0.5, color='navy', linestyle='--', lw=1.5, label='init mean=0.5')
ax.set_title('Bias (Resting Potential) per Type', fontsize=11, fontweight='bold')
ax.set_xlabel('Bias')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

# syn_strength by sign
ax = axes[1, 0]
exc_s2 = df_edges.loc[df_edges['sign'] > 0, 'syn_strength'].values
inh_s2 = df_edges.loc[df_edges['sign'] < 0, 'syn_strength'].values
ax.hist(exc_s2, bins=40, color='#2196F3', alpha=0.7, label=f'Excitatory (n={len(exc_s2)})', density=True)
ax.hist(inh_s2, bins=40, color='#F44336', alpha=0.7, label=f'Inhibitory (n={len(inh_s2)})', density=True)
ax.set_title('Synapse Strength Distribution', fontsize=11, fontweight='bold')
ax.set_xlabel('syn_strength')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# syn_count log
ax = axes[1, 1]
ax.hist(np.log1p(df_syn_count['syn_count']), bins=50, color='mediumpurple',
        edgecolor='white', alpha=0.8)
ax.set_title('Synapse Count Distribution (log scale)', fontsize=11, fontweight='bold')
ax.set_xlabel('log(1 + syn_count)')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.3)

plt.suptitle('FlyVis Network Parameter Summary', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(out_dir / '09_parameter_summary.png', dpi=150, bbox_inches='tight')
print('Saved: 09_parameter_summary.png')
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Print statistics
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('PARAMETER STATISTICS SUMMARY')
print('='*60)
print('\n--- Time Constant tau ---')
print(df_nodes[['neuron_type','tau']].sort_values('tau').to_string(index=False))
print(f'tau: all_same={tau_all_same}, mean={df_nodes["tau"].mean():.6f}, '
      f'std={df_nodes["tau"].std():.6f}, '
      f'min={df_nodes["tau"].min():.6f}, max={df_nodes["tau"].max():.6f}')

print('\n--- Bias ---')
print(df_nodes[['neuron_type','bias']].sort_values('bias').to_string(index=False))
print(f'bias: mean={df_nodes["bias"].mean():.4f}, std={df_nodes["bias"].std():.4f}')

print('\n--- Synapse Strength ---')
print(f'Total pairs: {len(df_edges)}')
print(f'Excitatory: {(df_edges.sign>0).sum()}  Inhibitory: {(df_edges.sign<0).sum()}')
print(f'syn_strength: mean={df_edges["syn_strength"].mean():.5f}, '
      f'std={df_edges["syn_strength"].std():.5f}, '
      f'min={df_edges["syn_strength"].min():.5f}, max={df_edges["syn_strength"].max():.5f}')

print('\n--- Synapse Count ---')
print(f'Entries: {len(df_syn_count)}')
print(f'syn_count: mean={df_syn_count["syn_count"].mean():.2f}, '
      f'std={df_syn_count["syn_count"].std():.2f}, '
      f'min={df_syn_count["syn_count"].min():.1f}, max={df_syn_count["syn_count"].max():.1f}')

print(f'\nAll plots saved to: {out_dir.absolute()}')
print('Done!')
