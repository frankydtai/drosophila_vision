#!/usr/bin/env python
# coding: utf-8

"""
FlyWire 连接可视化

这个脚本使用多种方式展示 FlyWire 连接组的连接模式。
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

# 设置中文字体支持
import locale
locale.setlocale(locale.LC_ALL, '')

import matplotlib.font_manager as fm
import platform
system = platform.system()

if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'DejaVu Sans']
elif system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置绘图样式
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass
sns.set_palette("husl")

# 设置输出目录
output_dir = Path('outputs/connections')
output_dir.mkdir(parents=True, exist_ok=True)

print("✓ 库导入成功")
print(f"✓ 输出目录: {output_dir}")

# ============================================================================
# 1. 加载数据
# ============================================================================

print("\n" + "="*60)
print("1. 加载 FlyWire 连接组数据")
print("="*60)

json_path = Path("../flyvis/connectome/flywire_v1.0.json")

with open(json_path, 'r') as f:
    flywire_data = json.load(f)

nodes = flywire_data['nodes']
edges = flywire_data['edges']
cell_types = [node['name'] for node in nodes]

print(f"✓ 加载完成")
print(f"  - 细胞类型: {len(cell_types)}")
print(f"  - 连接: {len(edges)}")

# 构建连接矩阵
n_types = len(cell_types)
type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
conn_matrix = np.zeros((n_types, n_types))
syn_matrix = np.zeros((n_types, n_types))

for edge in edges:
    src_idx = type_to_idx[edge['src']]
    tar_idx = type_to_idx[edge['tar']]
    conn_matrix[src_idx, tar_idx] = 1
    
    # 计算突触数
    if 'offsets' in edge and len(edge['offsets']) > 0:
        syn_count = sum(offset[1] for offset in edge['offsets'])
    else:
        syn_count = 1
    syn_matrix[src_idx, tar_idx] = syn_count

print(f"✓ 连接矩阵构建完成: {conn_matrix.shape}")

# ============================================================================
# 2. 网络图可视化
# ============================================================================

print("\n" + "="*60)
print("2. 网络图可视化")
print("="*60)

# 选择连接度最高的节点进行可视化
in_degree = defaultdict(int)
out_degree = defaultdict(int)
for edge in edges:
    out_degree[edge['src']] += 1
    in_degree[edge['tar']] += 1

total_degree = {ct: in_degree[ct] + out_degree[ct] for ct in cell_types}
top_nodes = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:30]
top_node_names = [node[0] for node in top_nodes]

print(f"选择 Top 30 节点进行可视化")

# 构建 NetworkX 图
G = nx.DiGraph()
for edge in edges:
    if edge['src'] in top_node_names and edge['tar'] in top_node_names:
        if 'offsets' in edge and len(edge['offsets']) > 0:
            weight = sum(offset[1] for offset in edge['offsets'])
        else:
            weight = 1
        G.add_edge(edge['src'], edge['tar'], weight=weight)

print(f"✓ 网络图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 绘制网络图
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# 节点大小根据度数
node_sizes = [total_degree[node] * 50 for node in G.nodes()]

# 边宽度根据权重
edges_list = G.edges()
weights = [G[u][v]['weight'] for u, v in edges_list]
max_weight = max(weights) if weights else 1
edge_widths = [w / max_weight * 3 for w in weights]

# 绘制
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                       alpha=0.7, edgecolors='black', linewidths=1.5)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                       edge_color='gray', arrows=True, arrowsize=15,
                       arrowstyle='->', connectionstyle='arc3,rad=0.1')
nx.draw_networkx_labels(G, pos, font_size=8, font_family='Arial Unicode MS')

plt.title('FlyWire 连接网络图 (Top 30 节点)', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'network_graph.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 网络图已保存")

# ============================================================================
# 3. 连接矩阵热图
# ============================================================================

print("\n" + "="*60)
print("3. 连接矩阵热图")
print("="*60)

# 选择 Top 40 细胞类型
top_40_types = [node[0] for node in sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:40]]
top_40_indices = [type_to_idx[ct] for ct in top_40_types]

sub_matrix = syn_matrix[np.ix_(top_40_indices, top_40_indices)]

# 绘制热图
plt.figure(figsize=(14, 12))
sns.heatmap(sub_matrix, xticklabels=top_40_types, yticklabels=top_40_types,
            cmap='YlOrRd', cbar_kws={'label': '突触数量'}, linewidths=0.5)
plt.title('FlyWire 连接矩阵热图 (Top 40 细胞类型)', fontsize=14, fontweight='bold')
plt.xlabel('目标细胞类型', fontsize=12)
plt.ylabel('源细胞类型', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / 'connection_matrix_heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 连接矩阵热图已保存")

# ============================================================================
# 4. 层次聚类分析
# ============================================================================

print("\n" + "="*60)
print("4. 层次聚类分析")
print("="*60)

# 基于连接模式进行聚类
features = np.hstack([sub_matrix, sub_matrix.T])  # 输入+输出连接
distances = pdist(features, metric='euclidean')
linkage_matrix = hierarchy.linkage(distances, method='ward')

# 绘制树状图
plt.figure(figsize=(16, 8))
dendro = hierarchy.dendrogram(linkage_matrix, labels=top_40_types, 
                              leaf_font_size=10, leaf_rotation=90)
plt.title('基于连接模式的层次聚类', fontsize=14, fontweight='bold')
plt.xlabel('细胞类型', fontsize=12)
plt.ylabel('距离', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'hierarchical_clustering.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 层次聚类图已保存")

# ============================================================================
# 5. 输入-输出通路图
# ============================================================================

print("\n" + "="*60)
print("5. 输入-输出通路分析")
print("="*60)

input_types = flywire_data['input_units']
output_types = flywire_data['output_units']

# 找到从输入到输出的所有中间神经元
intermediate_neurons = set()
for edge in edges:
    if edge['src'] in input_types:
        intermediate_neurons.add(edge['tar'])
    if edge['tar'] in output_types:
        intermediate_neurons.add(edge['src'])

# 移除输入和输出神经元
intermediate_neurons = intermediate_neurons - set(input_types) - set(output_types)

print(f"输入神经元: {len(input_types)}")
print(f"输出神经元: {len(output_types)}")
print(f"中间神经元: {len(intermediate_neurons)}")

# 统计每个中间神经元的连接
intermediate_stats = []
for neuron in intermediate_neurons:
    from_input = sum(1 for e in edges if e['src'] in input_types and e['tar'] == neuron)
    to_output = sum(1 for e in edges if e['src'] == neuron and e['tar'] in output_types)
    if from_input > 0 or to_output > 0:
        intermediate_stats.append({
            'neuron': neuron,
            'from_input': from_input,
            'to_output': to_output,
            'total': from_input + to_output
        })

intermediate_stats = sorted(intermediate_stats, key=lambda x: x['total'], reverse=True)[:20]

# 绘制通路图
fig, ax = plt.subplots(figsize=(14, 10))

y_positions = {
    'input': 0.9,
    'intermediate': 0.5,
    'output': 0.1
}

# 绘制输入神经元
for i, neuron in enumerate(input_types):
    x = i / max(len(input_types) - 1, 1)
    ax.scatter(x, y_positions['input'], s=500, c='green', alpha=0.7, edgecolors='black', linewidths=2)
    ax.text(x, y_positions['input'] + 0.05, neuron, ha='center', fontsize=10, fontweight='bold')

# 绘制输出神经元
for i, neuron in enumerate(output_types):
    x = i / max(len(output_types) - 1, 1)
    ax.scatter(x, y_positions['output'], s=500, c='red', alpha=0.7, edgecolors='black', linewidths=2)
    ax.text(x, y_positions['output'] - 0.05, neuron, ha='center', va='top', fontsize=10, fontweight='bold')

# 绘制关键中间神经元
top_intermediate = [stat['neuron'] for stat in intermediate_stats[:10]]
for i, neuron in enumerate(top_intermediate):
    x = i / max(len(top_intermediate) - 1, 1)
    ax.scatter(x, y_positions['intermediate'], s=300, c='blue', alpha=0.6, edgecolors='black', linewidths=1.5)
    ax.text(x, y_positions['intermediate'], neuron, ha='center', va='center', fontsize=8)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.set_title('输入-输出通路图 (Top 10 中间神经元)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'input_output_pathway.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 输入-输出通路图已保存")

# ============================================================================
# 6. 连接强度分布
# ============================================================================

print("\n" + "="*60)
print("6. 连接强度分布")
print("="*60)

# 统计突触数量分布
synapse_counts = []
for edge in edges:
    if 'offsets' in edge and len(edge['offsets']) > 0:
        syn_count = sum(offset[1] for offset in edge['offsets'])
        synapse_counts.append(syn_count)

if synapse_counts:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    axes[0].hist(synapse_counts, bins=50, color='#4ecdc4', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('突触数量', fontsize=12)
    axes[0].set_ylabel('连接数', fontsize=12)
    axes[0].set_title('突触数量分布', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 对数尺度
    axes[1].hist(synapse_counts, bins=50, color='#ff6b6b', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('突触数量', fontsize=12)
    axes[1].set_ylabel('连接数', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].set_title('突触数量分布 (对数尺度)', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synapse_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ 连接强度分布图已保存")
    print(f"  - 平均突触数: {np.mean(synapse_counts):.1f}")
    print(f"  - 中位数: {np.median(synapse_counts):.1f}")
    print(f"  - 最大值: {np.max(synapse_counts)}")
    print(f"  - 最小值: {np.min(synapse_counts)}")

# ============================================================================
# 7. 总结
# ============================================================================

print("\n" + "="*60)
print("FlyWire 连接可视化总结")
print("="*60)
print(f"\n✓ 生成的可视化:")
print(f"  1. 网络图 (Top 30 节点)")
print(f"  2. 连接矩阵热图 (Top 40 细胞类型)")
print(f"  3. 层次聚类树状图")
print(f"  4. 输入-输出通路图")
print(f"  5. 连接强度分布")
print(f"\n✓ 所有图表已保存到: {output_dir}")
print(f"\n🎉 可视化完成！")
print("="*60)
