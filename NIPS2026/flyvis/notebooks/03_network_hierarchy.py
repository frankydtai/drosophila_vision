#!/usr/bin/env python
# coding: utf-8

"""
FlyWire 网络层次结构分析

使用广度优先搜索（BFS）从输入神经元（R1-6, R7, R8）出发，
将所有神经元按照距离输入的层级进行分层可视化。
"""

import sys
import json
import numpy as np
import pandas as pd
import warnings

# 过滤字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')

# 在 Jupyter 中不使用 Agg 后端
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx

# 设置中文字体支持
import locale
try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

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
output_dir = Path('outputs/hierarchy')
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
input_types = flywire_data['input_units']
output_types = flywire_data['output_units']

print(f"✓ 加载完成")
print(f"  - 细胞类型: {len(cell_types)}")
print(f"  - 连接: {len(edges)}")
print(f"  - 输入神经元: {input_types}")
print(f"  - 输出神经元: {output_types}")

# ============================================================================
# 2. 构建有向图
# ============================================================================

print("\n" + "="*60)
print("2. 构建有向图")
print("="*60)

# 构建邻接表
adjacency = defaultdict(list)
for edge in edges:
    src = edge['src']
    tar = edge['tar']
    adjacency[src].append(tar)

print(f"✓ 邻接表构建完成")
print(f"  - 节点数: {len(cell_types)}")
print(f"  - 有出边的节点: {len(adjacency)}")

# ============================================================================
# 3. 广度优先搜索（BFS）分层
# ============================================================================

print("\n" + "="*60)
print("3. 广度优先搜索分层")
print("="*60)

def bfs_layering(start_nodes, adjacency):
    """
    从起始节点进行 BFS，返回每个节点的层级
    
    Args:
        start_nodes: 起始节点列表（输入神经元）
        adjacency: 邻接表
    
    Returns:
        layers: dict，节点 -> 层级
        layer_nodes: dict，层级 -> 节点列表
    """
    layers = {}
    queue = deque()
    
    # 初始化：输入神经元在第 0 层
    for node in start_nodes:
        layers[node] = 0
        queue.append(node)
    
    # BFS
    while queue:
        current = queue.popleft()
        current_layer = layers[current]
        
        # 遍历所有邻居
        for neighbor in adjacency.get(current, []):
            if neighbor not in layers:
                layers[neighbor] = current_layer + 1
                queue.append(neighbor)
    
    # 按层级组织节点
    layer_nodes = defaultdict(list)
    for node, layer in layers.items():
        layer_nodes[layer].append(node)
    
    return layers, layer_nodes

# 执行 BFS
node_layers, layer_nodes = bfs_layering(input_types, adjacency)

print(f"✓ BFS 分层完成")
print(f"  - 已分层节点: {len(node_layers)}")
print(f"  - 未到达节点: {len(cell_types) - len(node_layers)}")
print(f"  - 最大层级: {max(node_layers.values()) if node_layers else 0}")

# 打印每层的统计信息
print("\n层级统计:")
for layer in sorted(layer_nodes.keys()):
    nodes_in_layer = layer_nodes[layer]
    print(f"  第 {layer} 层: {len(nodes_in_layer)} 个神经元")
    if layer <= 2:  # 只显示前几层的详细信息
        print(f"    {', '.join(nodes_in_layer[:10])}" + 
              (f" ... (共 {len(nodes_in_layer)} 个)" if len(nodes_in_layer) > 10 else ""))

# 检查输出神经元在哪一层
print("\n输出神经元层级:")
for out_neuron in output_types:
    if out_neuron in node_layers:
        print(f"  {out_neuron}: 第 {node_layers[out_neuron]} 层")
    else:
        print(f"  {out_neuron}: 未到达")

# ============================================================================
# 4. 可视化层次结构
# ============================================================================

print("\n" + "="*60)
print("4. 可视化层次结构")
print("="*60)

# 4.1 层级分布柱状图
fig, ax = plt.subplots(figsize=(12, 6))

layers_list = sorted(layer_nodes.keys())
counts = [len(layer_nodes[layer]) for layer in layers_list]

bars = ax.bar(layers_list, counts, color='#4ecdc4', alpha=0.7, edgecolor='black', linewidth=1.5)

# 标注每个柱子的数值
for i, (layer, count) in enumerate(zip(layers_list, counts)):
    ax.text(layer, count + 0.5, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('层级（距离输入神经元的跳数）', fontsize=12)
ax.set_ylabel('神经元数量', fontsize=12)
ax.set_title('FlyWire 网络层次结构分布', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(layers_list)

plt.tight_layout()
plt.savefig(output_dir / 'layer_distribution.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 层级分布图已保存")

# 4.2 层次结构网络图
print("\n绘制层次结构网络图...")

# 选择每层的代表性节点进行可视化（避免图太复杂）
max_nodes_per_layer = 15
selected_nodes = set(input_types)  # 包含所有输入神经元
selected_nodes.update(output_types)  # 包含所有输出神经元

for layer in sorted(layer_nodes.keys()):
    nodes_in_layer = layer_nodes[layer]
    # 优先选择输出神经元，然后选择连接度高的节点
    priority_nodes = [n for n in nodes_in_layer if n in output_types]
    other_nodes = [n for n in nodes_in_layer if n not in output_types]
    
    # 按出度排序
    other_nodes_sorted = sorted(other_nodes, 
                                key=lambda n: len(adjacency.get(n, [])), 
                                reverse=True)
    
    selected_from_layer = priority_nodes + other_nodes_sorted[:max_nodes_per_layer - len(priority_nodes)]
    selected_nodes.update(selected_from_layer[:max_nodes_per_layer])

# 构建子图
G = nx.DiGraph()
for edge in edges:
    if edge['src'] in selected_nodes and edge['tar'] in selected_nodes:
        G.add_edge(edge['src'], edge['tar'])

print(f"  选择了 {len(selected_nodes)} 个节点进行可视化")
print(f"  包含 {G.number_of_edges()} 条边")

# 使用层次布局
pos = {}
max_layer = max(node_layers.values())

for node in G.nodes():
    if node in node_layers:
        layer = node_layers[node]
        # 获取该层的所有选中节点
        nodes_in_this_layer = [n for n in selected_nodes if node_layers.get(n) == layer]
        # 计算该节点在该层中的位置
        if node in nodes_in_this_layer:
            idx = nodes_in_this_layer.index(node)
            total = len(nodes_in_this_layer)
            # x 坐标：层级，y 坐标：在该层中均匀分布
            pos[node] = (layer, (idx - total/2) * 0.5)

# 绘制
fig, ax = plt.subplots(figsize=(16, 12))

# 节点颜色：根据类型
node_colors = []
for node in G.nodes():
    if node in input_types:
        node_colors.append('#2ecc71')  # 绿色：输入
    elif node in output_types:
        node_colors.append('#e74c3c')  # 红色：输出
    else:
        node_colors.append('#3498db')  # 蓝色：中间

# 节点大小：根据度数
node_sizes = [len(adjacency.get(node, [])) * 30 + 100 for node in G.nodes()]

# 绘制
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                       alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', 
                       arrows=True, arrowsize=10, arrowstyle='->', 
                       connectionstyle='arc3,rad=0.1', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, font_family='Arial Unicode MS', ax=ax)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='输入神经元 (R1-6, R7, R8)'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='输出神经元 (T4/T5)'),
    Patch(facecolor='#3498db', edgecolor='black', label='中间神经元')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title('FlyWire 网络层次结构图\n（从输入神经元出发的 BFS 分层）', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('层级', fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'hierarchy_network.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 层次结构网络图已保存")

# 4.3 桑基图风格的层级流动图
print("\n绘制层级流动图...")

fig, ax = plt.subplots(figsize=(14, 10))

# 计算每层之间的连接数
layer_connections = defaultdict(int)
for edge in edges:
    src, tar = edge['src'], edge['tar']
    if src in node_layers and tar in node_layers:
        src_layer = node_layers[src]
        tar_layer = node_layers[tar]
        if tar_layer > src_layer:  # 只统计前向连接
            layer_connections[(src_layer, tar_layer)] += 1

# 绘制每一层
y_positions = {}
for layer in sorted(layer_nodes.keys()):
    count = len(layer_nodes[layer])
    y_positions[layer] = count
    
    # 绘制矩形表示该层
    rect = plt.Rectangle((layer - 0.3, 0), 0.6, count, 
                         facecolor='#4ecdc4', edgecolor='black', 
                         linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    
    # 标注
    ax.text(layer, count + 2, f'第 {layer} 层\n{count} 个神经元', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')

# 绘制层间连接（简化版）
for (src_layer, tar_layer), count in layer_connections.items():
    if count > 10:  # 只显示连接数较多的
        src_y = y_positions[src_layer] / 2
        tar_y = y_positions[tar_layer] / 2
        
        # 绘制箭头
        ax.annotate('', xy=(tar_layer - 0.3, tar_y), 
                   xytext=(src_layer + 0.3, src_y),
                   arrowprops=dict(arrowstyle='->', lw=count/50, 
                                 alpha=0.3, color='gray'))

ax.set_xlim(-0.5, max(layer_nodes.keys()) + 0.5)
ax.set_ylim(0, max(y_positions.values()) + 10)
ax.set_xlabel('层级', fontsize=12)
ax.set_ylabel('神经元数量', fontsize=12)
ax.set_title('FlyWire 网络层级流动图', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'layer_flow.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 层级流动图已保存")

# ============================================================================
# 5. 导出层级信息
# ============================================================================

print("\n" + "="*60)
print("5. 导出层级信息")
print("="*60)

# 保存为 CSV
layer_data = []
for node, layer in node_layers.items():
    layer_data.append({
        'neuron': node,
        'layer': layer,
        'is_input': node in input_types,
        'is_output': node in output_types,
        'out_degree': len(adjacency.get(node, []))
    })

df = pd.DataFrame(layer_data)
df = df.sort_values(['layer', 'neuron'])
csv_path = output_dir / 'neuron_layers.csv'
df.to_csv(csv_path, index=False)

print(f"✓ 层级信息已保存到: {csv_path}")
print(f"  - 总行数: {len(df)}")

# 保存为 JSON（用于后续分析）
layer_info = {
    'layers': {int(k): v for k, v in layer_nodes.items()},
    'node_layers': node_layers,
    'max_layer': max(node_layers.values()),
    'input_neurons': input_types,
    'output_neurons': output_types,
    'unreachable_neurons': [n for n in cell_types if n not in node_layers]
}

json_path = output_dir / 'layer_info.json'
with open(json_path, 'w') as f:
    json.dump(layer_info, f, indent=2)

print(f"✓ 层级信息已保存到: {json_path}")

# ============================================================================
# 6. 总结
# ============================================================================

print("\n" + "="*60)
print("FlyWire 网络层次结构分析总结")
print("="*60)

print(f"\n✓ BFS 分层完成")
print(f"  - 起始节点: {', '.join(input_types)}")
print(f"  - 已分层节点: {len(node_layers)} / {len(cell_types)}")
print(f"  - 未到达节点: {len(cell_types) - len(node_layers)}")
print(f"  - 层级数: {len(layer_nodes)}")
print(f"  - 最大层级: {max(node_layers.values())}")

print(f"\n✓ 生成的可视化:")
print(f"  1. 层级分布柱状图")
print(f"  2. 层次结构网络图")
print(f"  3. 层级流动图")

print(f"\n✓ 导出的数据:")
print(f"  1. neuron_layers.csv - 每个神经元的层级信息")
print(f"  2. layer_info.json - 完整的层级结构数据")

print(f"\n✓ 所有文件已保存到: {output_dir}")
print(f"\n🎉 分析完成！")
print("="*60)
