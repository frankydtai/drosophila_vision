#!/usr/bin/env python
# coding: utf-8

"""
FlyWire 连接组探索

这个脚本展示如何加载和可视化 FlyWire 连接组数据。
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
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass
sns.set_palette("husl")

# 设置输出目录
output_dir = Path('outputs/connectome')
output_dir.mkdir(parents=True, exist_ok=True)

print("✓ 库导入成功")
print(f"✓ 输出目录: {output_dir}")

# ============================================================================
# 1. 加载 FlyWire 连接组数据
# ============================================================================

print("\n" + "="*60)
print("1. 加载 FlyWire 连接组数据")
print("="*60)

# 加载 JSON 数据
json_path = Path("../flyvis/connectome/flywire_v1.0.json")

with open(json_path, 'r') as f:
    flywire_data = json.load(f)

print(f"✓ 加载 FlyWire 连接组: {json_path}")
print(f"  文件大小: {json_path.stat().st_size / 1024:.1f} KB")
print(f"\n数据结构:")
print(f"  - 节点数: {len(flywire_data['nodes'])}")
print(f"  - 边数: {len(flywire_data['edges'])}")
print(f"  - 输入单元: {flywire_data['input_units']}")
print(f"  - 输出单元: {flywire_data['output_units']}")

# ============================================================================
# 2. 基本统计信息
# ============================================================================

print("\n" + "="*60)
print("2. 基本统计信息")
print("="*60)

# 提取节点信息
nodes = flywire_data['nodes']
edges = flywire_data['edges']

# 细胞类型列表
cell_types = [node['name'] for node in nodes]
print(f"细胞类型总数: {len(cell_types)}")
print(f"\n前 20 个细胞类型:")
print(cell_types[:20])

# 分析边的统计信息
edge_stats = {
    'total_edges': len(edges),
    'excitatory': sum(1 for e in edges if e.get('alpha', 1) == 1),
    'inhibitory': sum(1 for e in edges if e.get('alpha', 1) == -1),
    'with_offsets': sum(1 for e in edges if len(e.get('offsets', [])) > 0),
}

print("\n边统计:")
print(f"  总边数: {edge_stats['total_edges']}")
print(f"  兴奋性连接: {edge_stats['excitatory']} ({edge_stats['excitatory']/edge_stats['total_edges']*100:.1f}%)")
print(f"  抑制性连接: {edge_stats['inhibitory']} ({edge_stats['inhibitory']/edge_stats['total_edges']*100:.1f}%)")
print(f"  有空间偏移: {edge_stats['with_offsets']} ({edge_stats['with_offsets']/edge_stats['total_edges']*100:.1f}%)")

# 计算每个细胞类型的连接数
in_degree = defaultdict(int)
out_degree = defaultdict(int)
total_synapses = defaultdict(int)

for edge in edges:
    src = edge['src']
    tar = edge['tar']
    out_degree[src] += 1
    in_degree[tar] += 1
    
    # 计算突触总数
    if 'offsets' in edge and len(edge['offsets']) > 0:
        syn_count = sum(offset[1] for offset in edge['offsets'])
    else:
        syn_count = 0
    total_synapses[(src, tar)] = syn_count

print("\n连接度最高的细胞类型:")
print("\n输出连接最多 (Top 10):")
for cell_type, count in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {cell_type}: {count} 个目标")

print("\n输入连接最多 (Top 10):")
for cell_type, count in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {cell_type}: {count} 个来源")

# ============================================================================
# 3. 可视化连接组结构
# ============================================================================

print("\n" + "="*60)
print("3. 可视化连接组结构")
print("="*60)

# 可视化兴奋性 vs 抑制性连接
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 饼图：兴奋性 vs 抑制性
labels = ['兴奋性', '抑制性']
sizes = [edge_stats['excitatory'], edge_stats['inhibitory']]
colors = ['#ff6b6b', '#4ecdc4']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[0].set_title('连接类型分布', fontsize=14, fontweight='bold')

# 柱状图：有/无空间偏移
offset_labels = ['有空间偏移', '无空间偏移']
offset_sizes = [edge_stats['with_offsets'], edge_stats['total_edges'] - edge_stats['with_offsets']]
axes[1].bar(offset_labels, offset_sizes, color=['#95e1d3', '#f38181'])
axes[1].set_ylabel('连接数', fontsize=12)
axes[1].set_title('空间偏移信息', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'connection_types.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 连接类型分布图已生成")

# 可视化连接度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 输入连接度分布
in_degrees = list(in_degree.values())
axes[0].hist(in_degrees, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('输入连接数', fontsize=12)
axes[0].set_ylabel('细胞类型数量', fontsize=12)
axes[0].set_title('输入连接度分布', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# 输出连接度分布
out_degrees = list(out_degree.values())
axes[1].hist(out_degrees, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('输出连接数', fontsize=12)
axes[1].set_ylabel('细胞类型数量', fontsize=12)
axes[1].set_title('输出连接度分布', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'degree_distribution.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"✓ 连接度分布图已生成")

# ============================================================================
# 4. 分析关键神经元类型
# ============================================================================

print("\n" + "="*60)
print("4. 分析关键神经元类型")
print("="*60)

# 分析输入和输出神经元
input_types = flywire_data['input_units']
output_types = flywire_data['output_units']

print("输入神经元（光感受器）:")
for cell_type in input_types:
    out_conn = out_degree.get(cell_type, 0)
    print(f"  {cell_type}: 输出到 {out_conn} 个目标")

print("\n输出神经元（运动检测）:")
for cell_type in output_types:
    in_conn = in_degree.get(cell_type, 0)
    print(f"  {cell_type}: 接收来自 {in_conn} 个来源")

# ============================================================================
# 5. 总结
# ============================================================================

print("\n" + "="*60)
print("FlyWire 连接组探索总结")
print("="*60)
print(f"\n✓ 成功加载 FlyWire 连接组")
print(f"  - 细胞类型: {len(cell_types)} 种")
print(f"  - 连接: {len(edges)} 个")
print(f"  - 兴奋性: {edge_stats['excitatory']} ({edge_stats['excitatory']/edge_stats['total_edges']*100:.1f}%)")
print(f"  - 抑制性: {edge_stats['inhibitory']} ({edge_stats['inhibitory']/edge_stats['total_edges']*100:.1f}%)")
print(f"  - 有空间偏移: {edge_stats['with_offsets']} ({edge_stats['with_offsets']/edge_stats['total_edges']*100:.1f}%)")
print(f"\n✓ 关键神经元")
print(f"  - 输入（光感受器）: {', '.join(input_types)}")
print(f"  - 输出（运动检测）: {', '.join(output_types)}")
print(f"\n🎉 探索完成！")
print("="*60)

print(f"\n生成的图表文件:")
print(f"  - {output_dir / 'connection_types.png'}")
print(f"  - {output_dir / 'degree_distribution.png'}")
