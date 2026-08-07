#!/usr/bin/env python
# coding: utf-8

"""
FlyWire 连接组探索

这个脚本展示如何加载和可视化 FlyWire 连接组数据。

目标:
1. 加载 FlyWire 连接组
2. 查看基本统计信息
3. 可视化连接组结构
4. 分析细胞类型和连接
5. 对比 FlyWire 和原始 FIB 数据
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
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置绘图样式
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass  # 使用默认样式
sns.set_palette("husl")

print("✓ 库导入成功")

# ============================================================================
# 1. 加载 FlyWire 连接组数据
# ============================================================================

print("\n" + "="*60)
print("1. 加载 FlyWire 连接组数据")
print("="*60)

# 加载 JSON 数据
json_path = Path("flyvis/connectome/flywire_v1.0.json")

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
plt.savefig('flywire_connection_types.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ 连接类型分布图已生成 (保存为 flywire_connection_types.png)")

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
plt.savefig('flywire_degree_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ 连接度分布图已生成 (保存为 flywire_degree_distribution.png)")

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

# 可视化 T4/T5 神经元的输入
t4_t5_types = [t for t in output_types if t.startswith('T4') or t.startswith('T5')]

# 收集每个 T4/T5 神经元的输入来源
t4_t5_inputs = {}
for t_type in t4_t5_types:
    inputs = [edge['src'] for edge in edges if edge['tar'] == t_type]
    t4_t5_inputs[t_type] = Counter(inputs)

# 绘制热图
if t4_t5_inputs:
    # 获取所有输入类型
    all_inputs = set()
    for inputs in t4_t5_inputs.values():
        all_inputs.update(inputs.keys())
    all_inputs = sorted(all_inputs)
    
    # 创建矩阵
    matrix = np.zeros((len(t4_t5_types), len(all_inputs)))
    for i, t_type in enumerate(t4_t5_types):
        for j, input_type in enumerate(all_inputs):
            matrix[i, j] = t4_t5_inputs[t_type].get(input_type, 0)
    
    # 绘制热图
    plt.figure(figsize=(16, 6))
    sns.heatmap(matrix, xticklabels=all_inputs, yticklabels=t4_t5_types, 
                cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': '连接数'})
    plt.xlabel('输入细胞类型', fontsize=12)
    plt.ylabel('T4/T5 神经元', fontsize=12)
    plt.title('T4/T5 神经元的输入连接', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('flywire_t4t5_inputs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ T4/T5 输入连接热图已生成 (保存为 flywire_t4t5_inputs.png)")

# ============================================================================
# 5. 空间偏移分析
# ============================================================================

print("\n" + "="*60)
print("5. 空间偏移分析")
print("="*60)

# 分析空间偏移模式
offset_patterns = []
for edge in edges:
    if 'offsets' in edge and len(edge['offsets']) > 0:
        for offset in edge['offsets']:
            if len(offset) >= 2 and len(offset[0]) >= 2:
                du, dv = offset[0][:2]
                syn_count = offset[1]
                offset_patterns.append((du, dv, syn_count))

if offset_patterns:
    # 转换为 DataFrame
    offset_df = pd.DataFrame(offset_patterns, columns=['du', 'dv', 'syn_count'])
    
    print(f"空间偏移统计:")
    print(f"  总偏移数: {len(offset_df)}")
    print(f"  唯一偏移模式: {len(offset_df[['du', 'dv']].drop_duplicates())}")
    print(f"\n最常见的偏移模式 (Top 10):")
    print(offset_df.groupby(['du', 'dv']).size().sort_values(ascending=False).head(10))
    
    # 可视化空间偏移分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散点图：空间偏移
    scatter = axes[0].scatter(offset_df['du'], offset_df['dv'], 
                              c=offset_df['syn_count'], s=50, 
                              cmap='viridis', alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('du (六边形坐标)', fontsize=12)
    axes[0].set_ylabel('dv (六边形坐标)', fontsize=12)
    axes[0].set_title('空间偏移分布', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=axes[0], label='突触数量')
    
    # 六边形热图
    offset_counts = offset_df.groupby(['du', 'dv']).size().reset_index(name='count')
    pivot = offset_counts.pivot(index='dv', columns='du', values='count').fillna(0)
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', ax=axes[1], 
                cbar_kws={'label': '连接数'})
    axes[1].set_xlabel('du', fontsize=12)
    axes[1].set_ylabel('dv', fontsize=12)
    axes[1].set_title('偏移模式热图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('flywire_spatial_offsets.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 空间偏移可视化已生成 (保存为 flywire_spatial_offsets.png)")
else:
    print("⚠ 没有找到空间偏移数据")
    offset_df = pd.DataFrame(columns=['du', 'dv', 'syn_count'])

# ============================================================================
# 6. 使用 Flyvis 加载连接组
# ============================================================================

print("\n" + "="*60)
print("6. 使用 Flyvis 加载连接组")
print("="*60)

# 尝试导入 Flyvis
try:
    from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
    from flyvis.connectome import ConnectomeView
    
    print("✓ Flyvis 导入成功")
    flyvis_available = True
except ImportError as e:
    print(f"⚠ Flyvis 未安装或导入失败: {e}")
    print("  请运行: pip install -e .")
    flyvis_available = False

if flyvis_available:
    try:
        # 创建连接组对象
        connectome = ConnectomeFromFlyWire(
            flywire_data_path="flyvis/connectome/flywire_v1.0.json",
            extent=15
        )
        
        print("✓ FlyWire 连接组已加载")
        print(f"\n连接组统计:")
        stats = connectome.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 创建 ConnectomeView
        view = ConnectomeView(connectome, extent=15)
        
        print("\n✓ ConnectomeView 已创建")
        print(f"\n视图信息:")
        print(f"  节点数: {len(view.nodes)}")
        print(f"  边数: {len(view.edges)}")
        print(f"  细胞类型: {list(view.unique_cell_types)[:10]}...")
    except Exception as e:
        print(f"⚠ 创建连接组时出错: {e}")

# ============================================================================
# 7. 对比 FlyWire 和原始 FIB 数据
# ============================================================================

print("\n" + "="*60)
print("7. 对比 FlyWire 和原始 FIB 数据")
print("="*60)

# 加载原始 FIB 连接组（如果存在）
fib_path = Path("flyvis/connectome/fib25-fib19_v2.2.json")

if fib_path.exists():
    with open(fib_path, 'r') as f:
        fib_data = json.load(f)
    
    print("✓ 原始 FIB 连接组已加载")
    print(f"\nFIB 数据结构:")
    print(f"  - 节点数: {len(fib_data['nodes'])}")
    print(f"  - 边数: {len(fib_data['edges'])}")
    
    # 对比统计
    comparison = pd.DataFrame({
        'FlyWire': [
            len(flywire_data['nodes']),
            len(flywire_data['edges']),
            len(flywire_data['input_units']),
            len(flywire_data['output_units'])
        ],
        'FIB': [
            len(fib_data['nodes']),
            len(fib_data['edges']),
            len(fib_data['input_units']),
            len(fib_data['output_units'])
        ]
    }, index=['节点数', '边数', '输入单元', '输出单元'])
    
    print("\n对比:")
    print(comparison)
    
    # 可视化对比
    comparison.plot(kind='bar', figsize=(10, 6), color=['#4ecdc4', '#ff6b6b'])
    plt.title('FlyWire vs FIB 连接组对比', fontsize=14, fontweight='bold')
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='数据集')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('flywire_vs_fib_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ 对比图已生成 (保存为 flywire_vs_fib_comparison.png)")
else:
    print("⚠ 原始 FIB 连接组文件未找到")

# ============================================================================
# 8. 总结
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
if len(offset_df) > 0:
    print(f"\n✓ 空间偏移")
    print(f"  - 总偏移数: {len(offset_df)}")
    print(f"  - 唯一模式: {len(offset_df[['du', 'dv']].drop_duplicates())}")
print(f"\n🎉 探索完成！")
print("="*60)

print("\n生成的图表文件:")
print("  - flywire_connection_types.png")
print("  - flywire_degree_distribution.png")
print("  - flywire_t4t5_inputs.png")
if len(offset_df) > 0:
    print("  - flywire_spatial_offsets.png")
if fib_path.exists():
    print("  - flywire_vs_fib_comparison.png")
