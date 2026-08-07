# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # LC 视觉投射神经元 — 连接可视化
#
# 本 notebook 使用 networkx 和 matplotlib 对 FlyWire v2.0 连接组中的 LC 神经元连接进行可视化：
# 1. 不同突触阈值下的网络图
# 2. LC 神经元接收/发出连接热图
# 3. T4/T5 vs LC 输出通路对比
# 4. 交互式（plotly）连接网络

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 路径
BASE_DIR = Path('/Users/lengyuner/Desktop/NIPS2026/flyvis')
JSON_PATH = BASE_DIR / 'flyvis/connectome/flywire_v2.0.json'
FIG_DIR = BASE_DIR / 'notebooks/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("✓ 环境加载完成")
print(f"  连接组文件: {JSON_PATH}")
print(f"  图像输出目录: {FIG_DIR}")

# +
# ============================================================
# 加载连接组
# ============================================================

with open(JSON_PATH) as f:
    data = json.load(f)

nodes = data['nodes']
edges = data['edges']
input_units = data['input_units']
output_units = data['output_units']

# 分类输出神经元
T45_TYPES = [t for t in output_units if t.startswith('T4') or t.startswith('T5')]
LC_TYPES  = [t for t in output_units if t.startswith('LC') or t.startswith('LCe') or t.startswith('LPLC')]
INT_TYPES = [n['name'] for n in nodes
             if n['name'] not in input_units and n['name'] not in output_units]

print(f"\n连接组统计:")
print(f"  细胞类型总数: {len(nodes)}")
print(f"  连接数: {len(edges)}")
print(f"  输入神经元 ({len(input_units)}): {input_units}")
print(f"  T4/T5 输出 ({len(T45_TYPES)}): {T45_TYPES}")
print(f"  LC 输出 ({len(LC_TYPES)}): {LC_TYPES}")
print(f"  中间神经元: {len(INT_TYPES)} 种")
# -

# +
# ============================================================
# 构建突触强度矩阵
# ============================================================

def edge_synapse_count(edge):
    """计算一条边的总突触数"""
    return sum(o[2] for o in edge['offsets']) if edge['offsets'] else 0

# 建立 (src, tar) -> 突触总数 和 sign
edge_dict = {}  # (src,tar) -> {'syn':int, 'sign':int}
for e in edges:
    key = (e['src'], e['tar'])
    syn = edge_synapse_count(e)
    if key not in edge_dict or syn > edge_dict[key]['syn']:
        edge_dict[key] = {'syn': syn, 'sign': e['alpha']}

print(f"\n唯一连接对: {len(edge_dict)}")
print(f"  突触数范围: {min(v['syn'] for v in edge_dict.values())} ~ {max(v['syn'] for v in edge_dict.values())}")
# -

# +
# ============================================================
# 颜色映射
# ============================================================

NODE_COLORS = {
    'input':  '#2ECC71',   # 绿色 - 输入
    'T45':    '#3498DB',   # 蓝色 - T4/T5 运动
    'LC':     '#E74C3C',   # 红色 - LC 视觉投射
    'inter':  '#95A5A6',   # 灰色 - 中间神经元
}

def node_color(name):
    if name in input_units:  return NODE_COLORS['input']
    if name in T45_TYPES:    return NODE_COLORS['T45']
    if name in LC_TYPES:     return NODE_COLORS['LC']
    return NODE_COLORS['inter']

def edge_color(sign):
    return '#C0392B' if sign < 0 else '#2980B9'
# -

# +
# ============================================================
# 图1: 不同突触阈值下的全局网络图（只展示 LC 相关子图）
# ============================================================

try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'networkx', '-q'])
    import networkx as nx

THRESHOLDS = [50, 200, 500, 1000]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('不同突触阈值下的 LC 相关连接网络', fontsize=16, fontweight='bold')

# 只展示涉及 LC 的节点（LC + 其直接前置/后置）
for ax, thresh in zip(axes.flatten(), THRESHOLDS):
    # 找涉及 LC 的连接
    lc_edges = [
        (src, tar, d)
        for (src, tar), d in edge_dict.items()
        if d['syn'] >= thresh and (src in LC_TYPES or tar in LC_TYPES)
    ]

    if not lc_edges:
        ax.text(0.5, 0.5, f'阈值 {thresh}\n无连接', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title(f'阈值 ≥ {thresh} 突触')
        ax.axis('off')
        continue

    # 收集涉及的节点
    involved = set()
    for src, tar, _ in lc_edges:
        involved.add(src)
        involved.add(tar)

    G = nx.DiGraph()
    G.add_nodes_from(involved)
    for src, tar, d in lc_edges:
        G.add_edge(src, tar, weight=d['syn'], sign=d['sign'])

    # 分层布局：输入 -> 中间 -> LC
    pos = {}
    input_nodes = [n for n in involved if n in input_units]
    inter_nodes  = [n for n in involved if n in INT_TYPES]
    lc_nodes     = [n for n in involved if n in LC_TYPES]
    t45_nodes    = [n for n in involved if n in T45_TYPES]

    def spread(names, x, y_range=(-1,1)):
        n = len(names)
        ys = np.linspace(y_range[0], y_range[1], max(n,1))
        return {name: (x, y) for name, y in zip(names, ys)}

    pos.update(spread(input_nodes, 0))
    pos.update(spread(inter_nodes,  1))
    pos.update(spread(t45_nodes,    2, (-0.4, 0.4)))
    pos.update(spread(lc_nodes,     2, (0.5, 1.0) if t45_nodes else (-1,1)))

    colors = [node_color(n) for n in G.nodes()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    ecolors = [edge_color(G[u][v]['sign']) for u, v in G.edges()]
    widths = [0.5 + 3.5 * w / max_w for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=300, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecolors,
                           width=widths, alpha=0.7,
                           arrows=True, arrowsize=12,
                           connectionstyle='arc3,rad=0.1')

    ax.set_title(f'阈值 ≥ {thresh} 突触  (节点:{G.number_of_nodes()}, 边:{G.number_of_edges()})',
                 fontsize=11)
    ax.axis('off')

# 图例
legend_handles = [
    mpatches.Patch(color=NODE_COLORS['input'], label='输入 (R1-8)'),
    mpatches.Patch(color=NODE_COLORS['T45'],   label='T4/T5 运动检测'),
    mpatches.Patch(color=NODE_COLORS['LC'],    label='LC 视觉投射'),
    mpatches.Patch(color=NODE_COLORS['inter'], label='中间神经元'),
    mpatches.Patch(color='#2980B9', label='兴奋性突触'),
    mpatches.Patch(color='#C0392B', label='抑制性突触'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(FIG_DIR / 'lc_network_thresholds.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ 图1 已保存: lc_network_thresholds.png")
# -

# +
# ============================================================
# 图2: LC 神经元接收突触热图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 左图：中间神经元 -> LC
# 只保留向 LC 发出连接的中间神经元（top N）
top_n = 30
lc_incoming = defaultdict(dict)
for (src, tar), d in edge_dict.items():
    if tar in LC_TYPES:
        lc_incoming[src][tar] = d['syn']

# 按总输出突触数排序，取 top_n
src_totals = {src: sum(v.values()) for src, v in lc_incoming.items()}
top_srcs = sorted(src_totals, key=src_totals.get, reverse=True)[:top_n]

# 构建矩阵
mat_in = np.zeros((len(top_srcs), len(LC_TYPES)))
for i, src in enumerate(top_srcs):
    for j, tar in enumerate(LC_TYPES):
        mat_in[i, j] = lc_incoming[src].get(tar, 0)

im = axes[0].imshow(mat_in, aspect='auto', cmap='YlOrRd', interpolation='nearest')
axes[0].set_xticks(range(len(LC_TYPES)))
axes[0].set_xticklabels(LC_TYPES, rotation=60, ha='right', fontsize=8)
axes[0].set_yticks(range(len(top_srcs)))
axes[0].set_yticklabels(top_srcs, fontsize=7)
axes[0].set_title(f'中间神经元 → LC (Top {top_n} 来源)', fontsize=12)
axes[0].set_xlabel('LC 类型')
axes[0].set_ylabel('前置神经元')
plt.colorbar(im, ax=axes[0], label='突触数')

# 右图：LC 发出的连接（LC -> X）
lc_outgoing = defaultdict(dict)
for (src, tar), d in edge_dict.items():
    if src in LC_TYPES:
        lc_outgoing[src][tar] = d['syn']

top_tars_set = set()
for d in lc_outgoing.values():
    top_tars_set.update(d.keys())
top_tars = sorted(top_tars_set)

mat_out = np.zeros((len(LC_TYPES), len(top_tars)))
for i, src in enumerate(LC_TYPES):
    for j, tar in enumerate(top_tars):
        mat_out[i, j] = lc_outgoing[src].get(tar, 0)

im2 = axes[1].imshow(mat_out, aspect='auto', cmap='PuBu', interpolation='nearest')
axes[1].set_yticks(range(len(LC_TYPES)))
axes[1].set_yticklabels(LC_TYPES, fontsize=8)
axes[1].set_xticks(range(len(top_tars)))
axes[1].set_xticklabels(top_tars, rotation=60, ha='right', fontsize=7)
axes[1].set_title('LC → 后置神经元', fontsize=12)
axes[1].set_xlabel('后置神经元')
axes[1].set_ylabel('LC 类型')
plt.colorbar(im2, ax=axes[1], label='突触数')

plt.suptitle('LC 视觉投射神经元 连接热图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'lc_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ 图2 已保存: lc_heatmap.png")
# -

# +
# ============================================================
# 图3: T4/T5 vs LC 输出通路对比（按输入突触总数）
# ============================================================

def total_incoming_syn(ntype):
    return sum(d['syn'] for (src, tar), d in edge_dict.items() if tar == ntype)

def total_outgoing_syn(ntype):
    return sum(d['syn'] for (src, tar), d in edge_dict.items() if src == ntype)

t45_in  = [total_incoming_syn(t) for t in T45_TYPES]
t45_out = [total_outgoing_syn(t) for t in T45_TYPES]
lc_in   = [total_incoming_syn(t) for t in LC_TYPES]
lc_out  = [total_outgoing_syn(t) for t in LC_TYPES]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# T4/T5
ax = axes[0]
x = np.arange(len(T45_TYPES))
bars_in  = ax.bar(x - 0.2, t45_in,  0.4, label='接收突触', color='#3498DB', alpha=0.8)
bars_out = ax.bar(x + 0.2, t45_out, 0.4, label='发出突触', color='#85C1E9', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(T45_TYPES, rotation=30, ha='right')
ax.set_title('T4/T5 运动检测神经元 突触强度', fontsize=12)
ax.set_ylabel('总突触数')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))

# LC
ax = axes[1]
lc_labels = [t[:8] for t in LC_TYPES]  # 缩短标签
x = np.arange(len(LC_TYPES))
bars_in  = ax.bar(x - 0.2, lc_in,  0.4, label='接收突触', color='#E74C3C', alpha=0.8)
bars_out = ax.bar(x + 0.2, lc_out, 0.4, label='发出突触', color='#F1948A', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(lc_labels, rotation=45, ha='right', fontsize=8)
ax.set_title('LC 视觉投射神经元 突触强度', fontsize=12)
ax.set_ylabel('总突触数')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))

plt.suptitle('T4/T5 vs LC 输出通路突触强度对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 't45_vs_lc_strength.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ 图3 已保存: t45_vs_lc_strength.png")
# -

# +
# ============================================================
# 图4: Plotly 交互式网络图（LC 子图，阈值可调）
# ============================================================

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("提示: 安装 plotly 可获得交互式图表: pip install plotly")

if PLOTLY_AVAILABLE:
    THRESH = 200  # 交互图使用的阈值

    # 收集 LC 相关的连接
    lc_relevant = {
        (src, tar): d
        for (src, tar), d in edge_dict.items()
        if d['syn'] >= THRESH and (src in LC_TYPES or tar in LC_TYPES)
    }

    involved = set()
    for src, tar in lc_relevant:
        involved.add(src)
        involved.add(tar)
    involved = list(involved)

    # 构建 networkx 图以获取布局
    G_plt = nx.DiGraph()
    G_plt.add_nodes_from(involved)
    for (src, tar), d in lc_relevant.items():
        G_plt.add_edge(src, tar, weight=d['syn'], sign=d['sign'])

    pos_plt = nx.spring_layout(G_plt, seed=42, k=2.0)

    # 节点
    nx_labels = list(G_plt.nodes())
    nx_x = [pos_plt[n][0] for n in nx_labels]
    nx_y = [pos_plt[n][1] for n in nx_labels]
    nx_colors = [node_color(n) for n in nx_labels]
    nx_sizes  = [20 if n in INT_TYPES else 30 for n in nx_labels]

    node_trace = go.Scatter(
        x=nx_x, y=nx_y,
        mode='markers+text',
        text=nx_labels,
        textposition='top center',
        textfont=dict(size=8),
        marker=dict(color=nx_colors, size=nx_sizes, line=dict(width=1, color='white')),
        hovertemplate='<b>%{text}</b><extra></extra>',
    )

    # 边
    edge_traces = []
    for (src, tar), d in lc_relevant.items():
        x0, y0 = pos_plt[src]
        x1, y1 = pos_plt[tar]
        color = '#C0392B' if d['sign'] < 0 else '#2980B9'
        width = 0.5 + 4 * d['syn'] / max(v['syn'] for v in lc_relevant.values())
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            opacity=0.6,
            hoverinfo='none',
        ))

    fig_plotly = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=f'LC 视觉投射神经元网络 (突触阈值 ≥ {THRESH})',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000, height=700,
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
            font=dict(color='white'),
        )
    )

    html_path = FIG_DIR / 'lc_network_interactive.html'
    fig_plotly.write_html(str(html_path))
    print(f"✓ 图4 已保存 (交互式): {html_path}")
else:
    print("跳过 Plotly 交互式图（未安装）")
# -

# +
# ============================================================
# 图5: LC 类型按突触强度排序的条形图（分接收来源类别）
# ============================================================

# 统计每个 LC 接收的来源类别
lc_source_breakdown = {}  # lc_type -> {from_input, from_inter, from_lc, from_t45}
for lc in LC_TYPES:
    from_input = sum(d['syn'] for (src,tar),d in edge_dict.items()
                     if tar==lc and src in input_units)
    from_t45   = sum(d['syn'] for (src,tar),d in edge_dict.items()
                     if tar==lc and src in T45_TYPES)
    from_lc    = sum(d['syn'] for (src,tar),d in edge_dict.items()
                     if tar==lc and src in LC_TYPES)
    from_inter = sum(d['syn'] for (src,tar),d in edge_dict.items()
                     if tar==lc and src in INT_TYPES)
    lc_source_breakdown[lc] = {
        'from_input': from_input,
        'from_inter': from_inter,
        'from_t45':   from_t45,
        'from_lc':    from_lc,
    }

df_lc = pd.DataFrame(lc_source_breakdown).T
df_lc['total'] = df_lc.sum(axis=1)
df_lc = df_lc.sort_values('total', ascending=False)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(df_lc))
w = 0.6

bottom = np.zeros(len(df_lc))
colors_src = {
    'from_input': '#2ECC71',
    'from_inter': '#95A5A6',
    'from_t45':   '#3498DB',
    'from_lc':    '#E74C3C',
}
labels_src = {
    'from_input': '来自光感受器 (R1-8)',
    'from_inter': '来自中间神经元',
    'from_t45':   '来自 T4/T5',
    'from_lc':    '来自其他 LC',
}
for col in ['from_input', 'from_inter', 'from_t45', 'from_lc']:
    vals = df_lc[col].values
    ax.bar(x, vals, w, bottom=bottom,
           color=colors_src[col], label=labels_src[col], alpha=0.85)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(df_lc.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('接收突触总数')
ax.set_title('各 LC 神经元接收突触来源分解', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
plt.tight_layout()
plt.savefig(FIG_DIR / 'lc_source_breakdown.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ 图5 已保存: lc_source_breakdown.png")
# -

# +
# ============================================================
# 图6: 阈值扫描 — 随阈值增加，LC 相关边数变化
# ============================================================

all_syn = sorted([d['syn'] for d in edge_dict.values()])
thresh_range = [10, 50, 100, 200, 300, 500, 700, 1000, 2000, 5000]

lc_edge_counts  = []
all_edge_counts = []
lc_node_counts  = []

for t in thresh_range:
    lc_e  = [(s,r) for (s,r),d in edge_dict.items() if d['syn']>=t and (s in LC_TYPES or r in LC_TYPES)]
    all_e = [(s,r) for (s,r),d in edge_dict.items() if d['syn']>=t]
    lc_nodes = set()
    for s,r in lc_e:
        if s in LC_TYPES: lc_nodes.add(s)
        if r in LC_TYPES: lc_nodes.add(r)
    lc_edge_counts.append(len(lc_e))
    all_edge_counts.append(len(all_e))
    lc_node_counts.append(len(lc_nodes))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(thresh_range, all_edge_counts, 'o-', color='#2C3E50', label='全部连接')
axes[0].plot(thresh_range, lc_edge_counts,  's-', color='#E74C3C', label='LC 相关连接')
axes[0].set_xlabel('突触阈值')
axes[0].set_ylabel('连接数')
axes[0].set_title('阈值 vs 连接数')
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresh_range, lc_node_counts, 'D-', color='#E74C3C')
axes[1].set_xlabel('突触阈值')
axes[1].set_ylabel('LC 类型数')
axes[1].set_title('阈值 vs 有连接的 LC 类型数')
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)

plt.suptitle('突触阈值对 LC 网络的影响', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'lc_threshold_sweep.png', bbox_inches='tight', dpi=150)
plt.close()
print("✓ 图6 已保存: lc_threshold_sweep.png")

print("\n" + "="*50)
print("全部图像已保存到:", FIG_DIR)
print("="*50)
# -
