#!/usr/bin/env python3
"""
FlyWire 扩展连接组构建

在原有 Motion/Color/OFF/Photoreceptors 基础上，加入 LC 视觉投射神经元，
生成包含更丰富输出层的 flywire_v2.0.json

LC 神经元是 visual projection neurons，将视觉信息从视叶传递到中央脑：
- LC4:  迫近检测（碰撞回避）
- LC6:  小目标运动（猎物捕捉）
- LC9:  迫近检测
- LC10a: 小目标运动
- LC11: 宽场运动
- LC16: 迫近检测
- LC17: 宽场运动
"""

import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/lengyuner/Desktop/data/flywire/Jun2025")
OUTPUT_PATH = Path("/Users/lengyuner/Desktop/NIPS2026/flyvis/flyvis/connectome/flywire_v2.0.json")

# ============================================================
# 神经元配置
# ============================================================

# 输入神经元（光感受器）
INPUT_TYPES = ['R1-6', 'R7', 'R8']

# T4/T5 运动检测输出（与原版相同）
MOTION_OUTPUT_TYPES = [
    'T4a', 'T4b', 'T4c', 'T4d',
    'T5a', 'T5b', 'T5c', 'T5d'
]

# LC 输出神经元（视觉投射，传递到中央脑）
# 包含右半球数量 >= 20 的所有 LC 类型，以及全部 LPLC 类型
LC_OUTPUT_TYPES = [
    # --- 高数量 LC (>= 50 neurons, right) ---
    'LC12',   # unknown function, most numerous
    'LC17',   # wide-field motion
    'LC10a',  # small target motion
    'LC10c',  # small target motion variant
    'LC10d',  # small target motion variant
    'LC9',    # looming response
    'LC18',   # unknown
    'LC16',   # looming detection
    'LC13',   # unknown
    'LC21',   # unknown
    'LC11',   # wide-field motion
    'LC6',    # small moving objects / prey capture
    'LC28a',  # unknown
    'LC15',   # unknown
    'LC10e',  # small target motion variant
    'LC4',    # looming / collision avoidance
    'LC22',   # unknown
    'LC10b',  # small target motion variant
    # --- 中等数量 LC (20-49 neurons, right) ---
    'LC24',   # unknown
    'LC26',   # unknown
    'LC25',   # unknown
    'LCe02',  # lobula complex efferent
    'LC20a',  # unknown
    'LCe03',  # lobula complex efferent
    # --- LPLC (lobula plate / lobula complex) ---
    'LPLC2',  # looming detection
    'LPLC1',  # looming detection
    'LPLC4',  # looming detection
]

OUTPUT_TYPES = MOTION_OUTPUT_TYPES + LC_OUTPUT_TYPES

# 要包含的 subsystem（中间神经元）
SUBSYSTEMS = ['Motion', 'Color', 'OFF', 'Photoreceptors', 'ON']

# 神经递质 -> 突触符号
NT_TO_SIGN = {
    'ACH': 1, 'GLUT': 1, 'SER': 1, 'DA': 1, 'OCT': 1,
    'GABA': -1,
}

MIN_SYN_COUNT = 5  # 最小突触数阈值


# ============================================================
# 1. 加载数据
# ============================================================

def load_data():
    logger.info("加载 FlyWire 数据...")

    vnt = pd.read_csv(DATA_DIR / 'visual_neuron_types.csv.gz')
    cols = pd.read_csv(DATA_DIR / 'column_assignment.csv.gz')
    conns = pd.read_csv(DATA_DIR / 'connections.csv')

    logger.info(f"  视觉神经元: {len(vnt)}")
    logger.info(f"  列分配: {len(cols)}")
    logger.info(f"  连接: {len(conns)}")

    return vnt, cols, conns


# ============================================================
# 2. 筛选神经元
# ============================================================

def select_neurons(vnt: pd.DataFrame) -> pd.DataFrame:
    """
    选择右半球的神经元:
    - subsystem 过滤得到的中间/输入神经元
    - 直接按名字添加 LC 输出神经元
    """
    # 按 subsystem 过滤（右半球）
    mask_subsys = (
        vnt['subsystem'].isin(SUBSYSTEMS) &
        (vnt['side'] == 'right')
    )
    neurons_subsys = vnt[mask_subsys].copy()

    # 按名字添加 LC 输出神经元（右半球）
    mask_lc = (
        vnt['type'].isin(LC_OUTPUT_TYPES) &
        (vnt['side'] == 'right')
    )
    neurons_lc = vnt[mask_lc].copy()

    # 合并，去重
    neurons = pd.concat([neurons_subsys, neurons_lc], ignore_index=True)
    neurons = neurons.drop_duplicates(subset='root_id')

    logger.info(f"\n选中神经元:")
    logger.info(f"  subsystem 过滤: {len(neurons_subsys)}")
    logger.info(f"  LC 输出神经元: {len(neurons_lc)}")
    logger.info(f"  合计（去重后）: {len(neurons)}")
    logger.info(f"  细胞类型数: {neurons['type'].nunique()}")

    return neurons


# ============================================================
# 3. 过滤连接
# ============================================================

def filter_connections(
    conns: pd.DataFrame,
    neurons: pd.DataFrame,
    cols: pd.DataFrame
) -> pd.DataFrame:
    """
    只保留两端都在选定神经元集合内的连接
    """
    neuron_ids = set(neurons['root_id'])

    # 添加 type 信息
    id_to_type = dict(zip(neurons['root_id'], neurons['type']))

    mask = (
        conns['pre_root_id'].isin(neuron_ids) &
        conns['post_root_id'].isin(neuron_ids) &
        (conns['syn_count'] >= MIN_SYN_COUNT)
    )
    filtered = conns[mask].copy()
    filtered['pre_type'] = filtered['pre_root_id'].map(id_to_type)
    filtered['post_type'] = filtered['post_root_id'].map(id_to_type)

    logger.info(f"\n过滤后的连接: {len(filtered)}")
    logger.info(f"  独特类型对: {filtered.groupby(['pre_type','post_type']).ngroups}")

    return filtered


# ============================================================
# 4. 计算空间偏移
# ============================================================

def compute_offsets(
    conns: pd.DataFrame,
    cols: pd.DataFrame
) -> Dict:
    """
    计算类型对的空间偏移（du, dv）
    """
    # 构建 root_id -> (p, q) 映射
    cols_right = cols[cols['hemisphere'] == 'right'] if 'hemisphere' in cols.columns else cols
    id_to_pq = dict(zip(cols_right['root_id'], zip(cols_right['p'], cols_right['q'])))

    offset_dict = defaultdict(list)

    for _, row in conns.iterrows():
        pre_pq = id_to_pq.get(row['pre_root_id'])
        post_pq = id_to_pq.get(row['post_root_id'])

        if pre_pq is None or post_pq is None:
            continue

        dp = int(post_pq[0] - pre_pq[0])
        dq = int(post_pq[1] - pre_pq[1])
        key = (row['pre_type'], row['post_type'])
        offset_dict[key].append((dp, dq, int(row['syn_count'])))

    return offset_dict


# ============================================================
# 5. 聚合为 Flyvis 格式
# ============================================================

def aggregate_edges(
    conns: pd.DataFrame,
    offset_dict: Dict
) -> List[Dict]:
    """
    按 (pre_type, post_type, nt_type) 分组，聚合偏移
    """
    # 按类型对聚合
    grouped = conns.groupby(['pre_type', 'post_type', 'nt_type']).agg(
        total_syn=('syn_count', 'sum'),
        n_connections=('syn_count', 'count')
    ).reset_index()

    edges = []
    for _, row in grouped.iterrows():
        src = row['pre_type']
        tar = row['post_type']
        nt = str(row['nt_type']).upper() if pd.notna(row['nt_type']) else 'ACH'
        sign = NT_TO_SIGN.get(nt, 1)

        key = (src, tar)
        offsets_raw = offset_dict.get(key, [])

        # 聚合偏移：按 (du, dv) 分组求和
        offset_agg = defaultdict(int)
        for dp, dq, syn in offsets_raw:
            offset_agg[(dp, dq)] += syn

        offsets_list = [[dp, dq, syn] for (dp, dq), syn in offset_agg.items()]

        # 如果没有空间信息，用 (0, 0) 占位
        if not offsets_list:
            offsets_list = [[0, 0, int(row['total_syn'])]]

        edges.append({
            'src': src,
            'tar': tar,
            'alpha': sign,
            'offsets': offsets_list
        })

    return edges


# ============================================================
# 6. 构建节点列表
# ============================================================

def build_nodes(cell_types: List[str]) -> List[Dict]:
    nodes = []
    for ct in sorted(cell_types):
        nodes.append({
            'name': ct,
            'pattern': ['stride', [1, 1]],
            'activation': 'relu',
            'bias': 0.5,
            'bias_fixed': False,
            'time_constant': None,
            'time_constant_fixed': False
        })
    return nodes


# ============================================================
# 主流程
# ============================================================

def main():
    print("="*60)
    print("FlyWire v2.0 扩展连接组构建")
    print("新增输出神经元：LC4, LC6, LC9, LC10a/b, LC11, LC16, LC17")
    print("="*60)

    # 1. 加载数据
    vnt, cols, conns = load_data()

    # 2. 选择神经元
    neurons = select_neurons(vnt)
    neuron_types = sorted(neurons['type'].unique().tolist())
    logger.info(f"\n细胞类型列表 ({len(neuron_types)}个):")
    logger.info(str(neuron_types[:30]) + (" ..." if len(neuron_types) > 30 else ""))

    # 3. 过滤连接
    filtered_conns = filter_connections(conns, neurons, cols)

    # 4. 计算空间偏移
    logger.info("\n计算空间偏移...")
    offset_dict = compute_offsets(filtered_conns, cols)
    logger.info(f"  有偏移信息的类型对: {len(offset_dict)}")

    # 5. 聚合为边
    logger.info("\n聚合连接...")
    edges = aggregate_edges(filtered_conns, offset_dict)
    logger.info(f"  边数量: {len(edges)}")

    # 6. 构建节点
    nodes = build_nodes(neuron_types)

    # 7. 确认输入输出
    actual_input = [t for t in INPUT_TYPES if t in neuron_types]
    actual_output = [t for t in OUTPUT_TYPES if t in neuron_types]

    logger.info(f"\n输入神经元: {actual_input}")
    logger.info(f"输出神经元: {actual_output}")

    # 8. 组装最终数据
    flyvis_data = {
        'nodes': nodes,
        'edges': edges,
        'input_units': actual_input,
        'output_units': actual_output,
        'metadata': {
            'version': '2.0',
            'source': 'FlyWire Jun2025',
            'n_cell_types': len(nodes),
            'n_edges': len(edges),
            'subsystems': SUBSYSTEMS,
            'min_syn_count': MIN_SYN_COUNT,
            'lc_output_types': LC_OUTPUT_TYPES,
            'description': 'Extended connectome with LC visual projection neurons as additional outputs'
        }
    }

    # 9. 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(flyvis_data, f, indent=2)

    print("\n" + "="*60)
    print("构建完成！")
    print("="*60)
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"细胞类型数: {len(nodes)}")
    print(f"连接数: {len(edges)}")
    print(f"输入神经元: {actual_input}")
    print(f"输出神经元: {actual_output}")
    print(f"  - T4/T5 运动检测: {[t for t in MOTION_OUTPUT_TYPES if t in neuron_types]}")
    print(f"  - LC 视觉投射: {[t for t in LC_OUTPUT_TYPES if t in neuron_types]}")

    # 统计 LC 相关的连接
    lc_edges_in = [e for e in edges if e['tar'] in LC_OUTPUT_TYPES]
    lc_edges_out = [e for e in edges if e['src'] in LC_OUTPUT_TYPES]
    print(f"\nLC 神经元接收的连接: {len(lc_edges_in)}")
    print(f"LC 神经元发出的连接: {len(lc_edges_out)}")

    # 打印 LC 接收连接的来源
    if lc_edges_in:
        print("\nLC 神经元接收连接的来源（Top 10）:")
        from collections import Counter
        src_counter = Counter(e['src'] for e in lc_edges_in)
        for src, cnt in src_counter.most_common(10):
            print(f"  {src} -> LC: {cnt} 条连接")

    return flyvis_data


if __name__ == '__main__':
    main()
