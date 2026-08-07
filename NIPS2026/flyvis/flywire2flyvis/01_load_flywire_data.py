"""
FlyWire 真实数据加载器

使用实际的 FlyWire 数据文件构建 Flyvis 兼容的连接组。
数据来源: /Users/lengyuner/Desktop/data/flywire/Jun2025/
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlyWireRealDataLoader:
    """加载真实的 FlyWire 数据"""
    
    def __init__(self, data_dir: str = "/Users/lengyuner/Desktop/data/flywire/Jun2025"):
        """初始化数据加载器
        
        Args:
            data_dir: FlyWire 数据目录
        """
        self.data_dir = Path(data_dir)
        self.connections_file = self.data_dir / "connections.csv"
        self.visual_types_file = self.data_dir / "visual_neuron_types.csv.gz"
        self.column_file = self.data_dir / "column_assignment.csv.gz"
        self.classification_file = self.data_dir / "classification.csv.gz"
        
        logger.info(f"FlyWire 数据目录: {self.data_dir}")
        
    def load_visual_neurons(self) -> pd.DataFrame:
        """加载视觉神经元类型数据
        
        Returns:
            包含视觉神经元信息的 DataFrame
            列: root_id, type, family, subsystem, category, side
        """
        logger.info("加载视觉神经元类型数据...")
        df = pd.read_csv(self.visual_types_file, compression='gzip')
        logger.info(f"  - 加载了 {len(df)} 个视觉神经元")
        logger.info(f"  - 细胞类型数量: {df['type'].nunique()}")
        logger.info(f"  - 主要类型: {df['type'].value_counts().head(10).to_dict()}")
        return df
        
    def load_column_assignments(self) -> pd.DataFrame:
        """加载列分配数据
        
        Returns:
            包含列分配信息的 DataFrame
            列: root_id, hemisphere, type, column_id, x, y, p, q
        """
        logger.info("加载列分配数据...")
        df = pd.read_csv(self.column_file, compression='gzip')
        logger.info(f"  - 加载了 {len(df)} 个神经元的列分配")
        logger.info(f"  - 列数量: {df['column_id'].nunique()}")
        return df
        
    def load_connections(
        self, 
        visual_neuron_ids: Optional[set] = None,
        neuropils: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """加载连接数据
        
        Args:
            visual_neuron_ids: 要过滤的视觉神经元 ID 集合
            neuropils: 要过滤的神经区域列表
            
        Returns:
            包含连接信息的 DataFrame
            列: pre_root_id, post_root_id, neuropil, syn_count, nt_type
        """
        logger.info("加载连接数据...")
        
        # 分块读取大文件
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(self.connections_file, chunksize=chunk_size):
            # 过滤视觉神经元
            if visual_neuron_ids is not None:
                chunk = chunk[
                    chunk['pre_root_id'].isin(visual_neuron_ids) | 
                    chunk['post_root_id'].isin(visual_neuron_ids)
                ]
            
            # 过滤神经区域
            if neuropils is not None:
                chunk = chunk[chunk['neuropil'].isin(neuropils)]
                
            if len(chunk) > 0:
                chunks.append(chunk)
                
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  - 加载了 {len(df)} 个连接")
        logger.info(f"  - 神经递质类型: {df['nt_type'].value_counts().to_dict()}")
        
        return df
        
    def filter_visual_system(
        self,
        subsystems: List[str] = ['Motion', 'Color', 'OFF', 'Photoreceptors']
    ) -> Dict:
        """过滤视觉系统数据
        
        Args:
            subsystems: 要包含的子系统列表
            
        Returns:
            包含过滤后数据的字典
        """
        logger.info("=" * 60)
        logger.info("过滤视觉系统数据")
        logger.info("=" * 60)
        
        # 加载视觉神经元
        visual_neurons = self.load_visual_neurons()
        
        # 过滤子系统
        if subsystems:
            visual_neurons = visual_neurons[
                visual_neurons['subsystem'].isin(subsystems)
            ]
            logger.info(f"过滤子系统后: {len(visual_neurons)} 个神经元")
        
        # 只保留右侧（与原始 Flyvis 一致）
        visual_neurons = visual_neurons[visual_neurons['side'] == 'right']
        logger.info(f"只保留右侧: {len(visual_neurons)} 个神经元")
        
        # 获取神经元 ID
        visual_neuron_ids = set(visual_neurons['root_id'].values)
        
        # 加载列分配
        columns = self.load_column_assignments()
        columns = columns[columns['root_id'].isin(visual_neuron_ids)]
        
        # 加载连接（只在视觉相关的神经区域）
        visual_neuropils = [
            'ME_R', 'LO_R', 'LOP_R',  # 髓质、小叶、小叶板
            'LA_R'  # 层板
        ]
        connections = self.load_connections(
            visual_neuron_ids=visual_neuron_ids,
            neuropils=visual_neuropils
        )
        
        return {
            'neurons': visual_neurons,
            'columns': columns,
            'connections': connections,
            'metadata': {
                'n_neurons': len(visual_neurons),
                'n_cell_types': visual_neurons['type'].nunique(),
                'n_connections': len(connections),
                'subsystems': subsystems
            }
        }
        
    def compute_connectivity_matrix(
        self,
        connections: pd.DataFrame,
        neurons: pd.DataFrame
    ) -> Dict[Tuple[str, str], Dict]:
        """计算细胞类型之间的连接矩阵
        
        Args:
            connections: 连接数据
            neurons: 神经元数据
            
        Returns:
            字典，键为 (源类型, 目标类型)，值为连接统计
        """
        logger.info("计算连接矩阵...")
        
        # 创建 ID 到类型的映射
        id_to_type = dict(zip(neurons['root_id'], neurons['type']))
        
        # 添加类型信息到连接
        connections['pre_type'] = connections['pre_root_id'].map(id_to_type)
        connections['post_type'] = connections['post_root_id'].map(id_to_type)
        
        # 移除未知类型
        connections = connections.dropna(subset=['pre_type', 'post_type'])
        
        # 按类型对聚合
        connectivity = defaultdict(lambda: {
            'syn_count': 0,
            'n_connections': 0,
            'nt_types': defaultdict(int)
        })
        
        for _, row in connections.iterrows():
            key = (row['pre_type'], row['post_type'])
            connectivity[key]['syn_count'] += row['syn_count']
            connectivity[key]['n_connections'] += 1
            connectivity[key]['nt_types'][row['nt_type']] += 1
            
        logger.info(f"  - 找到 {len(connectivity)} 个类型对连接")
        
        return dict(connectivity)
        
    def compute_spatial_offsets(
        self,
        connections: pd.DataFrame,
        columns: pd.DataFrame
    ) -> Dict[Tuple[str, str], List[Tuple[int, int, int]]]:
        """计算空间偏移
        
        Args:
            connections: 连接数据
            columns: 列分配数据
            
        Returns:
            字典，键为 (源类型, 目标类型)，值为 [(du, dv, syn_count), ...]
        """
        logger.info("计算空间偏移...")
        
        # 创建 ID 到位置的映射
        id_to_pos = {}
        for _, row in columns.iterrows():
            id_to_pos[row['root_id']] = {
                'type': row['type'],
                'x': row['x'],
                'y': row['y'],
                'p': row['p'],
                'q': row['q']
            }
        
        # 计算偏移
        offsets = defaultdict(lambda: defaultdict(int))
        
        for _, conn in connections.iterrows():
            pre_id = conn['pre_root_id']
            post_id = conn['post_root_id']
            
            if pre_id in id_to_pos and post_id in id_to_pos:
                pre_pos = id_to_pos[pre_id]
                post_pos = id_to_pos[post_id]
                
                # 使用 p, q 坐标（六边形坐标）
                du = post_pos['p'] - pre_pos['p']
                dv = post_pos['q'] - pre_pos['q']
                
                key = (pre_pos['type'], post_pos['type'])
                offsets[key][(du, dv)] += conn['syn_count']
        
        # 转换为列表格式
        result = {}
        for key, offset_dict in offsets.items():
            result[key] = [
                (du, dv, count) 
                for (du, dv), count in offset_dict.items()
            ]
            
        logger.info(f"  - 计算了 {len(result)} 个类型对的空间偏移")
        
        return result


def main():
    """测试数据加载"""
    loader = FlyWireRealDataLoader()
    
    # 过滤视觉系统数据
    data = loader.filter_visual_system()
    
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"神经元数量: {data['metadata']['n_neurons']}")
    print(f"细胞类型数量: {data['metadata']['n_cell_types']}")
    print(f"连接数量: {data['metadata']['n_connections']}")
    
    print("\n主要细胞类型:")
    print(data['neurons']['type'].value_counts().head(20))
    
    # 计算连接矩阵
    connectivity = loader.compute_connectivity_matrix(
        data['connections'],
        data['neurons']
    )
    
    print(f"\n类型对连接数量: {len(connectivity)}")
    print("\n最强的连接:")
    sorted_conn = sorted(
        connectivity.items(),
        key=lambda x: x[1]['syn_count'],
        reverse=True
    )
    for (pre, post), stats in sorted_conn[:10]:
        print(f"  {pre} -> {post}: {stats['syn_count']} 突触")
    
    # 计算空间偏移
    offsets = loader.compute_spatial_offsets(
        data['connections'],
        data['columns']
    )
    
    print(f"\n有空间偏移信息的类型对: {len(offsets)}")


if __name__ == '__main__':
    main()
