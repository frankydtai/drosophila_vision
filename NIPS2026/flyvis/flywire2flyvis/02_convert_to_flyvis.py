"""
FlyWire 到 Flyvis 格式转换器

将真实的 FlyWire 数据转换为 Flyvis 模型所需的 JSON 格式。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from load_flywire_data_01 import FlyWireRealDataLoader
except ImportError:
    # 兼容旧的导入方式
    from flywire_real_data_loader import FlyWireRealDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlyWireToFlyvisRealConverter:
    """将真实 FlyWire 数据转换为 Flyvis 格式"""
    
    # 神经递质到突触符号的映射
    NT_TO_SIGN = {
        'ACH': 1,      # 乙酰胆碱 - 兴奋性
        'GLUT': 1,     # 谷氨酸 - 兴奋性
        'GABA': -1,    # GABA - 抑制性
        'SER': 1,      # 血清素 - 通常兴奋性
        'DA': 1,       # 多巴胺 - 通常兴奋性
        'OCT': 1,      # 章鱼胺 - 通常兴奋性
    }
    
    # 输入细胞类型（光感受器）
    INPUT_TYPES = ['R1-6', 'R7', 'R8']
    
    # 输出细胞类型（运动检测神经元）
    OUTPUT_TYPES = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
    
    def __init__(self, extent: int = 15):
        """初始化转换器
        
        Args:
            extent: 六边形网格半径
        """
        self.extent = extent
        self.loader = FlyWireRealDataLoader()
        
    def convert(
        self,
        output_path: str = "flyvis/connectome/flywire_v1.0.json",
        subsystems: List[str] = ['Motion', 'Color', 'OFF', 'Photoreceptors'],
        min_syn_count: int = 5
    ) -> Dict:
        """转换 FlyWire 数据为 Flyvis 格式
        
        Args:
            output_path: 输出 JSON 文件路径
            subsystems: 要包含的子系统
            min_syn_count: 最小突触数量阈值
            
        Returns:
            Flyvis 格式的连接组数据
        """
        logger.info("=" * 60)
        logger.info("开始转换 FlyWire 数据为 Flyvis 格式")
        logger.info("=" * 60)
        
        # 1. 加载和过滤数据
        data = self.loader.filter_visual_system(subsystems=subsystems)
        
        # 2. 计算连接统计
        connectivity = self.loader.compute_connectivity_matrix(
            data['connections'],
            data['neurons']
        )
        
        # 3. 计算空间偏移
        offsets = self.loader.compute_spatial_offsets(
            data['connections'],
            data['columns']
        )
        
        # 4. 转换节点
        nodes = self._convert_nodes(data['neurons'])
        
        # 5. 转换边
        edges = self._convert_edges(
            connectivity,
            offsets,
            min_syn_count=min_syn_count
        )
        
        # 6. 确定输入和输出单元
        input_units = self._identify_input_units(nodes)
        output_units = self._identify_output_units(nodes)
        
        # 7. 构建最终数据结构
        flyvis_data = {
            'nodes': nodes,
            'edges': edges,
            'input_units': input_units,
            'output_units': output_units,
            'metadata': {
                'source': 'FlyWire v783',
                'date': '2025-06-23',
                'extent': self.extent,
                'subsystems': subsystems,
                'min_syn_count': min_syn_count,
                'n_cell_types': len(nodes),
                'n_connections': len(edges),
                'original_neurons': data['metadata']['n_neurons'],
                'original_connections': data['metadata']['n_connections']
            }
        }
        
        # 8. 保存
        if output_path:
            self._save_json(flyvis_data, output_path)
            
        logger.info("=" * 60)
        logger.info("转换完成！")
        logger.info("=" * 60)
        
        return flyvis_data
        
    def _convert_nodes(self, neurons: 'pd.DataFrame') -> List[Dict]:
        """转换神经元节点
        
        Args:
            neurons: 神经元数据
            
        Returns:
            Flyvis 格式的节点列表
        """
        logger.info("转换节点...")
        
        # 获取唯一的细胞类型
        cell_types = sorted(neurons['type'].unique())
        
        nodes = []
        for cell_type in cell_types:
            # 所有柱状细胞类型使用 stride [1, 1]
            # 即每个视觉列都有一个该类型的神经元
            node = {
                "name": cell_type,
                "pattern": ["stride", [1, 1]],
                "activation": "relu",
                "bias": 0.5,  # 默认值
                "bias_fixed": False,
                "time_constant": None,  # 将被优化
                "time_constant_fixed": False
            }
            nodes.append(node)
            
        logger.info(f"  - 转换了 {len(nodes)} 种细胞类型")
        
        return nodes
        
    def _convert_edges(
        self,
        connectivity: Dict[Tuple[str, str], Dict],
        offsets: Dict[Tuple[str, str], List[Tuple[int, int, int]]],
        min_syn_count: int = 5
    ) -> List[Dict]:
        """转换突触连接
        
        Args:
            connectivity: 连接统计
            offsets: 空间偏移
            min_syn_count: 最小突触数量
            
        Returns:
            Flyvis 格式的边列表
        """
        logger.info("转换边...")
        
        edges = []
        
        for (src_type, tgt_type), stats in connectivity.items():
            # 过滤弱连接
            if stats['syn_count'] < min_syn_count:
                continue
                
            # 确定突触符号（基于主要神经递质）
            nt_types = stats['nt_types']
            main_nt = max(nt_types.items(), key=lambda x: x[1])[0]
            sign = self.NT_TO_SIGN.get(main_nt, 1)
            
            # 获取空间偏移
            key = (src_type, tgt_type)
            if key in offsets:
                offset_list = [
                    [[int(du), int(dv)], int(count)]
                    for du, dv, count in offsets[key]
                ]
            else:
                # 如果没有空间信息，假设是中心连接
                offset_list = [[[0, 0], stats['syn_count']]]
                
            edge = {
                "src": src_type,
                "tar": tgt_type,
                "alpha": sign,
                "offsets": offset_list,
                "lambda_mult": 1.0  # 突触确定性
            }
            
            edges.append(edge)
            
        logger.info(f"  - 转换了 {len(edges)} 个连接")
        
        return edges
        
    def _identify_input_units(self, nodes: List[Dict]) -> List[str]:
        """识别输入细胞类型
        
        Args:
            nodes: 节点列表
            
        Returns:
            输入细胞类型名称列表
        """
        input_units = []
        for node in nodes:
            if node['name'] in self.INPUT_TYPES:
                input_units.append(node['name'])
                
        logger.info(f"  - 识别了 {len(input_units)} 种输入细胞类型: {input_units}")
        
        return input_units
        
    def _identify_output_units(self, nodes: List[Dict]) -> List[str]:
        """识别输出细胞类型
        
        Args:
            nodes: 节点列表
            
        Returns:
            输出细胞类型名称列表
        """
        output_units = []
        for node in nodes:
            if node['name'] in self.OUTPUT_TYPES:
                output_units.append(node['name'])
                
        logger.info(f"  - 识别了 {len(output_units)} 种输出细胞类型: {output_units}")
        
        return output_units
        
    def _save_json(self, data: Dict, filepath: str):
        """保存为 JSON 文件
        
        Args:
            data: 要保存的数据
            filepath: 输出文件路径
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"已保存到: {output_path}")
        logger.info(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """主函数"""
    converter = FlyWireToFlyvisRealConverter(extent=15)
    
    # 转换数据
    flyvis_data = converter.convert(
        output_path="flyvis/connectome/flywire_v1.0.json",
        subsystems=['Motion', 'Color', 'OFF', 'Photoreceptors'],
        min_syn_count=10  # 只保留较强的连接
    )
    
    # 打印统计
    print("\n" + "=" * 60)
    print("Flyvis 格式数据统计")
    print("=" * 60)
    print(f"细胞类型数量: {len(flyvis_data['nodes'])}")
    print(f"连接数量: {len(flyvis_data['edges'])}")
    print(f"输入类型: {flyvis_data['input_units']}")
    print(f"输出类型: {flyvis_data['output_units']}")
    
    print("\n主要细胞类型:")
    for node in flyvis_data['nodes'][:20]:
        print(f"  - {node['name']}")
        
    print("\n最强的连接:")
    # 按突触总数排序
    edges_with_count = []
    for edge in flyvis_data['edges']:
        total_syn = sum(count for _, count in edge['offsets'])
        edges_with_count.append((edge, total_syn))
        
    edges_with_count.sort(key=lambda x: x[1], reverse=True)
    
    for edge, count in edges_with_count[:10]:
        sign = "兴奋性" if edge['alpha'] > 0 else "抑制性"
        print(f"  {edge['src']} -> {edge['tar']}: {count} 突触 ({sign})")
        
    print("\n元数据:")
    for key, value in flyvis_data['metadata'].items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
