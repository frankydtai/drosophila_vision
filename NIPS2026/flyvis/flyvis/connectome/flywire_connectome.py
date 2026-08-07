"""
FlyWire 连接组类

实现从 FlyWire 数据构建连接组的功能，与 Flyvis 框架兼容。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datamate import Directory

from flyvis.connectome.connectome import (
    Node,
    Edge,
    register_connectome,
    add_strided_nodes,
    add_tiled_nodes,
    add_single_node
)
from flyvis.utils import nodes_edges_utils
import flyvis

logger = logging.getLogger(__name__)

__all__ = ["ConnectomeFromFlyWire"]


@register_connectome
class ConnectomeFromFlyWire(Directory):
    """从 FlyWire 数据构建连接组
    
    与 ConnectomeFromAvgFilters 保持相同的接口，但从 FlyWire 数据源构建。
    
    Args:
        flywire_data_path: FlyWire 数据文件路径（JSON 格式）
        extent: 六边形网格半径（列数）
        n_syn_fill: 数据缺口中假设的突触数量
        cell_type_mapping: 细胞类型名称映射字典（可选）
        
    Attributes:
        unique_cell_types: 识别的细胞类型
        input_cell_types: 输入细胞类型
        intermediate_cell_types: 隐藏细胞类型
        output_cell_types: 解码细胞类型
        central_cells_index: 每个细胞类型的中心细胞索引
        layout: 可视化的输入、隐藏、输出定义
        nodes: 节点表
        edges: 边表
    """
    
    def __init__(
        self,
        flywire_data_path: str,
        extent: int = 15,
        n_syn_fill: int = 1,
        cell_type_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """初始化 FlyWire 连接组
        
        Args:
            flywire_data_path: FlyWire 数据文件路径
            extent: 六边形网格半径
            n_syn_fill: 数据缺口填充的突触数量
            cell_type_mapping: 细胞类型名称映射
        """
        # 加载 FlyWire 数据
        flywire_path = Path(flywire_data_path)
        if not flywire_path.exists():
            raise FileNotFoundError(f"FlyWire 数据文件不存在: {flywire_path}")
            
        logger.info(f"从 FlyWire 数据构建连接组: {flywire_path}")
        
        with open(flywire_path, 'r') as f:
            spec = json.load(f)
            
        # 应用细胞类型映射（如果提供）
        if cell_type_mapping:
            spec = self._apply_cell_type_mapping(spec, cell_type_mapping)
            
        # 存储唯一细胞类型和布局变量
        self.unique_cell_types = np.bytes_([n["name"] for n in spec["nodes"]])
        self.input_cell_types = np.bytes_(spec["input_units"])
        self.output_cell_types = np.bytes_(spec["output_units"])
        
        # 确定中间细胞类型
        intermediate_cell_types, _ = nodes_edges_utils.order_node_type_list(
            np.array(
                list(
                    set(self.unique_cell_types)
                    - set(self.input_cell_types)
                    - set(self.output_cell_types)
                )
            ).astype(str)
        )
        self.intermediate_cell_types = np.array(intermediate_cell_types).astype("S")
        
        # 构建布局
        layout = []
        layout.extend(
            list(
                zip(
                    self.input_cell_types,
                    [b"retina" for _ in range(len(self.input_cell_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.intermediate_cell_types,
                    [b"intermediate" for _ in range(len(self.intermediate_cell_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.output_cell_types,
                    [b"output" for _ in range(len(self.output_cell_types))],
                )
            )
        )
        self.layout = np.bytes_(layout)
        
        # 构建节点和边
        nodes: List[Node] = []
        edges: List[Edge] = []
        self._add_nodes(nodes, spec["nodes"], extent)
        self._add_edges(edges, nodes, spec["edges"], n_syn_fill)
        
        # 定义节点角色
        _role = {node: "intermediate" for node in set([n.type for n in nodes])}
        _role.update({node: "input" for node in _role if node in spec["input_units"]})
        _role.update({node: "output" for node in _role if node in spec["output_units"]})
        
        # 存储图
        self.nodes = dict(
            index=np.int64([n.id for n in nodes]),
            type=np.bytes_([n.type for n in nodes]),
            u=np.int32([n.u for n in nodes]),
            v=np.int32([n.v for n in nodes]),
            role=np.bytes_([_role[n.type] for n in nodes]),
        )
        
        self.edges = dict(
            # 必需字段
            source_index=np.int64([e.source.id for e in edges]),
            target_index=np.int64([e.target.id for e in edges]),
            sign=np.float32([e.sign for e in edges]),
            n_syn=np.float32([e.n_syn for e in edges]),
            # 便利字段
            source_type=np.bytes_([e.source.type for e in edges]),
            target_type=np.bytes_([e.target.type for e in edges]),
            source_u=np.int32([e.source.u for e in edges]),
            target_u=np.int32([e.target.u for e in edges]),
            source_v=np.int32([e.source.v for e in edges]),
            target_v=np.int32([e.target.v for e in edges]),
            du=np.int32([e.target.u - e.source.u for e in edges]),
            dv=np.int32([e.target.v - e.source.v for e in edges]),
            n_syn_certainty=np.float32([e.n_syn_certainty for e in edges]),
        )
        
        # 存储中心索引
        self.central_cells_index = np.int64(
            np.nonzero((self.nodes['u'] == 0) & (self.nodes['v'] == 0))[0]
        )
        
        # 存储层索引
        layer_index = {}
        for cell_type in self.unique_cell_types:
            node_indices = np.nonzero(self.nodes["type"] == cell_type)[0]
            layer_index[cell_type.decode()] = np.int64(node_indices)
        self.nodes['layer_index'] = layer_index
        
        logger.info(
            f"FlyWire 连接组构建完成: "
            f"{len(nodes)} 个神经元, "
            f"{len(edges)} 个连接, "
            f"{len(self.unique_cell_types)} 种细胞类型"
        )
        
    def _apply_cell_type_mapping(
        self, 
        spec: Dict, 
        mapping: Dict[str, str]
    ) -> Dict:
        """应用细胞类型名称映射
        
        Args:
            spec: 连接组规范
            mapping: 名称映射字典
            
        Returns:
            更新后的规范
        """
        # 映射节点名称
        for node in spec["nodes"]:
            if node["name"] in mapping:
                node["name"] = mapping[node["name"]]
                
        # 映射边的源和目标
        for edge in spec["edges"]:
            if edge["src"] in mapping:
                edge["src"] = mapping[edge["src"]]
            if edge["tar"] in mapping:
                edge["tar"] = mapping[edge["tar"]]
                
        # 映射输入和输出单元
        spec["input_units"] = [
            mapping.get(u, u) for u in spec["input_units"]
        ]
        spec["output_units"] = [
            mapping.get(u, u) for u in spec["output_units"]
        ]
        
        return spec
        
    def _add_nodes(
        self, 
        seq: List[Node], 
        node_spec: List[Dict], 
        extent: int
    ) -> None:
        """添加节点到序列
        
        Args:
            seq: 要添加节点的列表
            node_spec: 节点规范字典
            extent: 数组半径（列数）
        """
        for n in node_spec:
            typ, (pattern, args) = n["name"], n["pattern"]
            if pattern == "stride":
                add_strided_nodes(seq, typ, extent, args)
            elif pattern == "tile":
                add_tiled_nodes(seq, typ, extent, args)
            elif pattern == "single":
                add_single_node(seq, typ, extent)
                
    def _add_edges(
        self,
        seq: List[Edge],
        nodes: List[Node],
        edge_spec: List[Dict],
        n_syn_fill: float
    ) -> None:
        """添加边到序列
        
        Args:
            seq: 要添加边的列表
            nodes: 所有节点列表
            edge_spec: 边规范字典
            n_syn_fill: 数据缺口填充的突触数量
        """
        from toolz import groupby
        from contextlib import suppress
        
        node_index = {
            **groupby(lambda n: n.type, nodes),
            **groupby(lambda n: (n.type, n.u, n.v), nodes),
        }
        
        for e in edge_spec:
            offsets = e["offsets"]
            
            # 如果需要填充凸包
            if n_syn_fill > 0 and len(offsets) >= 3:
                offsets = self._fill_hull(offsets, n_syn_fill)
                
            # 添加卷积边
            for (du, dv), n_syn in offsets:
                for src in node_index.get(e["src"], []):
                    u_tgt = src.u + du
                    v_tgt = src.v + dv
                    with suppress(KeyError):
                        tgt = node_index[e["tar"], u_tgt, v_tgt][0]
                        seq.append(
                            Edge(
                                len(seq),
                                src,
                                tgt,
                                e["alpha"],
                                n_syn,
                                e.get("lambda_mult", 1.0)
                            )
                        )
                        
    def _fill_hull(
        self, 
        offsets: List[List], 
        n_syn_fill: float
    ) -> List[List]:
        """填充报告边的凸包
        
        Args:
            offsets: 边偏移列表
            n_syn_fill: 数据缺口填充的突触数量
            
        Returns:
            填充后的偏移列表
        """
        import scipy.spatial as ss
        import matplotlib.path as mp
        
        # 收集报告为边的点（列偏移，(du, dv)）
        known_pts = np.array([offset[0] for offset in offsets])
        known_pts_as_set = set(map(tuple, known_pts))
        
        # 计算报告边的凸包
        hull = ss.ConvexHull((1 + 1e-6) * known_pts, False, "QJ")
        hull_vertices = known_pts[hull.vertices]
        
        # 找到凸包内的点
        grid = np.concatenate(
            np.dstack(
                np.mgrid[
                    known_pts[:, 0].min() : known_pts[:, 0].max() + 1,
                    known_pts[:, 1].min() : known_pts[:, 1].max() + 1,
                ]
            )
        )
        contained_pts = grid[mp.Path(hull_vertices).contains_points(grid)]
        
        # 将凸包内的未知点添加为偏移
        return offsets + [
            [[u, v], n_syn_fill] 
            for u, v in contained_pts 
            if (u, v) not in known_pts_as_set
        ]
        
    def get_statistics(self) -> Dict:
        """获取连接组统计信息
        
        Returns:
            包含统计信息的字典
        """
        return {
            'n_neurons': len(self.nodes['index']),
            'n_synapses': len(self.edges['source_index']),
            'n_cell_types': len(self.unique_cell_types),
            'n_input_types': len(self.input_cell_types),
            'n_output_types': len(self.output_cell_types),
            'n_intermediate_types': len(self.intermediate_cell_types),
            'excitatory_synapses': np.sum(self.edges['sign'] > 0),
            'inhibitory_synapses': np.sum(self.edges['sign'] < 0),
            'avg_synapses_per_connection': np.mean(self.edges['n_syn']),
        }
