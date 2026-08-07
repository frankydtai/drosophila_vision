# Neuron Simplification with Physics Preservation

将简化后的神经元转换为Jaxley可加载格式，同时保留物理性质。

## 项目结构

```
mcHH/simplification/
├── neuron_simplifier.py          # 核心简化算法（保留物理性质）
├── test_jaxley_integration.py    # Jaxley集成测试
├── README.md                      # 本文件
└── logs/
    ├── algorithm_analysis.md      # 算法分析文档
    └── physics_preservation_math.md  # 数学推导
```

## 快速开始

### 1. 安装依赖

```bash
pip install numpy matplotlib jaxley jax
```

### 2. 基本使用

```python
from neuron_simplifier import NeuronSimplifier

# 创建简化器
simplifier = NeuronSimplifier(angle_threshold_degrees=5.0)

# 加载SWC文件
nodes, root = simplifier.load_swc("input.swc")

# 简化（保留50%的分支）
simplified_nodes, stats = simplifier.simplify(nodes, root, keep_ratio=0.5)

# 保存为Jaxley兼容格式
simplifier.write_swc(simplified_nodes, "output.swc")
simplifier.save_metadata(stats, "input.swc", "output.swc")
```

### 3. 使用Jaxley加载

```python
import jaxley as jx

# 加载简化后的神经元
cell = jx.read_swc(
    "output.swc",
    ncomp=1,  # 每个分支的compartment数量
    assign_groups=True
)

# 设置生物物理参数
cell.set("axial_resistivity", 100.0)  # Ω·cm
cell.set("capacitance", 1.0)  # μF/cm²
```

### 4. 运行完整测试

```bash
python test_jaxley_integration.py /path/to/neuron.swc
```

这将生成：
- 多个简化级别的SWC文件
- 元数据JSON文件
- 对比图表
- 测试报告

## 核心特性

### 1. 物理性质保留

简化算法通过**长度加权平均半径**保留：
- ✓ 总表面积 (Surface Area)
- ✓ 膜电容 (Membrane Capacitance)
- ✓ 膜电导 (Membrane Conductance)

数学公式：
```
r_eq = (r1 * L1 + r2 * L2) / (L1 + L2)
L_eq = L1 + L2
```

详见：`logs/physics_preservation_math.md`

### 2. 两阶段简化

#### 阶段1：共线节点合并
- 移除在直线上的中间节点
- 角度阈值可调（默认5°）
- 自动计算等效半径

#### 阶段2：基于持久性的剪枝
- 保留最重要的分支
- 持久性 = 分支长度的拓扑重要性
- 可控的保留比例（0-1）

### 3. Jaxley兼容性

输出标准SWC格式：
```
# id type x y z radius parent
1 1 0.0 0.0 0.0 2.5 -1
2 3 1.0 0.0 0.0 1.8 1
...
```

Jaxley可直接加载，无需额外转换。

## API文档

### NeuronSimplifier类

#### 初始化
```python
simplifier = NeuronSimplifier(
    angle_threshold_degrees=5.0,  # 共线判断角度阈值
    min_segment_length=1e-6       # 最小段长度
)
```

#### 主要方法

**load_swc(filepath)**
- 加载SWC文件
- 返回：`(nodes, root)`

**simplify(nodes, root, keep_ratio=1.0)**
- 执行完整简化流程
- 返回：`(simplified_nodes, stats)`

**write_swc(nodes, out_path)**
- 保存为SWC格式
- Jaxley兼容

**save_metadata(stats, original_file, output_file)**
- 保存简化统计信息到JSON

**compute_surface_area(nodes)**
- 计算总表面积
- 使用公式：A = 2πrL

### SimplificationStats类

统计信息包括：
- `original_nodes`: 原始节点数
- `simplified_nodes`: 简化后节点数
- `removed_by_collinear`: 共线合并删除数
- `removed_by_pruning`: 剪枝删除数
- `total_surface_area_original`: 原始表面积
- `total_surface_area_simplified`: 简化后表面积
- `surface_area_preservation_ratio`: 表面积保留率

## 算法详解

### 共线节点合并算法

```
对于每个节点 B (父节点 A, 子节点 C):
    1. 检查是否只有一个父节点和一个子节点
    2. 计算向量 v1 = A→B, v2 = B→C
    3. 计算夹角 θ = arccos(v1·v2 / |v1||v2|)
    4. 如果 θ ≥ 180° - angle_threshold:
        a. 计算 L1 = |A→B|, L2 = |B→C|
        b. 计算等效半径: r_eq = (r_A*L1 + r_B*L2) / (L1+L2)
        c. 删除节点 B
        d. 连接 A→C，设置 C 的半径为 r_eq
```

### 持久性剪枝算法

```
1. 计算每个节点到根的距离 g(v)
2. 对每个叶节点 L:
    a. 向上遍历找到分支点 S
    b. 持久性 = g(L) - g(S)
3. 按持久性降序排序所有分支
4. 保留前 keep_ratio 比例的分支
5. 删除其他分支
```

## 性能指标

典型简化效果（基于FlyWire数据）：

| Keep Ratio | 节点减少 | 表面积保留 |
|-----------|---------|-----------|
| 1.0       | ~50%    | ~98%      |
| 0.8       | ~60%    | ~95%      |
| 0.5       | ~75%    | ~90%      |
| 0.2       | ~85%    | ~80%      |
| 0.05      | ~95%    | ~60%      |

注：节点减少包括共线合并（angle=5°）

## 与Jaxley的集成

### Jaxley的SWC加载机制

Jaxley假设：
1. **圆柱形compartment**: A = 2πrL
2. **单点分支**: A = 4πr², L = 2r
3. **自动分组**: 根据SWC type分配到soma/axon/dendrite

### 兼容性保证

我们的简化算法：
- ✓ 保持SWC格式标准
- ✓ 保留type字段（用于分组）
- ✓ 保证parent关系正确
- ✓ 表面积计算与Jaxley一致

### 参数建议

```python
cell = jx.read_swc(
    "simplified.swc",
    ncomp=1,              # 推荐：4-8个compartment
    max_branch_len=None,  # 不额外分割
    min_radius=0.1,       # 根据数据调整
    assign_groups=True    # 启用自动分组
)
```

## 验证方法

### 1. 表面积验证

```python
original_area = simplifier.compute_surface_area(original_nodes)
simplified_area = simplifier.compute_surface_area(simplified_nodes)
preservation = simplified_area / original_area
print(f"Surface area preservation: {preservation*100:.2f}%")
```

### 2. Jaxley加载验证

```python
try:
    cell = jx.read_swc("simplified.swc", ncomp=4)
    print("✓ Jaxley loading successful")
except Exception as e:
    print(f"✗ Jaxley loading failed: {e}")
```

### 3. 电生理模拟验证

```python
# 设置参数
cell.set("axial_resistivity", 100.0)
cell.set("capacitance", 1.0)

# 运行简单模拟
# (需要完整的Jaxley模拟代码)
```

## 常见问题

### Q1: 为什么表面积保留率不是100%？

A: 剪枝阶段会删除整个分支，这些分支的表面积无法保留。只有共线合并阶段能完美保留表面积。

### Q2: 如何选择angle_threshold？

A: 
- 5°（默认）：严格，只合并几乎完全共线的节点
- 10-20°：中等，适合大多数情况
- 40°+：宽松，可能改变形态

### Q3: 如何选择keep_ratio？

A:
- 1.0：只做共线合并，不剪枝
- 0.5-0.8：保留主要结构
- 0.2-0.5：大幅简化
- <0.2：极简化，可能丢失重要特征

### Q4: 简化会影响电生理模拟吗？

A: 
- 共线合并：影响很小（表面积保留）
- 剪枝：会影响，因为删除了分支
- 建议：先用高keep_ratio测试

## 未来改进

- [ ] 支持多种物理量保留策略（轴向电阻、混合）
- [ ] 自适应angle_threshold（根据局部曲率）
- [ ] 基于电生理重要性的剪枝
- [ ] GPU加速（大规模神经元）
- [ ] 可视化工具（3D交互）

## 参考文献

1. [Jaxley Documentation](https://jaxley.readthedocs.io/)
2. [SWC Format Specification](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
3. Rall, W. (1959). Branching dendritic trees and motoneuron membrane resistivity. *Experimental Neurology*.

## 许可证

MIT License

## 联系方式

如有问题或建议，请查看 `logs/` 目录中的详细文档。
