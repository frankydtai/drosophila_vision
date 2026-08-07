# 神经元简化算法分析与Jaxley转换方案

## 日期
2026-04-08

## 1. 原始简化算法分析

### 1.1 算法流程

原始算法包含两个主要步骤：

#### 步骤1: 共线节点合并 (simplify_collinear_nodes)
- **目标**: 移除在直线上的中间节点
- **条件**: 
  - 节点只有一个父节点和一个子节点
  - 转折角接近180° (由 `angle_threshold_degrees` 控制)
- **方法**: 
  - 计算入射向量 v1 = parent - node
  - 计算出射向量 v2 = child - node
  - 使用余弦相似度判断: cos(θ) ≤ -cos(angle_threshold)
  - 如果满足条件，删除中间节点，将child直接连接到parent

#### 步骤2: 基于持久性的分支剪枝 (prune_tree)
- **持久性定义**: persistence = g(leaf) - g(s)
  - g(v): 节点v到根节点的累积距离
  - s: 分支点（该叶节点路径上第一个有其他更长分支的节点）
- **剪枝策略**: 保留持久性最高的 keep_ratio 比例的分支

### 1.2 问题：物理性质未保留

**关键问题**: 原算法只保留了几何形态，但删除节点时**没有保留电生理特性**：
- 表面积 (Surface Area)
- 电导 (Conductance) 
- 电容 (Capacitance)
- 轴向电阻 (Axial Resistance)

## 2. Jaxley的SWC加载机制

根据 [Jaxley文档](https://jaxley.readthedocs.io/en/latest/reference/_autosummary/jaxley.read_swc.html):

### 2.1 Jaxley的假设
- **圆柱形compartment**: 每个compartment由长度(length)和半径(radius)定义
- **表面积计算**: 
  - 正常分支: `A = 2πr·length`
  - 单点分支: `A = 4πr²`, 此时设置 `length = 2r`

### 2.2 关键参数
- `ncomp`: 每个分支的compartment数量
- `max_branch_len`: 超过此长度的分支会被分割
- `min_radius`: 最小半径阈值
- `assign_groups`: 是否根据SWC type分配组(soma, axon, basal, apical)

### 2.3 SWC格式
```
id type x y z radius parent
```

## 3. 物理性质保留的数学推导

### 3.1 需要保留的物理量

对于神经元的电生理模拟，关键物理量包括：

#### (1) 膜表面积 (Membrane Surface Area)
```
A_membrane = 2πr·L
```
其中 r 是半径，L 是长度

#### (2) 膜电容 (Membrane Capacitance)
```
C_m = c_m · A_membrane = c_m · 2πr·L
```
其中 c_m 是比电容 (通常 ~1 μF/cm²)

#### (3) 轴向电导 (Axial Conductance)
```
G_axial = (πr²) / (R_a · L)
```
其中 R_a 是轴向电阻率 (通常 ~100 Ω·cm)

#### (4) 膜电导 (Membrane Conductance)
```
G_m = g_m · A_membrane = g_m · 2πr·L
```
其中 g_m 是比电导

### 3.2 合并共线节点时的物理量保留

当删除中间节点 B，将 A-B-C 合并为 A-C 时：

**原始配置**:
- 段1 (A→B): 长度 L₁, 半径 r₁
- 段2 (B→C): 长度 L₂, 半径 r₂

**目标**: 找到等效的单段 (A→C): 长度 L_eq, 半径 r_eq

#### 方案1: 保留总表面积 + 平均半径
```
L_eq = L₁ + L₂
r_eq = (r₁·L₁ + r₂·L₂) / (L₁ + L₂)  # 长度加权平均

验证表面积:
A_total = 2πr₁L₁ + 2πr₂L₂
A_eq = 2πr_eq·L_eq = 2π·[(r₁L₁ + r₂L₂)/(L₁+L₂)]·(L₁+L₂) = 2π(r₁L₁ + r₂L₂) ✓
```

#### 方案2: 保留轴向电阻
```
轴向电阻: R_axial = R_a · L / (πr²)

总轴向电阻 (串联):
R_total = R_a·L₁/(πr₁²) + R_a·L₂/(πr₂²)

等效电阻:
R_eq = R_a·L_eq/(πr_eq²)

求解 r_eq:
L_eq = L₁ + L₂
r_eq = sqrt[(L₁ + L₂) / (L₁/r₁² + L₂/r₂²)]
```

#### 方案3: 保留电容 (推荐)
```
总电容 (并联):
C_total = c_m·2π(r₁L₁ + r₂L₂)

等效电容:
C_eq = c_m·2πr_eq·L_eq

因此:
r_eq = (r₁L₁ + r₂L₂) / (L₁ + L₂)  # 与方案1相同
L_eq = L₁ + L₂
```

**结论**: 方案1和方案3等价，这是最简单且物理上合理的方法。

### 3.3 剪枝分支时的处理

当删除整个分支时，无法简单地"保留"其物理性质到其他部分。但可以：

1. **记录删除的分支信息** (用于后续分析)
2. **调整父节点半径** (可选，保守策略)
   - 如果删除的分支很小，可以略微增加父节点半径以补偿表面积损失
   - 但这可能改变形态，需谨慎使用

## 4. 转换为Jaxley兼容格式的策略

### 4.1 修改后的简化算法

```python
def simplify_with_physics_preservation(nodes, angle_threshold_degrees=5.0):
    """
    改进的共线节点合并，保留物理性质
    """
    while True:
        removed_any = False
        children = build_children_dict(nodes)
        
        for nid in list(nodes.keys()):
            if should_merge_node(nid, nodes, children, angle_threshold_degrees):
                # 获取父节点、当前节点、子节点
                pid = nodes[nid]['parent']
                cid = children[nid][0]
                
                # 计算物理量保留的新半径
                L1 = distance(nodes[pid], nodes[nid])
                L2 = distance(nodes[nid], nodes[cid])
                r1 = nodes[pid]['r']
                r2 = nodes[nid]['r']
                
                # 长度加权平均半径（保留表面积和电容）
                r_new = (r1 * L1 + r2 * L2) / (L1 + L2)
                
                # 更新子节点的半径
                nodes[cid]['r'] = r_new
                nodes[cid]['parent'] = pid
                
                # 删除中间节点
                del nodes[nid]
                removed_any = True
        
        if not removed_any:
            break
    
    return nodes
```

### 4.2 输出格式

输出标准SWC格式，Jaxley可直接加载：
```
# id type x y z radius parent
1 1 0.0 0.0 0.0 2.5 -1
2 3 1.0 0.0 0.0 1.8 1
3 3 2.0 0.0 0.0 1.5 2
...
```

### 4.3 元数据记录

创建配套的JSON文件记录简化信息：
```json
{
  "original_file": "neuron_001.swc",
  "simplification_params": {
    "angle_threshold": 5.0,
    "keep_ratio": 0.5
  },
  "statistics": {
    "original_nodes": 10000,
    "simplified_nodes": 2500,
    "removed_by_collinear": 5000,
    "removed_by_pruning": 2500,
    "total_surface_area_original": 15000.5,
    "total_surface_area_simplified": 14800.2,
    "surface_area_preservation_ratio": 0.987
  },
  "removed_branches": [
    {
      "leaf_id": 5432,
      "persistence": 12.5,
      "surface_area": 45.2
    }
  ]
}
```

## 5. 实现计划

1. ✅ 分析原始算法
2. ✅ 研究Jaxley加载机制
3. ✅ 推导物理量保留公式
4. ⏳ 实现改进的简化算法
5. ⏳ 添加物理量验证
6. ⏳ 测试Jaxley加载
7. ⏳ 性能对比分析

## 6. 参考资料

- [Jaxley read_swc文档](https://jaxley.readthedocs.io/en/latest/reference/_autosummary/jaxley.read_swc.html)
- 原始简化代码: `/Users/lengyuner/Desktop/ConVis/simplification/neuron_mst_visualization.ipynb`
- SWC格式规范: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
