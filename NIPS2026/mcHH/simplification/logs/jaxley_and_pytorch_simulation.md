# Jaxley加载验证 & PyTorch多compartment HH模拟

**日期**: 2026-04-08

---

## 1. 简化后SWC能否被Jaxley正确加载？

### 结论：需要两个修复后可以

#### 发现的两个关键Bug

**Bug 1: `graph.nodes[1]` KeyError**

Jaxley的 `_add_missing_graph_attrs` (graph.py:149) 硬编码访问 `graph.nodes[1]`。
如果简化后的SWC不包含ID=1的节点，会抛出 `KeyError`。

**Bug 2: 多棵树（forest）结构**

简化器的 `prune_tree` 可能将某些节点的parent设为-1（因为原parent被删除），
导致SWC变成多棵树。Jaxley假设SWC是单棵树。

#### 修复方案（已实现）

在 `neuron_simplifier.py` 的 `write_swc` 方法中：
1. 连接所有额外的根节点到主根 → 保证单棵树
2. 重新编号ID从1开始连续 → 保证 `graph.nodes[1]` 存在

```python
# 修复后的write_swc关键逻辑:
roots = [nid for nid, n in nodes.items() if n['parent'] == -1]
if len(roots) > 1:
    main_root = roots[0]
    for extra_root in roots[1:]:
        nodes[extra_root]['parent'] = main_root

old_ids = sorted(nodes.keys())
id_map = {old: new for new, old in enumerate(old_ids, start=1)}
```

#### 其他验证结果

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `#` 注释行 | ✅ 安全 | `np.loadtxt` 自动跳过 |
| 不连续ID | ✅ 安全 | NetworkX支持任意整数键，但已修复为连续 |
| 表面积计算差异 | ⚠️ 微小 | Jaxley用截锥体(truncated cone)，我们用圆柱体 |
| 单点分支 | ✅ 安全 | Jaxley有球体近似处理逻辑 |

---

## 2. Jaxley的完整加载流程

```
read_swc(fname, ncomp)
  └→ to_swc_graph(fname)
      └→ np.loadtxt(fname)           # 7列数值，跳过#注释
      └→ nx.DiGraph 构建             # id, x, y, z, r, p 属性
      └→ _add_edge_lengths()          # L = sqrt(Σ(dx²)), 最小1e-5
  └→ build_compartment_graph(swc_graph, ncomp)
      └→ _trace_branches()            # DFS遍历，识别分支和分支点
      └→ 对每个branch:
          └→ split_xyzr_into_equal_length_segments()  # 等距分割
          └→ morph_attrs_from_xyzr()   # 计算radius, area, volume, r_load
      └→ 连接compartment和branchpoint
  └→ from_graph(comp_graph)            # 转换为 jx.Cell
```

### Jaxley的表面积计算

**截锥体近似** (非圆柱体):
```
A = π(r₁ + r₂) · slant_length
slant_length = sqrt(Δr² + L²)
```

### Jaxley的resistive load计算

```
r_load = (1/π) ∫ 1/r² dl

对恒定半径: r_load = L / (πr²)
对变半径(截锥): r_load = L/(Δr) · (1/r₁ - 1/r₂) / π
```

---

## 3. PyTorch多compartment HH模拟器

### 3.1 核心方程

分支电缆方程（离散化）:

```
C_m · dV_i/dt = -Σ_k g_k(V_i - E_k) + (1/A_i)·Σ_j G_ij(V_j - V_i) + I_ext,i
```

其中轴向电导:
```
G_ij = 1 / (R_a · (r_load_out_i + r_load_in_j))
```

### 3.2 半隐式求解器

**关键发现**: 显式Euler在细小compartment上不稳定，因为轴向耦合系数
可达 ~25000 /ms，要求 dt < 2×10⁻⁵ ms，完全不实用。

**解决方案**: 半隐式（Strang分裂）方案，匹配Jaxley的默认策略:
- 通道/门控动力学: 显式 Euler
- 轴向耦合: 隐式 backward Euler

矩阵方程:
```
(I + dt·diag(voltage_terms) - dt·G) · V^{n+1} = V^n + dt·constant_terms
```

其中:
- `voltage_terms = Σg_k / C_m` (膜电导)
- `constant_terms = (Σg_k·E_k + I_ext) / C_m`
- `G` = 轴向耦合矩阵

使用 `torch.linalg.solve` 求解，完全可微分。

### 3.3 单位约定（与Jaxley一致）

| 量 | 单位 | 说明 |
|----|------|------|
| V | mV | 膜电位 |
| t | ms | 时间 |
| C_m | μF/cm² | 比膜电容 |
| g | mS/cm² | 通道电导密度 |
| I | μA/cm² | 电流密度 |
| R_a | Ω·cm | 轴向电阻率 |
| r, L | μm | 半径、长度 |
| A | μm² | 表面积 |
| r_load | μm⁻¹ | 电阻负载 |

1e7 转换因子: 将 `G/(R_a·r_load·A·C_m)` 从物理单位转为 `mV/ms`。

### 3.4 测试结果

**单compartment HH** (验证正确性):
```
V range: [-75.1, 40.0] mV  ← 标准HH动作电位
10 μA/cm² → 正常放电
```

**多compartment分支神经元** (12 compartments, 3 branches):
```
V range: [-75.5, 39.3] mV  ← 动作电位正常传播
50 μA/cm² into soma → 动作电位从soma传播到dendrites
```

**可微分性验证**:
```
所有8个参数的梯度成功计算:
  d(loss)/d(log_g_Na)  = 11.41
  d(loss)/d(log_g_K)   = -14.25
  d(loss)/d(log_g_L)   = -0.028
  d(loss)/d(E_Na)      = 0.189
  d(loss)/d(E_K)       = 0.540
  d(loss)/d(E_L)       = 0.014
  d(loss)/d(log_C_m)   = 1.580
  d(loss)/d(log_R_a)   = 0.146
```

---

## 4. 与现有代码的关系

### 4.1 现有 dynamics.py 的 HHModel

```python
class HHModel(BaseDynamics):
    """单compartment HH模型 - 每个"node"是独立的HH神经元"""
    # 状态: v, m, h, n  (batch, n_nodes)
    # 突触: weight * v_at_pre → target_sum → 某节点
    # 无轴向耦合！
```

### 4.2 新的 mc_hh_torch.py 的 MultiCompartmentHH

```python
class MultiCompartmentHH(nn.Module):
    """多compartment HH模型 - 一个"神经元"包含多个compartment"""
    # 状态: V, m, h, n  (batch, n_comp)
    # 轴向耦合: 隐式Euler求解 (I - dt·G) V = rhs
    # 从SWC加载形态学
    # 所有参数可微分
```

### 4.3 集成路径

可以将 `MultiCompartmentHH` 作为一个"超级节点"嵌入 `ConnectomeNetwork`:
- 每个 cell type 是一个 `MultiCompartmentHH`
- 突触连接到特定compartment
- 网络级别用 `ConnectomeNetwork` 管理

---

## 5. 文件清单

```
mcHH/simplification/
├── neuron_simplifier.py              # 简化算法（修复了Jaxley兼容性）
├── mc_hh_torch.py                    # PyTorch多compartment HH (新)
├── test_jaxley_integration.py        # Jaxley集成测试
├── batch_simplify.py                 # 批量处理
├── README.md                          # 文档
└── logs/
    ├── algorithm_analysis.md          # 原始算法分析
    ├── physics_preservation_math.md   # 物理量保留推导
    ├── comparison_with_original.md    # 与原版对比
    ├── jaxley_loading_analysis.md     # Jaxley加载分析+公式推导
    └── jaxley_and_pytorch_simulation.md  # 本文档
```

---

## 6. 关键发现总结

1. **简化后SWC可以被Jaxley加载**，但需要重编号ID和确保单棵树结构
2. **显式Euler不适用于多compartment模型** — 轴向耦合系数太大导致数值不稳定
3. **半隐式Euler + `torch.linalg.solve`** 是正确的PyTorch实现方案
4. **所有HH参数和形态学参数都可微分** — 支持端到端训练
5. **单位转换因子 1e7** 是连接SWC几何（μm）和电生理（mV/ms）的关键
