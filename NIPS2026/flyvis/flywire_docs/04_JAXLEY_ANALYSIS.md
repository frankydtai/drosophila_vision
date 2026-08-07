# FlyWire + Flyvis 集成项目总结

## 项目完成情况

### ✅ 已完成

1. **环境设置**
   - 创建独立的 conda 环境 `flywire_flyvis` (Python 3.10)
   - 避免影响现有环境

2. **真实数据访问**
   - 定位到实际的 FlyWire 数据: `/Users/lengyuner/Desktop/data/flywire/Jun2025/`
   - 数据文件:
     - `connections.csv` (189MB) - 连接数据
     - `visual_neuron_types.csv.gz` (617KB) - 视觉神经元类型
     - `column_assignment.csv.gz` (452KB) - 列分配/位置信息
     - `classification.csv.gz` (913KB) - 分类信息

3. **代码框架**
   - `flywire_real_data_loader.py` - 真实数据加载器
   - `flywire_data_loader.py` - 通用数据加载框架
   - `flyvis/connectome/flywire_connectome.py` - FlyWire 连接组类
   - `train_flywire_model.py` - 训练脚本框架

4. **文档**
   - `README_FLYWIRE.md` - 完整使用指南
   - `flywire_integration_plan.md` - 技术方案
   - `SUMMARY.md` - 项目总结
   - `FLYWIRE_JAXLEY_PLAN.md` (本文件)

## FlyWire 数据结构分析

### 视觉神经元数据 (visual_neuron_types.csv.gz)
```
列: root_id, type, family, subsystem, category, side
- 95,079 个视觉神经元
- 741 种细胞类型
- 64 个细胞家族
- 9 个子系统 (Motion, Color, OFF, Photoreceptors 等)
```

主要细胞类型示例:
- T4a, T4b, T4c, T4d - ON 运动检测
- T5a, T5b, T5c, T5d - OFF 运动检测
- R1-6, R7, R8 - 光感受器
- Tm1-Tm30 - 跨髓质神经元
- Mi1-Mi15 - 髓质内神经元

### 列分配数据 (column_assignment.csv.gz)
```
列: root_id, hemisphere, type, column_id, x, y, p, q
- 45,528 个神经元的列分配
- 796 个视觉列
- 31 种柱状细胞类型
- p, q: 六边形坐标系统
```

### 连接数据 (connections.csv)
```
列: pre_root_id, post_root_id, neuropil, syn_count, nt_type
- 数百万个连接
- 神经递质类型: ACH, GABA, GLUT, SER, DA, OCT
- 神经区域: ME_R (髓质), LO_R (小叶), LOP_R (小叶板), LA_R (层板)
```

## Jaxley 库分析

### 什么是 Jaxley?
Jaxley 是基于 JAX 的可微分生物物理神经元模拟器:
- **自动微分**: 支持梯度优化
- **硬件灵活**: CPU/GPU/TPU 无缝切换
- **JIT 编译**: 高性能
- **多隔室支持**: 后向欧拉求解器

### Jaxley vs Flyvis 对比

| 特性 | Flyvis (PyTorch) | Jaxley (JAX) |
|------|------------------|--------------|
| 神经元模型 | 阈值线性 (简化) | 多隔室 Hodgkin-Huxley |
| 突触模型 | 阈值线性释放 | 生物物理突触 |
| 空间结构 | 单隔室 | 多隔室 (树突、轴突) |
| 优化 | PyTorch autograd | JAX autodiff |
| 性能 | GPU 加速 | GPU/TPU 加速 + JIT |
| 生物真实性 | 中等 | 高 |
| 计算复杂度 | 低 | 高 |

### Jaxley 的层次结构
```
Network (网络)
  ├── Cell (细胞)
  │   ├── Branch (分支)
  │   │   └── Compartment (隔室)
  │   └── Branch
  └── Cell
```

## 集成方案对比

### 方案 A: 保持 Flyvis 架构 + FlyWire 数据
**优点:**
- 代码已完成 80%
- 计算效率高
- 快速验证

**缺点:**
- 生物真实性有限
- 单隔室模型

**适用场景:**
- 快速原型
- 大规模网络
- 功能验证

### 方案 B: 使用 Jaxley + FlyWire 数据
**优点:**
- 高生物真实性
- 多隔室模型
- 更精确的动力学

**缺点:**
- 需要重写大部分代码
- 计算成本高
- 需要更多生物参数

**适用场景:**
- 详细机制研究
- 小规模网络
- 生物物理建模

### 方案 C: 混合方案 (推荐)
**阶段 1: Flyvis + FlyWire**
- 使用现有框架
- 快速验证连接组
- 建立基线

**阶段 2: 关键神经元用 Jaxley**
- T4/T5 运动检测神经元
- 光感受器
- 关键中间神经元

**阶段 3: 逐步迁移**
- 根据需要增加生物细节
- 保持计算可行性

## 下一步行动计划

### 立即行动 (1-2 天)
1. ✅ 创建独立 conda 环境
2. ✅ 定位真实 FlyWire 数据
3. ✅ 创建真实数据加载器
4. 🔄 测试数据加载和统计
5. 🔄 生成 Flyvis 格式的 JSON

### 短期目标 (1 周)
6. 使用 FlyWire 数据创建第一个连接组
7. 在 Flyvis 框架中测试
8. 与原始 FIB 数据对比
9. 评估 Jaxley 的可行性

### 中期目标 (2-3 周)
10. 训练基于 FlyWire 的 Flyvis 模型
11. 验证功能特性 (ON/OFF, 方向选择性)
12. 如果需要，为关键神经元实现 Jaxley 版本
13. 性能和准确性对比

### 长期目标 (1-2 个月)
14. 完整的混合模型 (Flyvis + Jaxley)
15. 详细的机制分析
16. 发表结果

## 技术细节

### FlyWire 到 Flyvis 的映射

```python
# FlyWire 数据
{
    'root_id': 720575940596125868,
    'type': 'T5c',
    'family': 'T5 Neuron',
    'subsystem': 'Motion',
    'column_id': 97,
    'p': 6, 'q': -4  # 六边形坐标
}

# 转换为 Flyvis 格式
{
    "name": "T5c",
    "pattern": ["stride", [1, 1]],  # 每列一个
    "activation": "relu",
    ...
}
```

### 连接转换

```python
# FlyWire 连接
pre: T4a (column 487, p=1, q=1)
post: Tm9 (column 487, p=1, q=1)
syn_count: 15
nt_type: ACH

# Flyvis 格式
{
    "src": "T4a",
    "tar": "Tm9",
    "alpha": 1,  # ACH -> 兴奋性
    "offsets": [
        [[0, 0], 15]  # du=0, dv=0, 15个突触
    ]
}
```

## 资源需求

### 计算资源
- **Flyvis 方案**: 
  - GPU: 8GB+ VRAM
  - RAM: 32GB
  - 训练时间: 数小时

- **Jaxley 方案**:
  - GPU: 16GB+ VRAM
  - RAM: 64GB
  - 训练时间: 数天

### 软件依赖

**Flyvis 环境:**
```bash
conda create -n flywire_flyvis python=3.10
conda activate flywire_flyvis
pip install torch pandas numpy matplotlib
pip install -e /path/to/flyvis
```

**Jaxley 环境 (如需要):**
```bash
conda create -n jaxley_env python=3.10
conda activate jaxley_env
pip install jaxley jax[cuda]  # 或 jax[cpu]
```

## 参考资料

1. **Flyvis**: https://github.com/TuragaLab/flyvis
2. **Jaxley**: https://github.com/jaxleyverse/jaxley
3. **FlyWire**: https://flywire.ai/
4. **FlyWire 数据**: 本地 `/Users/lengyuner/Desktop/data/flywire/Jun2025/`

## 建议

基于当前情况，我建议:

1. **先完成 Flyvis + FlyWire 集成**
   - 框架已经 80% 完成
   - 可以快速看到结果
   - 建立基线性能

2. **评估是否需要 Jaxley**
   - 如果 Flyvis 结果足够好 → 继续优化
   - 如果需要更多生物细节 → 考虑 Jaxley
   - 可以混合使用

3. **优先级**
   - 高: 完成数据加载和转换
   - 高: 训练第一个模型
   - 中: 功能验证
   - 低: Jaxley 集成 (如需要)

---

**当前状态**: 环境已设置，数据已定位，框架已完成，准备测试数据加载

**下一步**: 运行 `flywire_real_data_loader.py` 验证数据加载
