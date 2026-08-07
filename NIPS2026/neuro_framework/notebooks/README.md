# Notebooks 目录说明

**更新日期**: 2026-04-04

---

## 📚 Notebook 列表

### 入门指南

**`00_quick_start.ipynb`** ⭐ 推荐新手
- **目的**: 快速入门，展示如何加载和探索连接组数据
- **内容**:
  - 加载不同数据源（BANC, FAFB, Optic Lobe）
  - 使用 print 函数展示神经元和连接
  - 基本统计和数据探索
  - 数据集对比
- **运行时间**: ~30 秒
- **输出**: 控制台输出（无图片）

### 核心演示

**`01_connectome_and_network.ipynb`** / **`01_connectome_and_network_v2.ipynb`**
- **目的**: 完整演示连接组加载、网络构建和模拟
- **内容**:
  - 加载 Optic Lobe 和 BANC 数据
  - 构建 PyTorch 网络（Voltage, LIF, HH 模型）
  - 前向传播和梯度流测试
  - 活动可视化
- **运行时间**: ~2 分钟
- **输出**: 5 张图片（保存在 `figures/`）
  - `fig1_cell_types.png` - 细胞类型分布
  - `fig2_activity.png` - 神经活动热图
  - `fig3_mean_activity.png` - 平均活动
  - `fig4_connectivity.png` - 连接统计
  - `fig5_nt_distribution.png` - 神经递质分布

### 详细分析

**`02_full_connectome_visualization.ipynb`**
- **目的**: 完整视叶连接组的详细可视化和分析
- **数据**: Optic Lobe (925 neurons, 25 cell types)
- **内容**:
  - 细胞类型统计
  - 连接矩阵
  - 层次聚类
  - 功能分组分析
  - 网络拓扑
- **运行时间**: ~1 分钟
- **输出**: 5 张图片
  - `fig_full_01_overview.png`
  - `fig_full_02_connectivity_matrix.png`
  - `fig_full_03_clustering.png`
  - `fig_full_04_functional_groups.png`
  - `fig_full_05_topology.png`

**`03_fafb_full_brain.ipynb`**
- **目的**: FAFB 全脑连接组的大规模分析
- **数据**: FAFB v783 (138,043 neurons, 8,772 cell types)
- **内容**:
  - Super-class 分析
  - 细胞类型多样性
  - 视觉系统分析
  - 连接模式分析
- **运行时间**: ~3 分钟
- **输出**: 4 张图片
  - `fig_fafb_01_super_class.png`
  - `fig_fafb_02_cell_types.png`
  - `fig_fafb_03_visual_system.png`
  - `fig_fafb_04_connectivity.png`

### 网络可视化

**`04_network_visualization.ipynb`**
- **目的**: 使用 NetworkX 生成网络图
- **内容**:
  - 类型到类型的网络聚合
  - 多阈值可视化（5, 10, 20, 50, 100 突触）
  - 神经递质颜色编码
- **运行时间**: ~10 秒
- **输出**: 15 张网络图（保存在 `figures/`）

**`04_network_visualization_display.ipynb`** ⭐ 推荐查看
- **目的**: 展示所有网络可视化结果
- **内容**:
  - 显示 15 张网络图
  - 详细观察和分析
  - 网络统计总结
- **运行时间**: <1 秒（仅显示）
- **输出**: 无（显示已有图片）

### 突触模型 ⭐ NEW

**`05_synapse_models.ipynb`**
- **目的**: 演示不同的突触模型（受 Jaxley 启发）
- **内容**:
  - Simple 权重突触（默认）
  - TanhRateSynapse（无状态）
  - TanhConductanceSynapse（电导模型）
  - IonotropicSynapse（生物物理模型，有状态变量）
  - 模型对比和可视化
  - 训练示例
- **运行时间**: ~1 分钟
- **输出**: 3 张图片（保存在 `figures/`）
  - `synapse_models_comparison.png` - 活动热图对比
  - `synapse_models_traces.png` - 时间序列对比
  - `synapse_training_curve.png` - 训练曲线
- **参考**: Jaxley, Abbott & Marder (1998)

---

## 📁 文件组织

```
notebooks/
├── 00_quick_start.ipynb              ⭐ 入门指南
├── 01_connectome_and_network.ipynb   核心演示
├── 02_full_connectome_visualization.ipynb  视叶分析
├── 03_fafb_full_brain.ipynb          全脑分析
├── 04_network_visualization.ipynb    网络生成
├── 04_network_visualization_display.ipynb  ⭐ 网络展示
├── figures/                          📊 所有图片输出
│   ├── fig1_cell_types.png
│   ├── fig2_activity.png
│   ├── ...
│   ├── network_optic_lobe_threshold_5.png
│   └── ...
├── *.py                              Python 脚本版本
└── README.md                         本文件
```

---

## 🚀 快速开始

### 1. 新手入门
```bash
jupyter notebook 00_quick_start.ipynb
```
运行所有单元格，查看如何加载和探索数据。

### 2. 查看网络可视化
```bash
jupyter notebook 04_network_visualization_display.ipynb
```
运行所有单元格，查看 15 张网络图和分析。

### 3. 完整演示
```bash
jupyter notebook 01_connectome_and_network.ipynb
```
运行所有单元格，体验完整的工作流程。

### 4. 突触模型 ⭐ NEW
```bash
jupyter notebook 05_synapse_models.ipynb
```
运行所有单元格，了解不同的突触模型及其应用。

---

## 📊 图片输出

所有图片统一保存在 `figures/` 文件夹中：

### 基础演示图片（5 张）
- `fig1_cell_types.png` - 细胞类型分布
- `fig2_activity.png` - 神经活动热图
- `fig3_mean_activity.png` - 平均活动
- `fig4_connectivity.png` - 连接统计
- `fig5_nt_distribution.png` - 神经递质分布

### 视叶分析图片（5 张）
- `fig_full_01_overview.png` - 概览
- `fig_full_02_connectivity_matrix.png` - 连接矩阵
- `fig_full_03_clustering.png` - 聚类
- `fig_full_04_functional_groups.png` - 功能分组
- `fig_full_05_topology.png` - 拓扑

### FAFB 分析图片（4 张）
- `fig_fafb_01_super_class.png` - 超类分析
- `fig_fafb_02_cell_types.png` - 细胞类型
- `fig_fafb_03_visual_system.png` - 视觉系统
- `fig_fafb_04_connectivity.png` - 连接模式

### 网络可视化图片（15 张）
- `network_optic_lobe_threshold_*.png` (5 张)
- `network_t4t5_pathway_threshold_*.png` (5 张)
- `network_fafb_visual_threshold_*.png` (5 张)

### 突触模型图片（3 张）⭐ NEW
- `synapse_models_comparison.png` - 活动热图对比
- `synapse_models_traces.png` - 时间序列对比
- `synapse_training_curve.png` - 训练曲线

**总计**: 32 张图片

---

## 🔧 重新生成图片

如果需要重新生成所有图片：

```bash
# 基础演示
python test_demo.py

# 视叶分析
python 02_full_connectome_visualization.py

# FAFB 分析
python 03_fafb_full_brain.py

# 网络可视化
python 04_network_visualization.py
```

所有图片将自动保存到 `figures/` 文件夹。

---

## 📝 Python 脚本

每个 notebook 都有对应的 `.py` 脚本版本：
- 便于版本控制
- 可以直接运行（无需 Jupyter）
- 使用 `jupytext` 同步

### 转换命令

```bash
# Notebook → Python
jupytext --to py:percent notebook.ipynb

# Python → Notebook
jupytext --to notebook script.py
```

---

## 💡 使用建议

### 学习路径

1. **第一步**: `00_quick_start.ipynb` - 了解数据加载
2. **第二步**: `04_network_visualization_display.ipynb` - 查看网络结构
3. **第三步**: `01_connectome_and_network.ipynb` - 学习网络构建
4. **第四步**: `02_full_connectome_visualization.ipynb` - 深入分析

### 数据集选择

- **测试/学习**: Optic Lobe (925 neurons) - 快速
- **中等规模**: T4/T5 Pathway (695 neurons) - 特定通路
- **大规模**: FAFB Visual (35k neurons) - 完整视觉系统
- **全脑**: BANC (115k neurons) 或 FAFB (138k neurons)

### 性能提示

- 小数据集 (<1k neurons): 秒级
- 中等数据集 (1k-10k neurons): 分钟级
- 大数据集 (>10k neurons): 可能需要数分钟

---

## 📚 相关文档

- **快速参考**: `NETWORK_VISUALIZATION_README.md`
- **详细报告**: `../docs/network_visualization_report.md`
- **全神经元支持**: `../docs/all_neurons_support.md`
- **项目文档**: `../docs/SUMMARY.md`

---

## ✅ 检查清单

运行 notebooks 前，确保：

- [ ] 已安装所有依赖（pandas, numpy, torch, networkx, matplotlib, seaborn）
- [ ] FAFB 数据已下载到 `/Users/lengyuner/Desktop/data/flywire/Jun2025`
- [ ] `figures/` 文件夹存在（会自动创建）
- [ ] Python 环境正确（推荐使用 Anaconda base）

---

**最后更新**: 2026-04-04  
**Notebooks 数量**: 6 个  
**图片数量**: 29 张  
**状态**: ✅ 已整理
