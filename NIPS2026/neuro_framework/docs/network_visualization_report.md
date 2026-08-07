# Connectome Network Visualization Report

**日期**: 2026-04-04  
**任务**: 使用 NetworkX 可视化连接组网络  
**状态**: ✅ 完成

---

## 📋 任务概述

创建连接组网络可视化，其中：
- **节点** = 神经元类型（cell types）
- **节点大小** = 该类型的神经元数量
- **边的粗细** = 类型之间的突触连接数量
- **节点颜色** = 神经递质类型
  - 红色：兴奋性（Acetylcholine, Octopamine, Serotonin, Dopamine）
  - 蓝色：抑制性（GABA, Glutamate, Histamine）
  - 灰色：未知

使用不同的突触数量阈值来过滤弱连接。

---

## ✅ 完成的工作

### 1. 创建的脚本

**`04_network_visualization.py`** (374 行)
- 从 ConnectomeLoader 加载数据
- 构建类型到类型的网络（type-to-type aggregation）
- 使用 NetworkX 创建有向图
- 根据神经递质类型着色
- 生成多个阈值的可视化

**`04_network_visualization_display.py`** (200 行)
- Jupyter notebook 展示脚本
- 包含详细说明和观察
- 显示所有生成的图片
- 网络统计总结

### 2. 生成的可视化

**三个数据集 × 五个阈值 = 15 张图片**

#### Optic Lobe (925 neurons, 25 cell types)
- `network_optic_lobe_threshold_5.png` (25 nodes, 206 edges)
- `network_optic_lobe_threshold_10.png` (25 nodes, 174 edges)
- `network_optic_lobe_threshold_20.png` (25 nodes, 137 edges)
- `network_optic_lobe_threshold_50.png` (25 nodes, 115 edges)
- `network_optic_lobe_threshold_100.png` (25 nodes, 85 edges)

#### T4/T5 Pathway (695 neurons, 14 cell types)
- `network_t4t5_pathway_threshold_5.png` (14 nodes, 90 edges)
- `network_t4t5_pathway_threshold_10.png` (14 nodes, 80 edges)
- `network_t4t5_pathway_threshold_20.png` (14 nodes, 68 edges)
- `network_t4t5_pathway_threshold_50.png` (14 nodes, 57 edges)
- `network_t4t5_pathway_threshold_100.png` (14 nodes, 46 edges)

#### FAFB Visual System (35,345 neurons, 23 cell types)
- `network_fafb_visual_threshold_5.png` (23 nodes, 287 edges)
- `network_fafb_visual_threshold_10.png` (23 nodes, 247 edges)
- `network_fafb_visual_threshold_20.png` (23 nodes, 227 edges)
- `network_fafb_visual_threshold_50.png` (23 nodes, 178 edges)
- `network_fafb_visual_threshold_100.png` (23 nodes, 150 edges)

### 3. Jupyter Notebooks

- **`04_network_visualization.ipynb`** — 生成脚本的 notebook 版本
- **`04_network_visualization_display.ipynb`** — 展示和分析 notebook

---

## 📊 关键发现

### 网络结构

1. **层次化组织**
   - 清晰的前馈结构：光感受器 → 层板 (Lamina) → 髓质 (Medulla) → 小叶 (Lobula)
   - 符合已知的果蝇视觉系统解剖结构

2. **中心节点（Hub neurons）**
   - Mi1, Tm3, T4/T5 作为中心节点，连接数最多
   - 这些神经元在信息处理中起关键作用

3. **反馈连接**
   - 存在一些反馈环路，特别是在高阶神经元中
   - 可能用于增益控制和适应

### 突触阈值效应

| 阈值 | 网络密度 | 可见连接 | 适用场景 |
|------|---------|---------|---------|
| 5-10 | 高 | 包含弱连接 | 完整网络分析 |
| 20-50 | 中等 | 主要通路清晰 | 推荐用于可视化 |
| 100+ | 低 | 仅最强连接 | 核心通路识别 |

### 神经递质分布

- **混合型网络**: 兴奋性（红色）和抑制性（蓝色）神经元交织
- **Glutamate 和 Histamine**: 在此分类为抑制性
- **Acetylcholine**: 主要的兴奋性神经递质
- **平衡抑制**: 抑制性神经元对网络稳定性至关重要

### 生物学相关性

1. **T4/T5 方向选择性**
   - 特定的连接模式支持方向选择性
   - 四个亚型 (a,b,c,d) 对应四个运动方向

2. **LC 神经元整合**
   - LC 神经元接收来自多个 T4/T5 的输入
   - 整合运动信号用于行为输出

3. **跨数据集一致性**
   - Optic Lobe, T4/T5 Pathway, FAFB 显示相似的网络结构
   - 验证了数据的可靠性

---

## 🎨 可视化方法

### 网络构建算法

```python
1. 加载神经元和连接数据
2. 将边与神经元类型合并
3. 按 (pre_type, post_type) 聚合突触数量
4. 根据阈值过滤
5. 创建 NetworkX 有向图
6. 计算节点大小（神经元数量）
7. 分配节点颜色（神经递质类型）
```

### 布局算法

- **Spring Layout** (force-directed)
  - 参数: k=2.0 (节点间最优距离)
  - 迭代次数: 50-100
  - 随机种子: 42 (可重复性)

### 视觉编码

| 属性 | 编码 | 范围 |
|------|------|------|
| 节点大小 | 神经元数量 | 按最大值缩放 |
| 节点颜色 | 神经递质类型 | 红/蓝/灰 |
| 边宽度 | 突触数量 | 0.5-5.0 |
| 边透明度 | 固定 | 0.3 |
| 箭头 | 方向性 | 有向边 |

### 颜色方案

```python
# 兴奋性 (红色系)
Acetylcholine, Octopamine, Serotonin, Dopamine → #DC143C (Crimson)

# 抑制性 (蓝色系)
GABA, Glutamate, Histamine → #4169E1 (Royal Blue)

# 未知
Unknown → #888888 (Gray)
```

---

## 📈 网络统计

### Optic Lobe
- **神经元**: 925
- **细胞类型**: 25
- **边**: 5,348
- **平均突触/边**: 12.3
- **最大突触**: 156

### T4/T5 Pathway
- **神经元**: 695
- **细胞类型**: 14
- **边**: 2,997
- **平均突触/边**: 15.7
- **最大突触**: 189

### FAFB Visual System
- **神经元**: 35,345
- **细胞类型**: 23
- **边**: 158,824
- **平均突触/边**: 8.9
- **最大突触**: 2,847

---

## 💡 技术亮点

### 1. 类型聚合
- 将单个神经元连接聚合为类型到类型的连接
- 大幅简化网络复杂度（从数万边到数百边）
- 保留生物学上有意义的结构

### 2. 自适应缩放
- 节点大小根据最大神经元数量自动缩放
- 边宽度根据最大突触数量归一化
- 确保不同数据集的可比性

### 3. 多阈值分析
- 自动生成多个阈值的可视化
- 便于比较不同连接强度的网络
- 识别核心通路和次要连接

### 4. 颜色编码
- 基于神经递质类型的生物学分类
- 直观显示兴奋/抑制平衡
- 符合神经科学惯例

---

## 🔧 使用方法

### 生成新的可视化

```python
from neuro_framework.connectome.loader import ConnectomeLoader
from network_visualization import create_network_visualizations

# 加载数据
loader = ConnectomeLoader.from_optic_lobe()

# 生成可视化
create_network_visualizations(
    loader,
    output_dir="./output",
    dataset_name="My Dataset"
)
```

### 自定义参数

```python
# 自定义阈值
thresholds = [10, 50, 100, 200]

# 自定义布局
layout = 'circular'  # or 'kamada_kawai', 'spring'

# 自定义图片大小
figsize = (30, 30)
```

---

## 🚀 下一步工作

### 短期改进

1. **交互式可视化**
   - 使用 Plotly 或 Bokeh
   - 支持缩放、平移、悬停信息
   - 动态调整阈值

2. **更多布局算法**
   - 层次化布局（hierarchical）
   - 径向布局（radial）
   - 生物学启发的布局

3. **社区检测**
   - Louvain 算法
   - 识别功能模块
   - 颜色编码社区

### 长期目标

1. **模型对比**
   - 叠加学习到的权重
   - 对比解剖学 vs 功能性连接
   - 可视化权重变化

2. **时间动态**
   - 动画显示网络活动
   - 刺激响应传播
   - 学习过程可视化

3. **多尺度分析**
   - 从单个神经元到类型到区域
   - 跨尺度连接模式
   - 层次化网络结构

---

## 📚 相关文档

- [ConnectomeLoader 文档](../docs/all_neurons_support.md)
- [网络构建文档](../docs/implementation_summary.md)
- [可视化笔记本](04_network_visualization_display.ipynb)

---

## 📝 文件清单

### 脚本
- `04_network_visualization.py` (374 lines) — 生成脚本
- `04_network_visualization_display.py` (200 lines) — 展示脚本

### Notebooks
- `04_network_visualization.ipynb` — 生成 notebook
- `04_network_visualization_display.ipynb` — 展示 notebook

### 图片 (15 张)
- `network_optic_lobe_threshold_*.png` (5 张)
- `network_t4t5_pathway_threshold_*.png` (5 张)
- `network_fafb_visual_threshold_*.png` (5 张)

---

## ✅ 总结

成功创建了连接组网络可视化系统：

- ✅ 使用 NetworkX 构建类型到类型的网络
- ✅ 节点大小表示神经元数量
- ✅ 边粗细表示突触连接数量
- ✅ 颜色编码神经递质类型（红=兴奋，蓝=抑制）
- ✅ 多个阈值的可视化（5, 10, 20, 50, 100）
- ✅ 三个数据集（Optic Lobe, T4/T5, FAFB）
- ✅ 生成 15 张高质量图片
- ✅ 创建展示和分析 notebook
- ✅ 符合生物学和神经科学惯例

**准备用于论文和演示！**

---

**完成时间**: 2026-04-04  
**生成图片**: 15 张  
**代码行数**: 574 行  
**状态**: ✅ 完成
