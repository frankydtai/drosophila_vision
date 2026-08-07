# FlyWire 连接组可视化 Notebooks

这个目录包含用于探索和可视化 FlyWire 连接组数据的 Jupyter Notebooks。

## 📓 Notebooks

### 01_explore_connectome.ipynb
**基础连接组探索**

探索 FlyWire 连接组的基本统计信息和结构。

**内容：**
- 加载 FlyWire 连接组数据
- 基本统计信息（节点数、边数、连接类型）
- 连接度分布分析
- 关键神经元类型识别（输入/输出神经元）
- 基础可视化（连接类型分布、度分布）

**生成的图表：**
- `outputs/connectome/connection_types.png` - 连接类型分布（兴奋性 vs 抑制性）
- `outputs/connectome/degree_distribution.png` - 连接度分布

---

### 02_visualize_connections.ipynb
**多样化连接可视化**

使用多种方式深入展示 FlyWire 连接组的连接模式。

**内容：**
1. **网络图** - 展示 Top 30 高连接度节点的网络结构
2. **连接矩阵热图** - Top 40 细胞类型的连接强度矩阵
3. **层次聚类分析** - 基于连接模式的细胞类型聚类
4. **输入-输出通路图** - 从光感受器到运动检测神经元的信息流
5. **连接强度分布** - 突触数量的统计分布

**生成的图表：**
- `outputs/connections/network_graph.png` - 网络图
- `outputs/connections/connection_matrix_heatmap.png` - 连接矩阵热图
- `outputs/connections/hierarchical_clustering.png` - 层次聚类树状图
- `outputs/connections/input_output_pathway.png` - 输入-输出通路图
- `outputs/connections/synapse_distribution.png` - 突触数量分布

---

### 03_network_hierarchy.ipynb
**网络层次结构分析**

使用广度优先搜索（BFS）从输入神经元出发，将所有神经元按层级分类并可视化。

**内容：**
1. **BFS 分层** - 从 R1-6, R7, R8 出发进行广度优先搜索
2. **层级统计** - 每层的神经元数量和分布
3. **层级可视化** - 柱状图、网络图、流动图
4. **输出神经元定位** - T4/T5 神经元在哪一层

**关键发现：**
- 网络共 5 层（第 0-4 层）
- 第 0 层：3 个输入神经元（R1-6, R7, R8）
- 第 1 层：15 个神经元
- 第 2 层：81 个神经元（包含 T4a/b/c/d）
- 第 3 层：44 个神经元（包含 T5a/b/c/d）
- 第 4 层：2 个神经元

**生成的图表：**
- `outputs/hierarchy/layer_distribution.png` - 层级分布柱状图
- `outputs/hierarchy/hierarchy_network.png` - 层次结构网络图
- `outputs/hierarchy/layer_flow.png` - 层级流动图

**导出数据：**
- `outputs/hierarchy/neuron_layers.csv` - 每个神经元的层级信息
- `outputs/hierarchy/layer_info.json` - 完整层级结构数据

---

### 04_network_activation.ipynb
**网络激活状态可视化**

模拟或运行 FlyWire 网络，给予视觉刺激，可视化神经元的激活状态分布。

**内容：**
1. **激活数据生成** - 使用 Flyvis 模型或模拟数据
2. **激活统计分析** - 每个神经元的平均激活、标准差
3. **按层级分析** - 不同层级的激活模式
4. **时间序列可视化** - 神经元激活随时间的变化
5. **输出神经元分析** - T4/T5 的方向选择性激活

**关键发现：**
- 输入神经元（R1-6, R7, R8）激活最高（~0.85）
- 激活随层级递减（第 0 层 0.85 → 第 4 层 0.28）
- 输出神经元（T4/T5）显示方向选择性激活模式

**生成的图表：**
- `outputs/activation/activation_distribution.png` - 激活分布（4个子图）
- `outputs/activation/activation_heatmap.png` - 激活时间序列热图
- `outputs/activation/output_neurons_activation.png` - 输出神经元激活

**导出数据：**
- `outputs/activation/activation_stats.csv` - 激活统计
- `outputs/activation/activations_sampled.json` - 采样激活数据

---

## 🚀 使用方法

### 1. 激活环境
```bash
conda activate flywire_flyvis
```

### 2. 安装依赖
```bash
# 基础依赖（如果还没安装）
pip install numpy pandas matplotlib seaborn

# 网络分析和聚类
pip install networkx scipy

# 可选：运行真实 Flyvis 模型（用于 04_network_activation.ipynb）
pip install torch torchvision torchaudio
```

### 3. 启动 Jupyter
```bash
cd /Users/lengyuner/Desktop/NIPS2026/flyvis/notebooks
jupyter notebook
```

### 4. 运行 Notebooks
在 Jupyter 界面中：
1. 按顺序打开 notebooks（01 → 02 → 03 → 04）
2. 点击 "Cell" → "Run All" 运行所有单元格
3. 查看生成的可视化结果

**推荐顺序：**
1. `01_explore_connectome.ipynb` - 了解基础结构
2. `02_visualize_connections.ipynb` - 深入连接模式
3. `03_network_hierarchy.ipynb` - 理解层次结构（生成 layer_info.json）
4. `04_network_activation.ipynb` - 查看激活状态（依赖 03 的输出）

---

## 📊 输出目录结构

```
notebooks/
├── outputs/
│   ├── connectome/          # 基础连接组可视化（272 KB）
│   │   ├── connection_types.png
│   │   └── degree_distribution.png
│   ├── connections/         # 多样化连接可视化（3.8 MB）
│   │   ├── network_graph.png
│   │   ├── connection_matrix_heatmap.png
│   │   ├── hierarchical_clustering.png
│   │   ├── input_output_pathway.png
│   │   └── synapse_distribution.png
│   ├── hierarchy/           # 网络层次结构（3.6 MB）
│   │   ├── layer_distribution.png
│   │   ├── hierarchy_network.png
│   │   ├── layer_flow.png
│   │   ├── neuron_layers.csv
│   │   └── layer_info.json
│   └── activation/          # 网络激活状态（1.4 MB）
│       ├── activation_distribution.png
│       ├── activation_heatmap.png
│       ├── output_neurons_activation.png
│       ├── activation_stats.csv
│       └── activations_sampled.json
```

---

## 🔧 技术细节

### 中文字体支持
两个 notebooks 都配置了中文字体支持：
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 数据来源
- **连接组文件**: `../flyvis/connectome/flywire_v1.0.json`
- **细胞类型**: 146 种
- **连接数**: 2,071 个
- **输入神经元**: R1-6, R7, R8（光感受器）
- **输出神经元**: T4a/b/c/d, T5a/b/c/d（运动检测）

### 关键统计
- **兴奋性连接**: 1,418 (68.5%)
- **抑制性连接**: 653 (31.5%)
- **平均突触数**: 990.5
- **突触数范围**: 10 - 90,970

---

## 🐛 调试说明

如果遇到问题，可以使用 Python 脚本版本进行调试：

```bash
# 运行 Python 脚本版本
python 01_explore_connectome.py
python 02_visualize_connections.py

# 修改后转换回 notebook
python -m jupytext --to notebook 01_explore_connectome.py
```

---

## 📚 相关文档

- **项目文档**: `../flywire_docs/`
- **技术细节**: `../flywire_docs/03_TECHNICAL_DETAILS.md`
- **快速开始**: `../flywire_docs/02_QUICKSTART.md`
- **完成报告**: `../flywire_docs/05_COMPLETION_REPORT.md`

---

## ⚠️ 注意事项

1. **字体警告**: 运行时可能会看到中文字体缺失的警告，这不影响图片生成，只是某些中文字符可能显示为方框。
2. **内存使用**: `02_visualize_connections.ipynb` 中的网络图可视化可能需要较多内存。
3. **运行时间**: 完整运行两个 notebooks 大约需要 2-3 分钟。

---

**最后更新**: 2026-03-15
