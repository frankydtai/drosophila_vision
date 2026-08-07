# 📚 FlyWire 项目文档索引

## 项目结构

```
flyvis/
├── notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_explore_connectome.ipynb
│   ├── 02_visualize_connections.ipynb
│   └── outputs/                   # 📊 输出图片
│       ├── connectome/
│       ├── connections/
│       └── analysis/
│
├── flywire_docs/                  # 📖 FlyWire 集成文档
│   ├── 00_START_HERE.md
│   ├── 01_README.md
│   ├── 02_QUICKSTART.md
│   ├── 03_TECHNICAL_DETAILS.md
│   ├── 04_JAXLEY_ANALYSIS.md
│   ├── 05_COMPLETION_REPORT.md
│   ├── ANSWERS.md
│   ├── SUMMARY.md
│   └── ...
│
├── 01_load_flywire_data.py        # 核心脚本
├── 02_convert_to_flyvis.py
├── 03_verify_connectome.py
└── explore_flywire_connectome.py
```

---

## 📓 Notebooks

### 01_explore_connectome.ipynb
**用途**: 基础连接组探索

**内容**:
- 加载 FlyWire 连接组数据
- 基本统计信息
- 连接类型分布
- 连接度分析
- 关键神经元识别

**输出**: `notebooks/outputs/connectome/`
- connection_types.png
- degree_distribution.png

---

### 02_visualize_connections.ipynb
**用途**: 多样化连接可视化

**内容**:
1. **网络图** - 展示主要节点的连接关系
2. **连接矩阵热图** - 显示细胞类型间的连接强度
3. **层次聚类** - 基于连接模式的细胞类型分组
4. **输入输出路径** - 从光感受器到运动检测的信号流
5. **连接强度分布** - 突触数量的统计分析

**输出**: `notebooks/outputs/connections/`
- network_graph.png
- connection_matrix_heatmap.png
- hierarchical_clustering.png
- input_output_pathways.png
- connection_strength_distribution.png

---

## 📖 文档

### 快速开始
- **00_START_HERE.md** - 项目快速入口
- **02_QUICKSTART.md** - 快速开始指南

### 核心文档
- **01_README.md** - 项目主页
- **ANSWERS.md** - 核心问题解答（重点！）
- **03_TECHNICAL_DETAILS.md** - 技术细节

### 参考文档
- **04_JAXLEY_ANALYSIS.md** - Jaxley 分析
- **05_COMPLETION_REPORT.md** - 完成报告
- **SUMMARY.md** - 项目总结

---

## 🚀 快速开始

### 运行 Notebooks

```bash
# 1. 激活环境
conda activate flywire_flyvis

# 2. 安装依赖
pip install jupyter matplotlib seaborn pandas networkx scipy

# 3. 启动 Jupyter
cd /Users/lengyuner/Desktop/NIPS2026/flyvis
jupyter notebook notebooks/

# 4. 打开并运行
# - 01_explore_connectome.ipynb
# - 02_visualize_connections.ipynb
```

### 查看文档

```bash
# 查看核心问题解答
cat flywire_docs/ANSWERS.md

# 查看技术细节
cat flywire_docs/03_TECHNICAL_DETAILS.md

# 查看快速开始
cat flywire_docs/02_QUICKSTART.md
```

---

## 📊 输出文件组织

### notebooks/outputs/connectome/
基础连接组分析图表
- connection_types.png - 连接类型分布
- degree_distribution.png - 连接度分布

### notebooks/outputs/connections/
连接可视化图表
- network_graph.png - 网络图
- connection_matrix_heatmap.png - 连接矩阵热图
- hierarchical_clustering.png - 层次聚类
- input_output_pathways.png - 输入输出路径
- connection_strength_distribution.png - 连接强度分布

### notebooks/outputs/analysis/
（预留）高级分析图表

---

## 🔧 核心脚本

### 01_load_flywire_data.py
加载和过滤 FlyWire 数据

### 02_convert_to_flyvis.py
转换为 Flyvis JSON 格式

### 03_verify_connectome.py
验证连接组集成

### explore_flywire_connectome.py
独立的探索脚本（可直接运行）

---

## 📝 改进说明

### 1. 项目结构优化
- ✅ 创建 `notebooks/` 目录存放所有 notebooks
- ✅ 创建 `notebooks/outputs/` 存放输出图片
- ✅ 创建 `flywire_docs/` 存放文档
- ✅ 根目录保持简洁

### 2. 中文支持增强
- ✅ 所有图表标签支持中文
- ✅ 设置 Arial Unicode MS 字体
- ✅ 解决负号显示问题

### 3. 可视化多样化
- ✅ 网络图（NetworkX）
- ✅ 连接矩阵热图
- ✅ 层次聚类树状图
- ✅ 输入输出路径图
- ✅ 连接强度分布

### 4. 代码改进
- ✅ 统一输出目录管理
- ✅ 更好的错误处理
- ✅ 清晰的代码结构
- ✅ 详细的注释

---

## 🎯 使用建议

### 初学者
1. 阅读 `flywire_docs/00_START_HERE.md`
2. 运行 `notebooks/01_explore_connectome.ipynb`
3. 查看生成的图表

### 进阶用户
1. 阅读 `flywire_docs/ANSWERS.md`
2. 运行 `notebooks/02_visualize_connections.ipynb`
3. 自定义分析参数

### 高级用户
1. 阅读 `flywire_docs/03_TECHNICAL_DETAILS.md`
2. 修改核心脚本
3. 创建新的分析 notebook

---

## 📞 相关资源

- **FlyWire 数据**: `/Users/lengyuner/Desktop/data/flywire/Jun2025/`
- **Flyvis 文档**: `docs/docs/`
- **原始论文**: https://www.nature.com/articles/s41586-024-07939-3

---

## ✅ 完成状态

- ✅ 项目结构重组
- ✅ 中文支持完善
- ✅ 多样化可视化
- ✅ 文档整理
- ✅ 代码优化

🎉 项目已完成整理！
