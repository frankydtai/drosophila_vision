# 网络可视化快速指南

## 📖 查看结果

**推荐方式**：打开 Jupyter notebook
```bash
cd /Users/lengyuner/Desktop/NIPS2026/neuro_framework/notebooks
jupyter notebook 04_network_visualization_display.ipynb
```

这个 notebook 包含：
- 15 张网络可视化图片
- 详细的观察和分析
- 网络统计总结
- 方法说明

## 🎨 可视化说明

### 节点（Nodes）
- **代表**: 神经元类型（cell types）
- **大小**: 该类型的神经元数量
- **颜色**: 神经递质类型
  - 🔴 红色：兴奋性（Ach, Oct, 5HT, DA）
  - 🔵 蓝色：抑制性（GABA, Glut, Hist）
  - ⚪ 灰色：未知

### 边（Edges）
- **代表**: 类型间的突触连接
- **粗细**: 突触数量（越粗=连接越强）
- **箭头**: 连接方向（有向图）

### 阈值
- **5-10**: 密集网络，包含所有连接
- **20-50**: 平衡视图，推荐 ⭐
- **100+**: 稀疏网络，仅核心连接

## 📊 生成的图片

### Optic Lobe（925 neurons, 25 types）
```
network_optic_lobe_threshold_5.png    (25 nodes, 206 edges)
network_optic_lobe_threshold_10.png   (25 nodes, 174 edges)
network_optic_lobe_threshold_20.png   (25 nodes, 137 edges)
network_optic_lobe_threshold_50.png   (25 nodes, 115 edges)
network_optic_lobe_threshold_100.png  (25 nodes, 85 edges)
```

### T4/T5 Pathway（695 neurons, 14 types）
```
network_t4t5_pathway_threshold_5.png    (14 nodes, 90 edges)
network_t4t5_pathway_threshold_10.png   (14 nodes, 80 edges)
network_t4t5_pathway_threshold_20.png   (14 nodes, 68 edges)
network_t4t5_pathway_threshold_50.png   (14 nodes, 57 edges)
network_t4t5_pathway_threshold_100.png  (14 nodes, 46 edges)
```

### FAFB Visual System（35,345 neurons, 23 types）
```
network_fafb_visual_threshold_5.png    (23 nodes, 287 edges)
network_fafb_visual_threshold_10.png   (23 nodes, 247 edges)
network_fafb_visual_threshold_20.png   (23 nodes, 227 edges)
network_fafb_visual_threshold_50.png   (23 nodes, 178 edges)
network_fafb_visual_threshold_100.png  (23 nodes, 150 edges)
```

## 🔧 重新生成图片

如果需要重新生成或修改参数：

```bash
cd /Users/lengyuner/Desktop/NIPS2026
/Users/lengyuner/anaconda3/bin/python3 neuro_framework/notebooks/04_network_visualization.py
```

### 自定义参数

编辑 `04_network_visualization.py`：

```python
# 修改阈值
thresholds = [10, 50, 100, 200]

# 修改布局
layout = 'circular'  # 或 'kamada_kawai', 'spring'

# 修改图片大小
figsize = (30, 30)

# 修改迭代次数（影响布局质量）
iterations = 200
```

## 💡 关键发现

1. **层次化结构**: L → Mi/Tm → T4/T5 → LC 清晰可见
2. **中心节点**: Mi1, Tm3, T4/T5 是网络中心
3. **平衡网络**: 兴奋性和抑制性神经元交织
4. **阈值效应**: 中等阈值（20-50）最适合可视化

## 📚 相关文档

- **详细报告**: `docs/network_visualization_report.md`
- **展示 Notebook**: `notebooks/04_network_visualization_display.ipynb`
- **生成脚本**: `notebooks/04_network_visualization.py`

## ✅ 文件位置

```
neuro_framework/
├── notebooks/
│   ├── 04_network_visualization.py              # 生成脚本
│   ├── 04_network_visualization.ipynb           # 生成 notebook
│   ├── 04_network_visualization_display.py      # 展示脚本
│   ├── 04_network_visualization_display.ipynb   # 展示 notebook ⭐
│   ├── network_optic_lobe_threshold_*.png       # 5 张图片
│   ├── network_t4t5_pathway_threshold_*.png     # 5 张图片
│   └── network_fafb_visual_threshold_*.png      # 5 张图片
└── docs/
    └── network_visualization_report.md          # 详细报告
```

---

**创建日期**: 2026-04-04  
**图片数量**: 15 张  
**总大小**: ~8 MB  
**状态**: ✅ 完成
