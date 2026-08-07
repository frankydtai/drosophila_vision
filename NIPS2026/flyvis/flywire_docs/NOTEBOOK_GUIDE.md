# 📓 Jupyter Notebook 使用指南

## 文件信息

**文件名**: `explore_flywire_connectome.ipynb`  
**位置**: `/Users/lengyuner/Desktop/NIPS2026/flyvis/`  
**大小**: ~20 KB  
**用途**: 探索和可视化 FlyWire 连接组数据

---

## 快速开始

### 1. 激活环境
```bash
conda activate flywire_flyvis
```

### 2. 安装 Jupyter（如果还没有）
```bash
pip install jupyter notebook matplotlib seaborn pandas
```

### 3. 启动 Notebook
```bash
cd /Users/lengyuner/Desktop/NIPS2026/flyvis
jupyter notebook explore_flywire_connectome.ipynb
```

### 4. 运行代码
- 方式 1: 点击 `Cell → Run All` 运行所有单元格
- 方式 2: 按 `Shift + Enter` 逐个运行单元格

---

## Notebook 内容

### 📊 第 1 部分: 加载数据
- 导入必要的库（numpy, pandas, matplotlib, seaborn）
- 加载 `flywire_v1.0.json` 文件
- 显示基本信息（节点数、边数、输入/输出单元）

**预期输出**:
```
✓ 加载 FlyWire 连接组: flyvis/connectome/flywire_v1.0.json
  文件大小: 542.0 KB

数据结构:
  - 节点数: 146
  - 边数: 2071
  - 输入单元: ['R1-6', 'R7', 'R8']
  - 输出单元: ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
```

---

### 📈 第 2 部分: 基本统计
- 细胞类型列表
- 兴奋性 vs 抑制性连接统计
- 空间偏移信息统计
- 连接度分析（输入/输出连接最多的细胞类型）

**预期输出**:
```
边统计:
  总边数: 2071
  兴奋性连接: 1856 (89.6%)
  抑制性连接: 215 (10.4%)
  有空间偏移: 119 (5.7%)

连接度最高的细胞类型:
输出连接最多 (Top 10):
  Mi1: 45 个目标
  Tm3: 38 个目标
  ...
```

---

### 🎨 第 3 部分: 可视化连接组结构

#### 图表 1: 连接类型分布
- **左图**: 兴奋性 vs 抑制性连接（饼图）
- **右图**: 有/无空间偏移信息（柱状图）

#### 图表 2: 连接度分布
- **左图**: 输入连接度分布（直方图）
- **右图**: 输出连接度分布（直方图）

**预期效果**: 2 个图表，每个包含 2 个子图

---

### 🧠 第 4 部分: 关键神经元分析

#### 输入神经元（光感受器）
- R1-6: 主要光感受器
- R7, R8: 颜色感受器

#### 输出神经元（运动检测）
- T4a-d: ON 通路运动检测（4 个方向）
- T5a-d: OFF 通路运动检测（4 个方向）

#### 图表 3: T4/T5 输入连接热图
- 显示每个 T4/T5 神经元接收来自哪些细胞类型的输入
- 颜色深度表示连接数量

**预期效果**: 大型热图，展示 T4/T5 的输入模式

---

### 🗺️ 第 5 部分: 空间偏移分析

#### 图表 4: 空间偏移可视化
- **左图**: 散点图显示 (du, dv) 分布，颜色表示突触数量
- **右图**: 热图显示偏移模式的频率

**关键发现**:
- 大部分连接集中在 (0, 0)，即同一列内的连接
- 相邻列的连接（如 (1, 0), (0, 1)）也很常见
- 六边形坐标系统清晰可见

---

### 🔧 第 6 部分: 使用 Flyvis 框架

如果已安装 Flyvis，这部分会:
1. 导入 `ConnectomeFromFlyWire` 类
2. 创建连接组对象
3. 创建 `ConnectomeView`
4. 显示详细统计信息

**如果 Flyvis 未安装**:
```
⚠ Flyvis 未安装或导入失败
  请运行: pip install -e .
```

**如果成功**:
```
✓ FlyWire 连接组已加载
✓ ConnectomeView 已创建

连接组统计:
  unique_cell_types: 146
  total_edges: 2071
  ...
```

---

### 📊 第 7 部分: 对比 FlyWire 和 FIB

如果原始 FIB 连接组文件存在，这部分会:
1. 加载 `fib25-fib19_v2.2.json`
2. 对比节点数、边数、输入/输出单元
3. 生成对比柱状图

**预期对比**:
```
              FlyWire  FIB
节点数            146   64
边数            2071  XXX
输入单元            3    3
输出单元            8    8
```

---

### 📝 第 8 部分: 总结

生成完整的探索总结报告，包括:
- 细胞类型数量
- 连接统计
- 关键神经元
- 空间偏移信息

---

## 预期可视化图表

运行完整个 notebook 后，您将看到:

1. **连接类型分布图** (2 子图)
   - 饼图: 兴奋性 vs 抑制性
   - 柱状图: 有/无空间偏移

2. **连接度分布图** (2 子图)
   - 输入连接度直方图
   - 输出连接度直方图

3. **T4/T5 输入热图** (1 大图)
   - 显示 T4/T5 神经元的输入模式

4. **空间偏移可视化** (2 子图)
   - 散点图: (du, dv) 分布
   - 热图: 偏移模式频率

5. **FlyWire vs FIB 对比图** (1 图)
   - 柱状图对比两个数据集

**总计**: 至少 8 个子图 / 5 个完整图表

---

## 常见问题

### Q1: Jupyter 无法启动？
```bash
# 安装 Jupyter
pip install jupyter notebook

# 或使用 conda
conda install jupyter notebook
```

### Q2: 缺少可视化库？
```bash
pip install matplotlib seaborn pandas numpy
```

### Q3: Flyvis 导入失败？
```bash
# 在 flyvis 目录下安装
cd /Users/lengyuner/Desktop/NIPS2026/flyvis
pip install -e .
```

### Q4: 图表不显示？
```python
# 在 notebook 开头添加
%matplotlib inline
```

### Q5: 想保存图表？
```python
# 在每个 plt.show() 之前添加
plt.savefig('figure_name.png', dpi=300, bbox_inches='tight')
```

---

## 自定义和扩展

### 修改过滤参数
如果想查看更多数据，可以重新生成连接组:
```bash
python 02_convert_to_flyvis.py
```
然后修改其中的参数:
```python
converter.convert(
    subsystems=['Motion', 'Color', 'OFF', 'Photoreceptors', 'Form'],  # 添加更多
    min_syn_count=5  # 降低阈值
)
```

### 添加自定义分析
在 notebook 中添加新的单元格，例如:
```python
# 分析特定细胞类型
mi1_connections = [e for e in edges if e['src'] == 'Mi1']
print(f"Mi1 的输出连接: {len(mi1_connections)}")
```

### 导出数据
```python
# 导出为 CSV
import pandas as pd
edge_df = pd.DataFrame(edges)
edge_df.to_csv('flywire_edges.csv', index=False)
```

---

## 性能提示

- **大数据集**: 如果使用全部 741 种细胞类型，某些可视化可能需要更长时间
- **内存**: 热图可能占用较多内存，可以限制显示的细胞类型数量
- **交互式**: 考虑使用 `plotly` 或 `bokeh` 创建交互式图表

---

## 下一步

完成 notebook 探索后，您可以:

1. **训练模型**: 使用 FlyWire 连接组训练 Flyvis 网络
2. **扩展数据**: 包含更多细胞类型和连接
3. **深入分析**: 研究特定通路（如 ON/OFF 通路）
4. **对比实验**: 比较 FlyWire 和 FIB 模型的性能

---

## 参考资料

- **FlyWire 数据**: `/Users/lengyuner/Desktop/data/flywire/Jun2025/`
- **技术细节**: `03_TECHNICAL_DETAILS.md`
- **核心问题**: `ANSWERS.md`
- **快速开始**: `02_QUICKSTART.md`

---

🎉 享受探索 FlyWire 连接组的过程！
