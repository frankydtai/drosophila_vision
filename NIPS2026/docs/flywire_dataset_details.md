# FlyWire数据集详细说明

## 数据集概述

**数据集名称**: FlyWire FAFB v783 (Female Adult Fly Brain)

**基本信息**:
- 细胞数量: 139,255
- 突触数量: 50,666,648
- 标注数量: 1,168,054
- 最后更新: 2025年6月23日

**数据访问**: https://codex.flywire.ai/

---

## 可下载数据文件

### 1. Cell Types (细胞类型)
**文件**: `consolidated_cell_types.csv.gz`
**大小**: 902 KB

**内容**:
- 行数: 138,327 (+ header)
- 列数: 3

**列说明**:
- `root_id`: 细胞唯一ID (138,327个唯一值)
- `primary_type`: 主要细胞类型 (8,772个唯一值)
- `additional_type(s)`: 额外类型信息 (1,508个唯一值，13,956行有数据)

**用途**: 识别神经元类型，筛选特定功能的神经元

**解压命令**:
```bash
gunzip consolidated_cell_types.csv.gz
```

---

### 2. Classification (分类/层级标注)
**文件**: `classification.csv.gz`
**大小**: 934 KB

**内容**:
- 行数: 139,255 (+ header)
- 列数: 8

**列说明**:
- `root_id`: 细胞唯一ID
- `flow`: 流向 (3个唯一值)
- `super_class`: 超类 (10个唯一值)
- `class`: 类别 (29个唯一值，107,591行有数据)
- `sub_class`: 子类 (118个唯一值，100,236行有数据)
- `hemilineage`: 半谱系 (214个唯一值，37,542行有数据)
- `side`: 侧别 (3个唯一值，139,225行有数据)
- `nerve`: 神经 (8个唯一值，9,647行有数据)

**用途**: 层级分类，理解神经元功能组织

**解压命令**:
```bash
gunzip classification.csv.gz
```

---

### 3. Cell Size Measurements (细胞尺寸测量)
**文件**: `cell_stats.csv.gz`
**大小**: 2,527 KB

**内容**:
- 行数: 139,246 (+ header)
- 列数: 4

**列说明** (单位: 纳米):
- `root_id`: 细胞唯一ID
- `length_nm`: 长度 (范围: 4,930 - 103,733,576)
- `area_nm`: 表面积 (范围: 15,141,888 - 258,113,806,848)
- `size_nm`: 体积 (范围: 427,591,680 - 16,416,917,166,080)

**用途**: 分析神经元形态特征

**解压命令**:
```bash
gunzip cell_stats.csv.gz
```

---

### 4. Proofread Cell Names And Groups (细胞名称和分组)
**文件**: `names.csv.gz`
**大小**: 1,182 KB

**内容**:
- 行数: 139,255 (+ header)
- 列数: 3

**列说明**:
- `root_id`: 细胞唯一ID
- `name`: 自动生成的名称 (所有行唯一)
- `group`: 分组 (629个唯一值)

**说明**: 分组基于输入/输出突触最多的脑区

**用途**: 神经元命名和功能分组

**解压命令**:
```bash
gunzip names.csv.gz
```

---

### 5. Neurotransmitter Type Predictions (神经递质类型预测)
**文件**: `neurons.csv.gz`
**大小**: 1,680 KB

**内容**:
- 行数: 139,255 (+ header)
- 列数: 10

**列说明**:
- `root_id`: 细胞唯一ID
- `group`: 分组 (629个唯一值)
- `nt_type`: 神经递质类型 (6个唯一值，119,597行有数据)
- `nt_type_score`: 预测置信度 (0-1)
- `da_avg`: 多巴胺平均分数
- `ser_avg`: 血清素平均分数
- `gaba_avg`: GABA平均分数
- `glut_avg`: 谷氨酸平均分数
- `ach_avg`: 乙酰胆碱平均分数
- `oct_avg`: 章鱼胺平均分数

**用途**: 确定神经元的神经递质类型，理解兴奋/抑制性质

**解压命令**:
```bash
gunzip neurons.csv.gz
```

---

### 6. Visual Neuron Annotations (视觉神经元标注)
**文件**: `visual_neuron_types.csv.gz`
**大小**: 632 KB

**内容**:
- 行数: 95,079 (+ header)
- 列数: 6

**列说明**:
- `root_id`: 细胞唯一ID
- `type`: 类型 (741个唯一值)
- `family`: 家族 (64个唯一值，95,046行有数据)
- `subsystem`: 子系统 (9个唯一值，86,635行有数据)
- `category`: 类别 (2个唯一值)
- `side`: 侧别 (3个唯一值)

**用途**: 专门用于视觉系统研究

**解压命令**:
```bash
gunzip visual_neuron_types.csv.gz
```

---

### 7. Visual Neuron Columns (视觉神经元列)
**文件**: `column_assignment.csv.gz`
**大小**: 463 KB

**内容**:
- 行数: 45,528 (+ header)
- 列数: 8

**列说明**:
- `root_id`: 细胞唯一ID
- `hemisphere`: 半球 (2个唯一值)
- `type`: 类型 (31个唯一值)
- `column_id`: 列ID (796个唯一值)
- `x`, `y`, `p`, `q`: 视网膜拓扑坐标

**用途**: 柱状细胞类型的视网膜拓扑坐标分配

**解压命令**:
```bash
gunzip column_assignment.csv.gz
```

---

### 8. Community Labels (Raw) (社区标签-原始)
**文件**: `labels.csv.gz`
**大小**: 4,771 KB

**内容**:
- 行数: 160,045 (+ header)
- 列数: 9

**列说明**:
- `root_id`: 细胞ID (110,038个唯一值，可重复)
- `label`: 标签 (7,573个唯一值)
- `user_id`: 用户ID (154个唯一值)
- `position`: 位置 (138,637个唯一值)
- `supervoxel_id`: 超体素ID (137,667个唯一值)
- `label_id`: 标签ID (所有行唯一)
- `date_created`: 创建日期 (52,156个唯一值)
- `user_name`: 用户名 (151个唯一值)
- `user_affiliation`: 用户机构 (58个唯一值)

**说明**: 每日更新，包含贡献者信息，原始未处理标签

**解压命令**:
```bash
gunzip labels.csv.gz
```

---

### 9. Community Labels (Refined) (社区标签-精炼)
**文件**: `processed_labels.csv.gz`
**大小**: 1,018 KB

**内容**:
- 行数: 100,091 (+ header)
- 列数: 2

**列说明**:
- `root_id`: 细胞唯一ID
- `processed_labels`: 处理后的标签 (7,652个唯一值)

**说明**: 清理、去重、移除无信息或错误部分，Codex搜索使用

**解压命令**:
```bash
gunzip processed_labels.csv.gz
```

---

### 10. Connections (Filtered) (连接-过滤)
**文件**: `connections_princeton.csv.gz`
**大小**: 68 MB

**内容**:
- 行数: 5,342,446 (+ header)
- 列数: 5

**列说明**:
- `pre_root_id`: 突触前细胞ID (137,518个唯一值)
- `post_root_id`: 突触后细胞ID (130,183个唯一值)
- `neuropil`: 神经纤维区域 (79个唯一值)
- `syn_count`: 突触数量 (范围: 1-2,633)
- `nt_type`: 神经递质类型 (6个唯一值)

**说明**: 
- 过滤阈值: ≥5个突触
- 排除自突触
- 同一对细胞在多个区域可能有多行

**用途**: 构建连接矩阵，分析神经网络

**解压命令**:
```bash
gunzip connections_princeton.csv.gz
```

---

### 11. Connections (Unfiltered) (连接-未过滤)
**文件**: `connections_princeton_no_threshold.csv.gz`
**大小**: 277 MB

**内容**:
- 行数: 22,285,323 (+ header)
- 列数: 5

**列说明**: 同上

**说明**: 
- 无阈值过滤
- 包含所有连接
- 排除自突触
- 仅作为资源提供

**用途**: 完整连接信息，研究弱连接

**解压命令**:
```bash
gunzip connections_princeton_no_threshold.csv.gz
```

---

### 12. Connectivity Tags (连接标签)
**文件**: `connectivity_tags.csv.gz`
**大小**: 638 KB

**内容**:
- 行数: 134,437 (+ header)
- 列数: 2

**列说明**:
- `root_id`: 细胞唯一ID
- `connectivity_tag`: 连接特征标签 (28个唯一值，逗号分隔)

**说明**: 从网络分析得出的神经元连接特征描述符

**解压命令**:
```bash
gunzip connectivity_tags.csv.gz
```

---

### 13. Marked Neuron Coordinates (标记神经元坐标)
**文件**: `coordinates.csv.gz`
**大小**: 5,315 KB

**内容**:
- 行数: 238,909 (+ header)
- 列数: 3

**列说明**:
- `root_id`: 细胞ID (139,255个唯一值)
- `position`: 位置坐标(纳米) (225,588个唯一值)
- `supervoxel_id`: 超体素ID (221,912个唯一值)

**说明**: 
- 一个细胞可能有0个或多个坐标
- 坐标通常指向人工审查最有用的位置
- 不一定是细胞体/胞体位置

**解压命令**:
```bash
gunzip coordinates.csv.gz
```

---

### 14. Synapse Table (突触表)
**文件**: `fafb_v783_princeton_synapse_table.csv.gz`
**大小**: 2,695 MB (2.7 GB)

**内容**:
- 行数: 80,215,790 (+ header)
- 列数: 13

**列说明** (坐标单位: 纳米):
- `pre_x`, `pre_y`, `pre_z`: 突触前坐标
- `ctr_x`, `ctr_y`, `ctr_z`: 突触中心坐标
- `post_x`, `post_y`, `post_z`: 突触后坐标
- `size`: 检测到的体素数量
- `pre_root_id_720575940`: 突触前细胞ID (需加前缀)
- `post_root_id_720575940`: 突触后细胞ID (需加前缀)
- `neuropil`: 包含突触的脑区

**说明**: 
- 包含自突触预测
- ID需要加上列标题中的前缀
- 非常大的文件，需要大量存储空间

**解压命令**:
```bash
gunzip fafb_v783_princeton_synapse_table.csv.gz
```

---

### 15. Neuron Skeletons (神经元骨架)
**文件**: `sk_lod1_783_healed.zip`
**大小**: 13 GB

**内容**: 详细骨架文件 (SWC格式)

**说明**: 
- 单位: 微米
- 包含所有神经元的3D骨架
- 用于形态学分析和可视化

**解压命令**:
```bash
unzip sk_lod1_783_healed.zip
```

---

## 数据使用建议

### 对于本项目

#### 必需数据
1. **Connections (Filtered)**: 构建神经网络连接
2. **Neurotransmitter Predictions**: 确定兴奋/抑制性质
3. **Visual Neuron Annotations**: 筛选视觉相关神经元
4. **Classification**: 理解神经元层级组织

#### 可选数据
1. **Cell Types**: 更详细的类型信息
2. **Connectivity Tags**: 网络特征分析
3. **Synapse Table**: 详细突触信息(如需要)
4. **Skeletons**: 形态学分析(如需要)

#### 不太需要的数据
1. Community Labels: 主要用于标注和搜索
2. Coordinates: 主要用于定位
3. Cell Size: 除非研究形态学

---

## 数据处理流程建议

### Step 1: 下载核心数据
```bash
# 下载并解压核心文件
wget [FlyWire_URL]/connections_princeton.csv.gz
wget [FlyWire_URL]/neurons.csv.gz
wget [FlyWire_URL]/visual_neuron_types.csv.gz
wget [FlyWire_URL]/classification.csv.gz

gunzip *.csv.gz
```

### Step 2: 加载和探索
```python
import pandas as pd

# 加载数据
connections = pd.read_csv('connections_princeton.csv')
neurons = pd.read_csv('neurons.csv')
visual = pd.read_csv('visual_neuron_types.csv')
classification = pd.read_csv('classification.csv')

# 基本统计
print(connections.shape)
print(neurons['nt_type'].value_counts())
print(visual['type'].value_counts())
```

### Step 3: 筛选相关神经元
```python
# 筛选视觉和运动相关神经元
visual_neurons = visual['root_id'].unique()
motor_neurons = classification[
    classification['class'].str.contains('motor', na=False)
]['root_id'].unique()

# 提取相关连接
relevant_connections = connections[
    (connections['pre_root_id'].isin(visual_neurons)) |
    (connections['post_root_id'].isin(motor_neurons))
]
```

### Step 4: 构建连接矩阵
```python
import numpy as np
from scipy.sparse import csr_matrix

# 创建ID到索引的映射
all_neurons = np.unique(np.concatenate([
    relevant_connections['pre_root_id'].values,
    relevant_connections['post_root_id'].values
]))
neuron_to_idx = {nid: idx for idx, nid in enumerate(all_neurons)}

# 构建稀疏连接矩阵
row = [neuron_to_idx[nid] for nid in relevant_connections['pre_root_id']]
col = [neuron_to_idx[nid] for nid in relevant_connections['post_root_id']]
data = relevant_connections['syn_count'].values

connectivity_matrix = csr_matrix(
    (data, (row, col)), 
    shape=(len(all_neurons), len(all_neurons))
)
```

---

## 注意事项

### 数据版本
- 当前数据与2024年10月发布的FlyWire论文数据可能不同
- 数据持续更新和策划
- 如需原始论文数据，参考Zenodo和GitHub链接

### 引用要求
- 使用数据必须遵循FlyWire引用指南
- 需要在出版物中适当引用
- 联系邮箱: flywire@princeton.edu

### 存储需求
- 完整数据集约20+ GB
- 建议只下载需要的文件
- 使用压缩格式存储

### 处理建议
- 大文件使用分块读取
- 考虑使用数据库(如SQLite)
- 稀疏矩阵存储连接信息
- 使用HDF5格式存储处理后的数据

---

**文档创建**: 2026年3月16日
**数据版本**: FAFB v783 (更新至2025年6月23日)
**参考**: https://codex.flywire.ai/
