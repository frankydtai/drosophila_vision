# 📋 项目文件清单

## 最终文件结构

```
flyvis/
├── 📖 文档文件（6 个）
│   ├── 00_START_HERE.md              ⭐ 快速入口（1.9 KB）
│   ├── 01_README.md                  📖 项目主页（7.9 KB）
│   ├── 02_QUICKSTART.md              🚀 快速开始（6.4 KB）
│   ├── 03_TECHNICAL_DETAILS.md       🔬 技术细节（新建）
│   ├── 04_JAXLEY_ANALYSIS.md         🧠 Jaxley 分析（6.5 KB）
│   ├── 05_COMPLETION_REPORT.md       📊 完成报告（8.2 KB）
│   ├── ANSWERS.md                    ❓ 核心问题解答（新建）
│   └── FILE_ORGANIZATION.md          📁 文件整理方案（新建）
│
├── 💻 代码文件（4 个）
│   ├── 01_load_flywire_data.py       💾 数据加载器（10 KB）
│   ├── 02_convert_to_flyvis.py       🔄 格式转换器（9.7 KB）
│   ├── 03_verify_connectome.py       ✅ 验证脚本（9.1 KB）
│   └── flyvis/connectome/
│       ├── flywire_connectome.py     🧬 连接组类（12 KB）
│       └── flywire_v1.0.json         📄 连接组数据（542 KB）
│
└── 🗑️ 已删除文件（11 个）
    ├── flywire_data_loader.py        ❌ 未使用的通用框架
    ├── train_flywire_model.py        ❌ 未实现的框架
    ├── test_flywire_integration.py   ❌ 与 verify 重复
    ├── SUMMARY.md                    ❌ 早期总结
    ├── FILE_LIST.md                  ❌ 冗余
    ├── FINAL_REPORT.md               ❌ 与 COMPLETION_REPORT 重复
    ├── PROJECT_SUMMARY.md            ❌ 与 COMPLETION_REPORT 重复
    ├── CHECKLIST.md                  ❌ 已完成
    ├── PROJECT_COMPLETE.txt          ❌ 冗余
    ├── flywire_integration_plan.md   ❌ 早期计划
    └── README_FLYWIRE.md             ❌ 内容已合并
```

---

## 文件说明

### 📖 文档文件（按阅读顺序）

#### 1. **00_START_HERE.md** ⭐
- **用途**: 项目快速入口
- **内容**: 最简洁的说明，告诉你从哪里开始
- **适合**: 第一次接触项目的人

#### 2. **01_README.md**
- **用途**: 项目主页
- **内容**: 完整的项目介绍、背景、目标
- **适合**: 想全面了解项目的人

#### 3. **02_QUICKSTART.md**
- **用途**: 快速开始指南
- **内容**: 安装步骤、使用示例、常见问题
- **适合**: 想快速上手的人

#### 4. **03_TECHNICAL_DETAILS.md** 🆕
- **用途**: 技术细节说明
- **内容**: 
  - 数据规模详解（为什么只用 146/741 种细胞类型）
  - 六边形坐标系统说明
  - 数据流程可视化
  - 如何扩展到更多数据
- **适合**: 想深入理解实现细节的人

#### 5. **04_JAXLEY_ANALYSIS.md**
- **用途**: Jaxley 库分析
- **内容**: Jaxley vs Flyvis 对比、集成方案建议
- **适合**: 考虑使用 Jaxley 的人

#### 6. **05_COMPLETION_REPORT.md**
- **用途**: 项目完成报告
- **内容**: 详细的统计数据、成就、下一步计划
- **适合**: 想了解项目完成情况的人

#### 7. **ANSWERS.md** 🆕
- **用途**: 核心问题解答
- **内容**: 
  - 怎么用 FlyWire 数据替代的？
  - 为什么只用了 146 种细胞类型？
  - FlyWire 真实位置怎么对应 hex 网格？
- **适合**: 想快速找到答案的人

#### 8. **FILE_ORGANIZATION.md** 🆕
- **用途**: 文件整理方案
- **内容**: 重命名映射表、删除文件列表、命名规范
- **适合**: 想了解文件结构的人

---

### 💻 代码文件（按执行顺序）

#### 1. **01_load_flywire_data.py**
```python
# 功能: 加载和过滤 FlyWire 数据
# 输入: /Users/lengyuner/Desktop/data/flywire/Jun2025/*.csv
# 输出: 过滤后的 DataFrame（神经元、连接、空间位置）

from load_flywire_data_01 import FlyWireRealDataLoader

loader = FlyWireRealDataLoader()
data = loader.filter_visual_system(
    subsystems=['Motion', 'Color', 'OFF', 'Photoreceptors']
)
# 结果: 25,732 个神经元，519,789 个连接
```

**关键方法**:
- `load_visual_neurons()`: 加载视觉神经元类型
- `load_column_assignments()`: 加载空间位置（p, q 坐标）
- `load_connections()`: 加载连接数据
- `filter_visual_system()`: 过滤视觉系统
- `compute_connectivity_matrix()`: 计算连接矩阵
- `compute_spatial_offsets()`: 计算空间偏移

#### 2. **02_convert_to_flyvis.py**
```python
# 功能: 转换为 Flyvis JSON 格式
# 输入: FlyWireRealDataLoader 的输出
# 输出: flyvis/connectome/flywire_v1.0.json

from convert_to_flyvis_02 import FlyWireToFlyvisRealConverter

converter = FlyWireToFlyvisRealConverter(extent=15)
result = converter.convert(
    output_path="flyvis/connectome/flywire_v1.0.json",
    subsystems=['Motion', 'Color', 'OFF', 'Photoreceptors'],
    min_syn_count=10
)
# 结果: 146 个节点，2,071 个边
```

**关键方法**:
- `_convert_nodes()`: 转换节点定义
- `_convert_edges()`: 转换边定义（包含神经递质映射）
- `_identify_input_units()`: 识别输入单元（R1-6, R7, R8）
- `_identify_output_units()`: 识别输出单元（T4a-d, T5a-d）

#### 3. **03_verify_connectome.py**
```python
# 功能: 验证生成的连接组
# 输入: flyvis/connectome/flywire_v1.0.json
# 输出: 5 个测试的结果

python 03_verify_connectome.py
```

**测试内容**:
1. JSON 文件格式验证
2. ConnectomeFromFlyWire 对象创建
3. ConnectomeView 创建
4. 节点和边数据验证
5. 统计信息验证

#### 4. **flyvis/connectome/flywire_connectome.py**
```python
# 功能: FlyWire 连接组类
# 用途: 在 Flyvis 框架中加载 FlyWire 数据

from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire

connectome = ConnectomeFromFlyWire(
    flywire_data_path="flyvis/connectome/flywire_v1.0.json",
    extent=15
)

# 完全兼容 Flyvis 接口
from flyvis.network import Network
network = Network(connectome=connectome, dynamics="PPNeuronIGRSynapses")
```

**关键特性**:
- 使用 `@register_connectome` 自动注册
- 继承自 `datamate.Directory`
- 完全兼容 Flyvis 的 `Connectome` 协议
- 支持细胞类型映射

---

### 📄 数据文件

#### **flyvis/connectome/flywire_v1.0.json** (542 KB)
```json
{
  "nodes": [
    {
      "name": "T4a",
      "pattern": ["stride", [1, 1]],
      "activation": "relu"
    },
    // ... 146 个节点
  ],
  "edges": [
    {
      "src": "Mi1",
      "tar": "T4a",
      "alpha": 1,
      "offsets": [
        [[0, 0], 150],
        [[1, 0], 80]
      ]
    },
    // ... 2,071 个边
  ],
  "input_units": ["R1-6", "R7", "R8"],
  "output_units": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
}
```

**统计信息**:
- 节点数: 146
- 边数: 2,071
- 输入单元: 3 种（光感受器）
- 输出单元: 8 种（运动检测神经元）
- 有空间偏移的边: 119 (5.7%)

---

## 命名规范

### ✅ 统一的命名规则

1. **序号前缀**: `00-99`，表示阅读/使用顺序
2. **文档文件**: 全大写 + 下划线（`00_START_HERE.md`）
3. **代码文件**: 全小写 + 下划线（`01_load_flywire_data.py`）
4. **描述性名称**: 文件名清楚表明内容

### 示例对比

| ❌ 旧命名 | ✅ 新命名 | 说明 |
|----------|----------|------|
| `readme.md` | `01_README.md` | 统一大写，添加序号 |
| `flywire_real_data_loader.py` | `01_load_flywire_data.py` | 简化名称，添加序号 |
| `flywire_to_flyvis_converter.py` | `02_convert_to_flyvis.py` | 简化名称，添加序号 |
| `verify_flywire_connectome.py` | `03_verify_connectome.py` | 简化名称，添加序号 |
| `flywire_real_v1.0.json` | `flywire_v1.0.json` | 简化名称 |
| `FLYWIRE_JAXLEY_PLAN.md` | `04_JAXLEY_ANALYSIS.md` | 更清晰的名称 |

---

## 代码引用更新

### ✅ 已更新的引用

#### 在 `02_convert_to_flyvis.py` 中:
```python
# 旧的导入
from flywire_real_data_loader import FlyWireRealDataLoader

# 新的导入（兼容两种方式）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from load_flywire_data_01 import FlyWireRealDataLoader
except ImportError:
    from flywire_real_data_loader import FlyWireRealDataLoader
```

#### 在 `03_verify_connectome.py` 中:
```python
# 旧的路径
json_path = "flyvis/connectome/flywire_real_v1.0.json"

# 新的路径
json_path = "flyvis/connectome/flywire_v1.0.json"
```

---

## 使用指南

### 快速开始

```bash
# 1. 查看快速入口
cat 00_START_HERE.md

# 2. 阅读项目主页
cat 01_README.md

# 3. 按照快速开始指南操作
cat 02_QUICKSTART.md

# 4. 如果有疑问，查看核心问题解答
cat ANSWERS.md

# 5. 如果需要深入了解，查看技术细节
cat 03_TECHNICAL_DETAILS.md
```

### 运行代码

```bash
# 激活环境
conda activate flywire_flyvis

# 验证集成（需要先安装 PyTorch）
python 03_verify_connectome.py

# 如果需要重新生成数据
python 02_convert_to_flyvis.py

# 在 Python 中使用
python
>>> from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
>>> connectome = ConnectomeFromFlyWire("flyvis/connectome/flywire_v1.0.json")
>>> print(connectome.get_statistics())
```

---

## 总结

### 整理前
- ❌ 20 个文件（6 代码 + 14 文档）
- ❌ 命名混乱（大小写不统一）
- ❌ 大量冗余文件

### 整理后
- ✅ 13 个文件（4 代码 + 1 数据 + 8 文档）
- ✅ 命名统一（序号 + 描述性名称）
- ✅ 结构清晰（按使用顺序编号）
- ✅ 无冗余（删除 11 个重复/未使用文件）

### 优势
- 📁 易于导航（序号排序）
- 🎯 逻辑清晰（按功能分类）
- 📖 文档完善（8 个文档覆盖所有方面）
- 💻 代码简洁（4 个核心文件）
