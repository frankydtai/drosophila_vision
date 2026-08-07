# 📁 文件整理方案

## 当前文件状态

### 问题
- 文件命名不统一（大小写混乱）
- 文档冗余（多个总结文件）
- 部分代码未使用

---

## 整理后的文件结构

### 📂 核心代码文件（保留并重命名）

```
code/
├── 01_load_flywire_data.py          # 数据加载器（重命名自 flywire_real_data_loader.py）
├── 02_convert_to_flyvis.py          # 格式转换器（重命名自 flywire_to_flyvis_converter.py）
├── 03_verify_connectome.py          # 验证脚本（重命名自 verify_flywire_connectome.py）
└── flyvis/connectome/
    ├── flywire_connectome.py        # FlyWire 连接组类
    └── flywire_v1.0.json            # 连接组数据（重命名自 flywire_real_v1.0.json）
```

### 📖 文档文件（整理并重命名）

```
docs/
├── 00_START_HERE.md                 # 快速入口（保留）
├── 01_README.md                     # 项目主页（重命名自 readme.md）
├── 02_QUICKSTART.md                 # 快速开始（保留）
├── 03_TECHNICAL_DETAILS.md          # 技术细节（新建）
├── 04_JAXLEY_ANALYSIS.md            # Jaxley 分析（重命名自 FLYWIRE_JAXLEY_PLAN.md）
└── 05_COMPLETION_REPORT.md          # 完成报告（保留）
```

### 🗑️ 删除的文件

```
删除原因：冗余或未使用

❌ flywire_data_loader.py            # 通用框架，实际未使用
❌ train_flywire_model.py            # 只是框架，未实现
❌ test_flywire_integration.py       # 功能与 verify 重复
❌ SUMMARY.md                        # 早期总结，已过时
❌ FILE_LIST.md                      # 冗余
❌ FINAL_REPORT.md                   # 与 COMPLETION_REPORT 重复
❌ PROJECT_SUMMARY.md                # 与 COMPLETION_REPORT 重复
❌ CHECKLIST.md                      # 已完成，不需要
❌ PROJECT_COMPLETE.txt              # 冗余
❌ flywire_integration_plan.md       # 早期计划，已完成
❌ README_FLYWIRE.md                 # 内容合并到主 README
❌ CHANGELOG.md                      # Flyvis 原始文件，不相关
```

---

## 重命名映射表

### 代码文件

| 原文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `flywire_real_data_loader.py` | `01_load_flywire_data.py` | 统一命名，添加序号 |
| `flywire_to_flyvis_converter.py` | `02_convert_to_flyvis.py` | 统一命名，添加序号 |
| `verify_flywire_connectome.py` | `03_verify_connectome.py` | 统一命名，添加序号 |
| `flyvis/connectome/flywire_real_v1.0.json` | `flyvis/connectome/flywire_v1.0.json` | 简化名称 |

### 文档文件

| 原文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `readme.md` | `01_README.md` | 统一大写，添加序号 |
| `START_HERE.md` | `00_START_HERE.md` | 添加序号，保持入口 |
| `QUICKSTART.md` | `02_QUICKSTART.md` | 添加序号 |
| `TECHNICAL_DETAILS.md` | `03_TECHNICAL_DETAILS.md` | 新建，添加序号 |
| `FLYWIRE_JAXLEY_PLAN.md` | `04_JAXLEY_ANALYSIS.md` | 简化名称，添加序号 |
| `COMPLETION_REPORT.md` | `05_COMPLETION_REPORT.md` | 添加序号 |

---

## 最终文件清单

### 代码文件（4 个）

1. **01_load_flywire_data.py** (342 行)
   - 加载 FlyWire v783 数据
   - 过滤视觉系统神经元
   - 计算连接矩阵和空间偏移

2. **02_convert_to_flyvis.py** (311 行)
   - 转换为 Flyvis JSON 格式
   - 神经递质映射
   - 生成节点和边定义

3. **03_verify_connectome.py** (292 行)
   - 5 个验证测试
   - JSON 格式验证
   - 连接组创建测试

4. **flyvis/connectome/flywire_connectome.py** (351 行)
   - ConnectomeFromFlyWire 类
   - 完全兼容 Flyvis 接口
   - 自动注册机制

### 数据文件（1 个）

5. **flyvis/connectome/flywire_v1.0.json** (542 KB)
   - 146 种细胞类型
   - 2,071 个连接
   - 完整的空间偏移信息

### 文档文件（6 个）

6. **00_START_HERE.md**
   - 项目快速入口
   - 最简洁的说明

7. **01_README.md**
   - 项目主页
   - 完整的项目介绍

8. **02_QUICKSTART.md**
   - 快速开始指南
   - 安装和使用步骤

9. **03_TECHNICAL_DETAILS.md**
   - 技术细节说明
   - 数据流程详解
   - 坐标系统说明

10. **04_JAXLEY_ANALYSIS.md**
    - Jaxley 库分析
    - 集成方案建议

11. **05_COMPLETION_REPORT.md**
    - 项目完成报告
    - 详细的统计数据

---

## 执行整理的命令

```bash
cd /Users/lengyuner/Desktop/NIPS2026/flyvis

# 1. 重命名代码文件
mv flywire_real_data_loader.py 01_load_flywire_data.py
mv flywire_to_flyvis_converter.py 02_convert_to_flyvis.py
mv verify_flywire_connectome.py 03_verify_connectome.py

# 2. 重命名数据文件
mv flyvis/connectome/flywire_real_v1.0.json flyvis/connectome/flywire_v1.0.json

# 3. 重命名文档文件
mv readme.md 01_README.md
mv START_HERE.md 00_START_HERE.md
mv QUICKSTART.md 02_QUICKSTART.md
mv TECHNICAL_DETAILS.md 03_TECHNICAL_DETAILS.md
mv FLYWIRE_JAXLEY_PLAN.md 04_JAXLEY_ANALYSIS.md
mv COMPLETION_REPORT.md 05_COMPLETION_REPORT.md

# 4. 删除冗余文件
rm -f flywire_data_loader.py
rm -f train_flywire_model.py
rm -f test_flywire_integration.py
rm -f SUMMARY.md
rm -f FILE_LIST.md
rm -f FINAL_REPORT.md
rm -f PROJECT_SUMMARY.md
rm -f CHECKLIST.md
rm -f PROJECT_COMPLETE.txt
rm -f flywire_integration_plan.md
rm -f README_FLYWIRE.md

# 5. 验证结果
echo "=== 代码文件 ==="
ls -1 0*.py
echo ""
echo "=== 文档文件 ==="
ls -1 0*.md
echo ""
echo "=== 连接组文件 ==="
ls -1 flyvis/connectome/*.json
```

---

## 整理后的目录结构

```
flyvis/
├── 00_START_HERE.md              ⭐ 快速入口
├── 01_README.md                  📖 项目主页
├── 02_QUICKSTART.md              🚀 快速开始
├── 03_TECHNICAL_DETAILS.md       🔬 技术细节
├── 04_JAXLEY_ANALYSIS.md         🧠 Jaxley 分析
├── 05_COMPLETION_REPORT.md       📊 完成报告
│
├── 01_load_flywire_data.py       💾 数据加载器
├── 02_convert_to_flyvis.py       🔄 格式转换器
├── 03_verify_connectome.py       ✅ 验证脚本
│
└── flyvis/
    └── connectome/
        ├── flywire_connectome.py  🧬 连接组类
        └── flywire_v1.0.json      📄 连接组数据
```

---

## 文件命名规范

### 规则

1. **序号前缀**: 00-99，表示阅读/使用顺序
2. **全大写**: Markdown 文件使用全大写（README.md 除外）
3. **下划线分隔**: Python 文件使用下划线
4. **描述性名称**: 文件名清楚表明内容

### 示例

✅ **好的命名**:
- `00_START_HERE.md` - 清晰的入口
- `01_load_flywire_data.py` - 描述性强
- `03_TECHNICAL_DETAILS.md` - 内容明确

❌ **不好的命名**:
- `flywire_real_data_loader.py` - 太长
- `readme.md` - 大小写不统一
- `SUMMARY.md` - 不够具体

---

## 更新代码中的引用

### 需要更新的文件

1. **02_convert_to_flyvis.py**
```python
# 修改导入
from 01_load_flywire_data import FlyWireRealDataLoader
# 改为
from load_flywire_data_01 import FlyWireRealDataLoader
# 或者使用模块导入
import sys
sys.path.append('.')
from load_flywire_data_01 import FlyWireRealDataLoader
```

2. **03_verify_connectome.py**
```python
# 修改路径
flywire_data_path="flyvis/connectome/flywire_v1.0.json"
```

3. **flyvis/connectome/flywire_connectome.py**
```python
# 无需修改，路径是参数传入的
```

---

## 总结

### 整理前
- 20 个文件（6 代码 + 14 文档）
- 命名混乱
- 大量冗余

### 整理后
- 11 个文件（4 代码 + 1 数据 + 6 文档）
- 命名统一
- 结构清晰

### 优势
- ✅ 易于导航（序号排序）
- ✅ 命名一致（全大写 MD，下划线 PY）
- ✅ 无冗余（删除 9 个重复文件）
- ✅ 逻辑清晰（按使用顺序编号）
