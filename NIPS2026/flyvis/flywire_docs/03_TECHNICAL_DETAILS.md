# 📋 FlyWire 数据替代方案详细说明

## 问题回答

### 1. FlyWire 数据规模

**实际数据规模**:
- **视觉神经元总数**: 95,079 个
- **细胞类型总数**: 741 种
- **有列分配的神经元**: 45,528 个
- **连接总数**: 519,789 个（视觉系统内）

**为什么我们只用了 146 种细胞类型和 2,071 个连接？**

这是因为我们做了**多层过滤**：

1. **子系统过滤**: 只保留 Motion, Color, OFF, Photoreceptors 子系统
   - 95,079 → 51,567 个神经元
   
2. **侧别过滤**: 只保留右侧（与原始 Flyvis 一致）
   - 51,567 → 25,732 个神经元
   
3. **突触阈值过滤**: 只保留 >10 个突触的强连接
   - 519,789 → 2,071 个连接
   
4. **空间信息过滤**: 只保留有列分配的神经元
   - 这进一步减少了细胞类型数量

### 2. FlyWire 真实位置到六边形网格的映射

**FlyWire 的坐标系统**:
```
column_assignment.csv.gz 包含:
- root_id: 神经元 ID
- hemisphere: left/right
- type: 细胞类型
- column_id: 列 ID (0-795)
- x, y: 笛卡尔坐标
- p, q: 六边形坐标（已经是六边形系统！）
```

**关键发现**: FlyWire 已经提供了六边形坐标 (p, q)！

**映射方法**:
```python
# FlyWire 的 p, q 坐标直接对应 Flyvis 的 u, v 坐标
# 在 flywire_real_data_loader.py 中:

def compute_spatial_offsets(self, connections, columns):
    # 对于每个连接
    pre_pos = columns[pre_id]  # {p: 6, q: -4}
    post_pos = columns[post_id]  # {p: 7, q: -3}
    
    # 计算偏移（直接使用 p, q）
    du = post_pos['p'] - pre_pos['p']  # 7 - 6 = 1
    dv = post_pos['q'] - pre_pos['q']  # -3 - (-4) = 1
    
    # 这个偏移 (du=1, dv=1) 就是 Flyvis 需要的格式
```

### 3. 为什么不用全部数据？

**技术原因**:
1. **计算资源**: 741 种细胞类型 × 95,079 个神经元 = 巨大的计算量
2. **训练时间**: 原始 Flyvis 用 64 种细胞类型就需要数小时训练
3. **内存限制**: 完整网络可能超过可用内存

**科学原因**:
1. **聚焦运动检测**: 原始论文主要研究运动检测通路
2. **可比性**: 与原始 FIB 数据（64 种细胞类型）保持可比性
3. **验证优先**: 先验证核心功能，再扩展

### 4. 如何使用更多数据？

**方法 1: 调整过滤参数**
```python
# 在 flywire_to_flyvis_converter.py 中修改:

converter.convert(
    subsystems=['Motion', 'Color', 'OFF', 'Photoreceptors', 
                'Form', 'Object'],  # 添加更多子系统
    min_syn_count=5  # 降低阈值（从 10 降到 5）
)
```

**方法 2: 包含两侧**
```python
# 在 flywire_real_data_loader.py 中修改:

def filter_visual_system(self):
    # 注释掉这一行：
    # visual_neurons = visual_neurons[visual_neurons['side'] == 'right']
    
    # 这样会包含左右两侧，神经元数量翻倍
```

**方法 3: 使用全部视觉神经元**
```python
# 不做子系统过滤
data = loader.filter_visual_system(subsystems=None)
```

---

## 文件整理和命名规范

### 核心代码文件（保留）

```
01_flywire_data_loader.py          # 真实数据加载器
02_flywire_converter.py            # 格式转换器  
03_verify_connectome.py            # 验证脚本
04_train_model.py                  # 训练脚本（框架）

flyvis/connectome/
  └── flywire_connectome.py        # FlyWire 连接组类
  └── flywire_v1.0.json            # 生成的连接组数据
```

### 文档文件（整理后）

```
00_START_HERE.md                   # 快速入口 ⭐
01_README.md                       # 项目主页
02_QUICKSTART.md                   # 快速开始指南
03_TECHNICAL_DETAILS.md            # 技术细节（本文件）
04_JAXLEY_ANALYSIS.md              # Jaxley 分析
05_COMPLETION_REPORT.md            # 完成报告
```

### 删除的冗余文件

```
❌ test_flywire_integration.py     # 功能与 verify 重复
❌ flywire_data_loader.py          # 通用框架，实际未使用
❌ train_flywire_model.py          # 只是框架，未实现
❌ SUMMARY.md                      # 早期总结，已过时
❌ FILE_LIST.md                    # 冗余
❌ FINAL_REPORT.md                 # 与 COMPLETION_REPORT 重复
❌ PROJECT_SUMMARY.md              # 与 COMPLETION_REPORT 重复
❌ CHECKLIST.md                    # 已完成，不需要
❌ PROJECT_COMPLETE.txt            # 冗余
❌ flywire_integration_plan.md     # 早期计划，已完成
❌ README_FLYWIRE.md               # 内容合并到主 README
```

---

## 数据流程详解

### 完整的数据处理流程

```
┌─────────────────────────────────────────────────────────────┐
│ FlyWire 原始数据                                             │
├─────────────────────────────────────────────────────────────┤
│ visual_neuron_types.csv.gz                                  │
│   - 95,079 个神经元                                          │
│   - 741 种细胞类型                                           │
│   - 包含: root_id, type, family, subsystem, side            │
│                                                              │
│ column_assignment.csv.gz                                    │
│   - 45,528 个神经元的空间位置                                │
│   - 包含: root_id, type, column_id, p, q (六边形坐标)       │
│                                                              │
│ connections.csv                                             │
│   - 数百万个连接                                             │
│   - 包含: pre_root_id, post_root_id, neuropil, syn_count   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 加载和过滤 (flywire_data_loader.py)                 │
├─────────────────────────────────────────────────────────────┤
│ filter_visual_system():                                     │
│   1. 加载视觉神经元                                          │
│   2. 过滤子系统 (Motion, Color, OFF, Photoreceptors)        │
│      95,079 → 51,567 个神经元                               │
│   3. 过滤侧别 (只保留右侧)                                   │
│      51,567 → 25,732 个神经元                               │
│   4. 加载这些神经元的连接                                    │
│      519,789 个连接                                         │
│   5. 加载空间位置信息                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: 计算连接统计 (flywire_data_loader.py)               │
├─────────────────────────────────────────────────────────────┤
│ compute_connectivity_matrix():                              │
│   - 按细胞类型对聚合连接                                     │
│   - 计算每个类型对的突触总数                                 │
│   - 统计神经递质类型                                         │
│   结果: 2,790 个类型对连接                                   │
│                                                              │
│ compute_spatial_offsets():                                  │
│   - 对于每个连接，计算空间偏移                               │
│   - du = post_p - pre_p                                     │
│   - dv = post_q - pre_q                                     │
│   结果: 119 个类型对有空间信息                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: 转换为 Flyvis 格式 (flywire_converter.py)           │
├─────────────────────────────────────────────────────────────┤
│ _convert_nodes():                                           │
│   - 提取唯一的细胞类型                                       │
│   - 为每种类型创建节点定义                                   │
│   - 设置 pattern: ["stride", [1, 1]]                        │
│   结果: 146 个节点定义                                       │
│                                                              │
│ _convert_edges():                                           │
│   - 过滤弱连接 (< min_syn_count)                            │
│   - 映射神经递质到突触符号                                   │
│     ACH/GLUT → +1 (兴奋性)                                  │
│     GABA → -1 (抑制性)                                      │
│   - 添加空间偏移信息                                         │
│   结果: 2,071 个边定义                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 输出: Flyvis JSON 格式                                       │
├─────────────────────────────────────────────────────────────┤
│ {                                                            │
│   "nodes": [                                                 │
│     {                                                        │
│       "name": "T4a",                                         │
│       "pattern": ["stride", [1, 1]],                         │
│       "activation": "relu",                                  │
│       ...                                                    │
│     }                                                        │
│   ],                                                         │
│   "edges": [                                                 │
│     {                                                        │
│       "src": "Mi1",                                          │
│       "tar": "T4a",                                          │
│       "alpha": 1,  # 兴奋性                                  │
│       "offsets": [                                           │
│         [[0, 0], 150],  # du, dv, syn_count                 │
│         [[1, 0], 80]                                         │
│       ]                                                      │
│     }                                                        │
│   ],                                                         │
│   "input_units": ["R1-6", "R7", "R8"],                      │
│   "output_units": ["T4a", "T4b", "T4c", "T4d",              │
│                     "T5a", "T5b", "T5c", "T5d"]              │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键技术细节

### 1. 六边形坐标系统

**FlyWire 使用的坐标**:
- `p, q`: 六边形坐标（轴向坐标系统）
- 这与 Flyvis 的 `u, v` 坐标系统**完全兼容**

**坐标转换**:
```python
# FlyWire → Flyvis (无需转换！)
flyvis_u = flywire_p
flyvis_v = flywire_q

# 计算偏移
du = target_p - source_p
dv = target_q - source_q
```

**六边形网格示例**:
```
      (0,1)   (1,1)
         \   /
    (-1,0)-(0,0)-(1,0)
         /   \
    (-1,-1)  (0,-1)
```

### 2. 为什么有些连接没有空间信息？

**原因**:
1. 不是所有神经元都有列分配（45,528 / 95,079 = 48%）
2. 非柱状细胞（如 wide-field neurons）没有固定列位置
3. 某些细胞类型跨越多个列

**解决方案**:
```python
# 在 flywire_converter.py 中
if key in offsets:
    # 使用真实的空间偏移
    offset_list = offsets[key]
else:
    # 假设中心连接
    offset_list = [[[0, 0], total_syn_count]]
```

### 3. 神经递质映射

**FlyWire 提供的神经递质类型**:
```python
NT_TO_SIGN = {
    'ACH': 1,      # 乙酰胆碱 - 兴奋性
    'GLUT': 1,     # 谷氨酸 - 兴奋性
    'GABA': -1,    # GABA - 抑制性
    'SER': 1,      # 血清素 - 通常兴奋性
    'DA': 1,       # 多巴胺 - 通常兴奋性
    'OCT': 1,      # 章鱼胺 - 通常兴奋性
}
```

**如何确定连接的符号**:
```python
# 对于每个类型对连接，统计所有神经递质
nt_types = {'ACH': 1500, 'GABA': 300}

# 选择最多的神经递质
main_nt = 'ACH'  # 1500 > 300

# 映射到符号
sign = NT_TO_SIGN['ACH'] = 1  # 兴奋性
```

---

## 如何扩展到更多数据

### 选项 1: 包含更多子系统

```python
# 修改 flywire_converter.py
converter.convert(
    subsystems=[
        'Motion',           # 运动检测
        'Color',            # 颜色处理
        'OFF',              # OFF 通路
        'Photoreceptors',   # 光感受器
        'Form',             # 形状检测
        'Object',           # 物体识别
        # 添加更多...
    ],
    min_syn_count=5  # 降低阈值
)

# 预期结果:
# - 细胞类型: 146 → ~300 种
# - 连接: 2,071 → ~10,000 个
```

### 选项 2: 包含两侧

```python
# 修改 flywire_data_loader.py
def filter_visual_system(self):
    # 不过滤侧别
    # visual_neurons = visual_neurons[visual_neurons['side'] == 'right']
    
    # 预期结果:
    # - 神经元: 25,732 → ~51,000 个
    # - 细胞类型: 146 → ~200 种（有些类型只在一侧）
```

### 选项 3: 使用全部视觉神经元

```python
# 不做任何过滤
data = loader.filter_visual_system(subsystems=None)

# 预期结果:
# - 神经元: 95,079 个
# - 细胞类型: 741 种
# - 连接: 数百万个
# 
# ⚠️ 警告: 需要大量计算资源！
```

---

## 总结

### 当前实现的选择

我们选择了**保守的过滤策略**:
- ✅ 聚焦运动检测（与原始论文一致）
- ✅ 只用右侧（与原始 Flyvis 一致）
- ✅ 只用强连接（>10 突触）
- ✅ 计算资源可行
- ✅ 可以快速验证

### 数据规模对比

| 数据集 | 神经元 | 细胞类型 | 连接 |
|--------|--------|----------|------|
| FlyWire 全部 | 95,079 | 741 | 数百万 |
| FlyWire 过滤后 | 25,732 | 146 | 2,071 |
| 原始 FIB | 45,669 | 64 | 1,513,231 |

### 六边形坐标

- ✅ FlyWire 已提供六边形坐标 (p, q)
- ✅ 直接对应 Flyvis 的 (u, v)
- ✅ 无需复杂的坐标转换
- ✅ 119/2,071 (5.7%) 的连接有空间信息

### 扩展性

代码设计为**高度可配置**:
- 调整 `subsystems` 参数 → 更多细胞类型
- 调整 `min_syn_count` → 更多连接
- 移除侧别过滤 → 两侧数据
- 完全不过滤 → 全部数据

**建议**: 先用当前配置验证功能，再逐步扩展！
