# 04_network_activation.ipynb 详细说明

## 回答您的问题

### 1. 现在使用的是什么刺激？

**当前状态：使用模拟数据，没有真实视觉刺激**

原因：
- PyTorch 未安装，Flyvis 框架无法导入
- 因此无法运行真实的神经网络模型
- 使用统计模拟生成激活数据

**模拟方法：**
```python
# 输入神经元：高激活（模拟光刺激）
for neuron in input_types:  # R1-6, R7, R8
    activations[neuron] = np.random.uniform(0.7, 1.0, 100)

# 中间神经元：根据层级衰减
layer = node_layers.get(neuron, 2)
decay = 0.9 ** layer
activations[neuron] = np.random.beta(2, 3, 100) * decay

# 输出神经元：方向选择性（Beta 分布）
activations[neuron] = np.random.beta(2, 5, 100) * 0.8
```

**如果使用真实 Flyvis 模型，应该使用的刺激：**
- **Sintel 数据集**：自然场景视频（来自动画电影）
- **运动模式**：平移、旋转、扩张等
- **刺激参数**：
  - 视频帧序列
  - 光流场
  - 对比度变化
  - 时间动态（通常 100-200 帧）

---

### 2. 记录的具体信息

**当前记录的数据：**

#### A. 激活统计（activation_stats.csv）
每个神经元的统计量：
- `neuron`: 神经元名称
- `mean`: 平均激活值（0-1）
- `std`: 标准差
- `max`: 最大激活值
- `min`: 最小激活值
- `layer`: 所在层级（0-4）
- `is_input`: 是否为输入神经元
- `is_output`: 是否为输出神经元

示例：
```
neuron,mean,std,max,min,layer,is_input,is_output
R8,0.855,0.088,0.998,0.701,0,True,False
R7,0.849,0.087,0.995,0.702,0,True,False
T4a,0.238,0.122,0.512,0.021,2,False,True
```

#### B. 时间序列激活（activations_sampled.json）
每个神经元在 100 个时间步的激活值（每 10 步采样一次）：
```json
{
  "R8": [0.855, 0.862, 0.841, ...],
  "T4a": [0.238, 0.245, 0.221, ...],
  ...
}
```

#### C. 层级信息（来自 03_network_hierarchy.ipynb）
- `neuron_layers.csv`: 每个神经元的层级
- `layer_info.json`: 完整的层级结构

**如果使用真实模型，还应该记录：**
- 膜电位（membrane potential）
- 突触电流（synaptic currents）
- 方向选择性指数（direction selectivity index）
- 时空感受野（spatiotemporal receptive fields）
- 响应延迟（response latency）

---

### 3. 网络里有多少 neuron？

**总数：146 个细胞类型**

**按层级分布：**
```
第 0 层:  3 个神经元（输入）
  - R1-6, R7, R8

第 1 层: 15 个神经元
  - L2, L4, Lai, T1, Tm5a, Tm5b, Tm5d, Dm8a, Dm8b, Dm11, 等

第 2 层: 81 个神经元（最多）
  - 包含 T4a, T4b, T4c, T4d
  - 大量 Tm 和 Mi 类型神经元

第 3 层: 44 个神经元
  - 包含 T5a, T5b, T5c, T5d
  - 各种中间神经元

第 4 层:  2 个神经元
  - Sm38, Sm42

未到达: 1 个神经元
```

**按功能分类：**
- **输入神经元（光感受器）**: 3 个
  - R1-6: 外周光感受器（6 种合并为 1 个类型）
  - R7: UV 敏感光感受器
  - R8: 蓝/绿敏感光感受器

- **输出神经元（运动检测）**: 8 个
  - T4a, T4b, T4c, T4d: ON 通路（检测亮度增加）
  - T5a, T5b, T5c, T5d: OFF 通路（检测亮度减少）
  - 每个对应 4 个基本方向（前、后、上、下）

- **中间神经元**: 135 个
  - Lamina 神经元（L1-L5, Lai, etc.）
  - Medulla 神经元（Tm1-35, Mi1-15, Dm1-20, etc.）
  - 其他类型（CT, LPi, Am, etc.）

**注意：**
- 这是 **细胞类型** 的数量，不是单个神经元
- 在真实的 Flyvis 模型中，每个类型会在六边形网格上复制多次
- 例如：如果网格是 15×15，则实际神经元数 = 146 × 225 = 32,850 个

---

### 4. 为什么使用 T4/T5 作为输出？

**生物学原因：**

#### A. T4/T5 是果蝇视觉系统的关键运动检测神经元
- **T4 细胞**：检测 **ON 边缘运动**（亮度增加）
  - 响应从暗到亮的变化
  - 4 个亚型对应 4 个方向

- **T5 细胞**：检测 **OFF 边缘运动**（亮度减少）
  - 响应从亮到暗的变化
  - 4 个亚型对应 4 个方向

#### B. 方向选择性
每个亚型对特定方向的运动最敏感：
- **T4a / T5a**: 向前运动（front）
- **T4b / T5b**: 向后运动（back）
- **T4c / T5c**: 向上运动（upward）
- **T4d / T5d**: 向下运动（downward）

#### C. 在视觉层级中的位置
```
光感受器 (R1-8)
    ↓
Lamina 神经元 (L1-5)
    ↓
Medulla 神经元 (Tm, Mi, Dm)
    ↓
T4 (ON) / T5 (OFF)  ← 输出层
    ↓
Lobula Plate (LPTC)
    ↓
运动行为
```

#### D. 计算模型的优势
- **明确的功能**：运动方向检测
- **可测量的输出**：方向选择性指数
- **生物学验证**：大量实验数据支持
- **行为相关**：直接影响飞行控制

#### E. 原始 Flyvis 论文的选择
原论文（Nature 2024）使用 T4/T5 作为输出是因为：
1. 它们是视觉运动处理的关键节点
2. 有丰富的电生理数据可以验证
3. 可以用光流任务（optic flow）训练模型
4. 方向选择性是可量化的性能指标

---

### 5. 还有什么其他问题？

#### A. 当前实现的局限性

**1. 使用模拟数据而非真实模型**
- **问题**：无法反映真实的神经动力学
- **解决**：安装 PyTorch 和 Flyvis 依赖
  ```bash
  pip install torch torchvision torchaudio
  ```

**2. 没有真实的视觉刺激**
- **问题**：激活模式是统计生成的，不是对刺激的响应
- **解决**：需要实现完整的 Flyvis 模型运行流程

**3. 缺少时间动力学**
- **问题**：当前只是静态激活值，没有真实的时间常数
- **解决**：使用 Flyvis 的动态神经元模型

**4. 没有训练过的权重**
- **问题**：连接权重是随机初始化的
- **解决**：需要在 Sintel 数据集上训练模型

#### B. 建议的改进方向

**1. 实现真实的 Flyvis 模型运行**
```python
# 创建网络
from flyvis import Network
network = Network(
    connectome="flywire_v1.0",
    dynamics="PPNeuron",  # 点过程神经元
    n_hexals=15  # 15×15 六边形网格
)

# 加载刺激
from flyvis.datasets.sintel import SequenceDataset
dataset = SequenceDataset(...)
stimulus = dataset[0]  # 获取一个视频序列

# 运行模型
responses = network(stimulus)
```

**2. 添加更多分析**
- 方向选择性指数（DSI）
- 时空感受野（STRF）
- 响应延迟分析
- 信噪比（SNR）

**3. 对比不同连接组**
- FlyWire vs 原始 FIB 连接组
- 性能差异
- 连接模式差异的影响

**4. 可视化增强**
- 3D 网络可视化
- 交互式激活图（Plotly）
- 动画显示时间演化
- 方向调谐曲线

#### C. 数据完整性问题

**1. 未到达的神经元**
- 有 1 个神经元在 BFS 中未被到达
- 可能是孤立节点或反向连接

**2. 层级深度**
- 只有 5 层，相对较浅
- 可能需要检查是否有遗漏的连接

**3. 连接数量**
- 2,071 条连接相对较少
- 原始 FlyWire 数据有更多连接
- 可能是过滤阈值（min_syn_count）太高

#### D. 性能验证问题

**需要验证的指标：**
1. **方向选择性**：T4/T5 是否对特定方向敏感？
2. **ON/OFF 分离**：T4 和 T5 是否正确分离？
3. **时间动力学**：响应延迟是否合理？
4. **空间整合**：感受野大小是否正确？

---

## 总结

### 当前状态
✅ 成功创建了 4 个可视化 notebooks
✅ 生成了 13 张图表和 4 个数据文件
✅ 实现了 BFS 层次结构分析
⚠️ 使用模拟数据（PyTorch 未安装）

### 下一步建议
1. **安装 PyTorch**：`pip install torch torchvision torchaudio`
2. **运行真实模型**：使用 Sintel 数据集
3. **训练网络**：优化连接权重
4. **验证性能**：测试方向选择性
5. **对比分析**：FlyWire vs FIB 连接组

### 关键数字
- **神经元类型**：146 个
- **连接**：2,071 条
- **层级**：5 层（0-4）
- **输入**：3 个光感受器
- **输出**：8 个运动检测神经元（T4/T5）
- **时间步**：100（模拟）
