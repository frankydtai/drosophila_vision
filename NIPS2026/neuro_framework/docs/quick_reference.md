# 快速参考手册

**快速查找常用代码和命令**

---

## 🚀 快速开始

### 1. 加载所有神经元（无过滤）

```python
from neuro_framework.connectome.loader import ConnectomeLoader

# BANC 全脑
loader = ConnectomeLoader.from_banc(min_syn_count=5)
nodes, edges = loader.load()
# 115,151 neurons, 1,373,303 edges

# FAFB 全脑
loader = ConnectomeLoader.from_fafb(
    data_dir="/Users/lengyuner/Desktop/data/flywire/Jun2025",
    min_syn_count=5
)
nodes, edges = loader.load()
# 138,043 neurons, 2,699,071 edges
```

### 2. 按 Super-Class 过滤（推荐）

```python
# 仅视觉系统
loader = ConnectomeLoader.from_fafb(
    data_dir="/Users/lengyuner/Desktop/data/flywire/Jun2025",
    super_classes=['optic'],
    min_syn_count=5
)
nodes, edges = loader.load()
# ~77,865 neurons
```

### 3. 按细胞类型过滤

```python
# T4/T5 → LC 通路
loader = ConnectomeLoader.from_optic_lobe(
    cell_types=['T4a', 'T4b', 'T4c', 'T4d',
                'T5a', 'T5b', 'T5c', 'T5d',
                'LC4', 'LC6', 'LC9', 'LC10'],
    min_syn_count=5
)
nodes, edges = loader.load()
```

### 4. 构建网络

```python
from neuro_framework.models.network_torch import ConnectomeNetwork

# Voltage 模型（DMN 风格）
net = ConnectomeNetwork.from_loader(loader, dynamics='voltage')

# LIF 模型（脉冲神经网络）
net = ConnectomeNetwork.from_loader(loader, dynamics='lif')

# HH 模型（Hodgkin-Huxley）
net = ConnectomeNetwork.from_loader(loader, dynamics='hh')

print(f"{net.n_nodes} nodes, {net.n_edges} edges, {net.n_parameters():,} params")
```

### 5. 前向传播

```python
import torch

B, T = 2, 100  # batch_size, timesteps
x = torch.zeros(B, T, net.n_nodes)
x[:, :, :10] = torch.randn(B, T, 10) * 0.1  # 输入到前 10 个神经元

# 前向传播
activity = net(x, dt=1.0)  # (B, T, n_nodes)
print(f"Activity shape: {activity.shape}")
```

### 6. 训练

```python
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(100):
    # 前向传播
    activity = net(x, dt=1.0)
    
    # 计算损失
    target = torch.randn_like(activity) * 0.1
    loss = ((activity - target) ** 2).mean()
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 📊 数据集统计

| 数据集 | 神经元数 | 边数 | 细胞类型数 | Super-Class 数 |
|--------|---------|------|-----------|---------------|
| BANC | 115,151 | 1,373,303 | 11,193 | 14 |
| FAFB | 138,043 | 2,699,071 | 8,772 | 10 |
| Optic Lobe | 925 | 7,302 | 25 | 5 |
| FlyVis | ~700 | ~5,000 | ~50 | - |

---

## 🎯 常用过滤器

### FAFB Super-Classes

```python
super_classes = [
    'optic',              # 77,865 neurons (视觉)
    'central',            # 32,381 neurons (中枢)
    'sensory',            # 16,946 neurons (感觉)
    'visual_projection',  # 7,684 neurons (视投射)
    'ascending',          # 1,750 neurons (上行)
    'descending',         # 1,305 neurons (下行)
    'motor',              # 110 neurons (运动)
    'endocrine',          # 80 neurons (内分泌)
]
```

### 常用细胞类型

```python
# 视觉输入
cell_types = ['L1', 'L2', 'L3', 'L4', 'L5']

# Medulla 中间神经元
cell_types = ['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20']

# T4/T5 方向选择性神经元
cell_types = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']

# Lobula Columnar (LC) 神经元
cell_types = ['LC4', 'LC6', 'LC9', 'LC10', 'LC11', 'LC15', 'LC16', 'LC17']

# Centrifugal (C) 神经元
cell_types = ['C2', 'C3']
```

---

## 💾 内存估算

### 公式

```python
# 参数内存
param_memory = (n_edges + 2 * n_nodes) * 4 bytes

# 激活内存
activation_memory = n_nodes * T * batch_size * 4 bytes

# 训练总内存（包括梯度和优化器）
total_memory = 4 * param_memory + activation_memory
```

### 快速估算表

| 神经元数 | 边数 | 参数内存 | 激活内存 (T=100) | 训练内存 |
|---------|------|---------|-----------------|---------|
| 1,000 | 5,000 | ~28 KB | ~400 KB | ~500 MB |
| 10,000 | 50,000 | ~280 KB | ~4 MB | ~2 GB |
| 50,000 | 500,000 | ~2.4 MB | ~20 MB | ~8 GB |
| 100,000 | 2,000,000 | ~8.4 MB | ~40 MB | ~32 GB |
| 138,000 | 2,700,000 | ~11.3 MB | ~55 MB | ~48 GB |

---

## 🧪 测试命令

### 运行单元测试

```bash
cd /Users/lengyuner/Desktop/NIPS2026
/Users/lengyuner/anaconda3/bin/python3 -m pytest neuro_framework/tests/test_loader.py -v
```

### 快速验证

```bash
cd /Users/lengyuner/Desktop/NIPS2026
/Users/lengyuner/anaconda3/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from neuro_framework.connectome.loader import ConnectomeLoader

# 测试 BANC
loader = ConnectomeLoader.from_banc(min_syn_count=5)
nodes, edges = loader.load()
print(f'✓ BANC: {len(nodes):,} neurons, {len(edges):,} edges')

# 测试 Optic Lobe
loader = ConnectomeLoader.from_optic_lobe()
nodes, edges = loader.load()
print(f'✓ Optic Lobe: {len(nodes):,} neurons, {len(edges):,} edges')
"
```

---

## 📁 文件路径

### 数据路径

```python
# BANC
data_dir = "/Users/lengyuner/Desktop/NIPS2026/BANC_dataset/data"

# FAFB
data_dir = "/Users/lengyuner/Desktop/data/flywire/Jun2025"

# Optic Lobe
data_dir = "/Users/lengyuner/Desktop/NIPS2026/Jaxley_notebook/jaxley_tutorial-sjcabs/tutorial"

# FlyVis
data_dir = "/Users/lengyuner/Desktop/NIPS2026/flyvis/data/connectome/ConnectomeFromAvgFilters_0000"
```

### 输出路径

```python
# 日志
log_dir = "/Users/lengyuner/Desktop/NIPS2026/neuro_framework/logs"

# 笔记本
notebook_dir = "/Users/lengyuner/Desktop/NIPS2026/neuro_framework/notebooks"

# 文档
docs_dir = "/Users/lengyuner/Desktop/NIPS2026/neuro_framework/docs"
```

---

## 🔧 常见问题

### Q1: 如何加载所有神经元？

**A**: 不指定任何过滤器，或显式设置为 `None`：

```python
loader = ConnectomeLoader.from_banc()  # 默认加载全部
# 或
loader = ConnectomeLoader.from_banc(
    cell_types=None,
    super_classes=None,
    neuropils=None,
    sides=None
)
```

### Q2: 如何减少内存使用？

**A**: 使用以下策略：

1. 增加 `min_syn_count`（过滤弱连接）
2. 使用 `super_classes` 过滤（而不是加载全部）
3. 减少 batch_size 和 timesteps
4. 使用 `torch.no_grad()` 进行推理

```python
# 示例：减少内存
loader = ConnectomeLoader.from_fafb(
    data_dir="/path/to/fafb",
    super_classes=['optic'],  # 仅视觉系统
    min_syn_count=10,         # 仅强连接
)
```

### Q3: 如何加速训练？

**A**: 

1. 使用 GPU：`net = net.to('cuda')`
2. 使用混合精度训练：`torch.cuda.amp`
3. 减少网络规模（使用过滤器）
4. 使用更小的 timesteps

```python
# GPU 加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
x = x.to(device)
```

### Q4: LIF 模型没有输出脉冲？

**A**: 增加输入电流强度：

```python
# 对于 LIF 模型，需要更强的输入
x = torch.zeros(B, T, net.n_nodes)
x[:, :, :n_input] = torch.abs(torch.randn(B, T, n_input)) * 20.0  # 强输入
```

---

## 📚 相关文档

- [全神经元建模支持](all_neurons_support.md) — 详细指南
- [架构文档](architecture.md) — 项目结构
- [实现总结](implementation_summary.md) — 技术细节
- [TODO 列表](todo.md) — 开发路线图

---

**最后更新**: 2026-04-04  
**版本**: v1.0
