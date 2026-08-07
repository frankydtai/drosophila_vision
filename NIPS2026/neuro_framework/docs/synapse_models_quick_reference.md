# Synapse Models - Quick Reference Card

## 快速选择指南

```
需要快速原型？          → Simple
需要速率模型？          → TanhRate
需要电导模型？          → TanhConductance
需要生物物理建模？      → Ionotropic
```

---

## 一行代码切换模型

```python
from neuro_framework.connectome import ConnectomeLoader
from neuro_framework.models import ConnectomeNetwork

loader = ConnectomeLoader.from_optic_lobe(cell_types=['T4a', 'T4b'])

# Simple (默认)
net = ConnectomeNetwork.from_loader(loader, synapse_model='simple')

# TanhRate
net = ConnectomeNetwork.from_loader(loader, synapse_model='tanh_rate')

# TanhConductance
net = ConnectomeNetwork.from_loader(loader, synapse_model='tanh_conductance')

# Ionotropic
net = ConnectomeNetwork.from_loader(loader, synapse_model='ionotropic')
```

---

## 模型对比表

| 特性 | Simple | TanhRate | TanhConductance | Ionotropic |
|------|--------|----------|-----------------|------------|
| **状态变量** | 0 | 0 | 0 | 1 (s) |
| **参数/边** | 1 | 3 | 4 | 5 |
| **生物真实性** | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **计算速度** | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ |
| **可训练** | ✅ | ✅ | ✅ | ✅ |

---

## 数学公式

### Simple
```
I_syn = Σ w_ji * a_j
```

### TanhRate
```
I = -gS * tanh((V_pre - x_offset) * slope)
```

### TanhConductance
```
I = tanh((V_pre - x_offset) * slope) * gS * (V_post - e_syn)
```

### Ionotropic
```
s_inf = 1 / (1 + exp((v_th - V_pre) / delta))
tau_s = (1 - s_inf) / k_minus
ds/dt = (s_inf - s) / tau_s
I = gS * s * (V_post - e_syn)
```

---

## 参数说明

### TanhRate
- `gS`: 最大突触电导 (默认: 1e-4)
- `x_offset`: 电压偏移 (默认: -70.0 mV)
- `slope`: tanh 斜率 (默认: 1.0)

### TanhConductance
- `gS`: 最大突触电导 (默认: 1e-4 uS)
- `e_syn`: 反转电位 (默认: 0.0 mV)
- `x_offset`: 电压偏移 (默认: -70.0 mV)
- `slope`: tanh 斜率 (默认: 1.0)

### Ionotropic
- `gS`: 最大电导 (默认: 1e-4 uS)
- `e_syn`: 反转电位 (默认: 0.0 mV)
- `k_minus`: 解离速率常数 (默认: 0.025 s^-1)
- `v_th`: 电压阈值 (默认: -35.0 mV)
- `delta`: 电压敏感性 (默认: 10.0 mV)

---

## 训练示例

```python
import torch

# 创建网络
net = ConnectomeNetwork.from_loader(
    loader,
    dynamics='voltage',
    synapse_model='ionotropic',
    learn_weights=True,
    learn_synapse_params=True  # 学习突触参数
)

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    
    # 前向传播
    activity = net(stimulus, dt=1.0)
    
    # 损失
    loss = ((activity - target) ** 2).mean()
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

---

## 性能参考

**测试配置**: 227 neurons, 175 edges

| 模型 | 参数总数 | 相对速度 | 内存 |
|------|---------|---------|------|
| Simple | 629 | 1.0x | 最低 |
| TanhRate | 1,154 | 1.2x | 低 |
| TanhConductance | 1,329 | 1.3x | 低 |
| Ionotropic | 1,504 | 1.5x | 中 |

---

## 常见问题

### Q: 如何选择突触模型？
**A**: 
- 快速实验 → Simple
- 需要平滑激活 → TanhRate
- 需要驱动力 → TanhConductance
- 需要突触动力学 → Ionotropic

### Q: 可以混合使用不同模型吗？
**A**: 目前每个网络只能使用一种突触模型。未来可能支持混合模型。

### Q: 突触参数可以学习吗？
**A**: 是的！设置 `learn_synapse_params=True` 即可。

### Q: 与 Jaxley 兼容吗？
**A**: 我们的实现受 Jaxley 启发，但是 PyTorch 版本。数学公式相同。

---

## 查看完整演示

```bash
cd neuro_framework/notebooks
jupyter notebook 05_synapse_models.ipynb
```

---

## 参考资料

- **文档**: `docs/synapse_models_integration.md`
- **Jaxley**: https://jaxley.readthedocs.io/
- **文献**: Abbott & Marder (1998), "Modeling Small Networks"

---

**版本**: 0.2.0  
**更新**: 2026-04-04
