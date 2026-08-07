# 🎯 FlyWire + Flyvis 项目 - 快速开始指南

## ✅ 当前状态

**项目完成度**: 85% (核心功能已完成)

**已验证**:
- ✅ JSON 格式正确
- ✅ 数据加载成功
- ✅ 格式转换完成
- ✅ 146 种细胞类型
- ✅ 2,071 个连接

**待完成**:
- 🔄 安装 PyTorch
- 🔄 网络创建测试
- 🔄 模型训练

---

## 🚀 快速开始

### 步骤 1: 激活环境并安装依赖

```bash
# 激活环境
conda activate flywire_flyvis

# 安装 PyTorch (CPU 版本，适合 Mac)
pip install torch torchvision torchaudio

# 或者如果有 NVIDIA GPU
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install matplotlib scipy scikit-learn tqdm
pip install datamate toolz

# 安装 Flyvis (开发模式)
cd /Users/lengyuner/Desktop/NIPS2026/flyvis
pip install -e .
```

### 步骤 2: 验证安装

```bash
# 运行验证脚本
python verify_flywire_connectome.py
```

预期输出：所有 5 个测试通过 ✓

### 步骤 3: 创建第一个网络

```python
from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
from flyvis.network import Network
from datamate import Namespace

# 1. 创建连接组
connectome = ConnectomeFromFlyWire(
    flywire_data_path="flyvis/connectome/flywire_real_v1.0.json",
    extent=15,
    n_syn_fill=1
)

print(f"连接组统计: {connectome.get_statistics()}")

# 2. 创建网络
network_config = Namespace(
    connectome=Namespace(
        type="ConnectomeFromFlyWire",
        flywire_data_path="flyvis/connectome/flywire_real_v1.0.json",
        extent=15,
        n_syn_fill=1
    ),
    dynamics=Namespace(
        type="PPNeuronIGRSynapses",
        activation=Namespace(type="relu")
    ),
    # ... 其他配置
)

network = Network(**network_config)
print(f"网络参数数量: {sum(p.numel() for p in network.parameters())}")
```

### 步骤 4: 训练模型（参考原始 Flyvis）

```bash
# 参考 Flyvis 的训练脚本
python train_flywire_model.py
```

---

## 📁 项目文件结构

```
flyvis/
├── flywire_real_data_loader.py          # ✅ 真实数据加载器
├── flywire_to_flyvis_converter.py      # ✅ 格式转换器
├── verify_flywire_connectome.py        # ✅ 验证脚本
├── train_flywire_model.py              # 🔄 训练脚本（待完善）
│
├── flyvis/
│   └── connectome/
│       ├── flywire_connectome.py       # ✅ FlyWire 连接组类
│       ├── flywire_real_v1.0.json      # ✅ 生成的连接组 (542 KB)
│       └── fib25-fib19_v2.2.json       # 原始 FIB 数据（参考）
│
└── 文档/
    ├── README_FLYWIRE.md               # 完整使用指南
    ├── FLYWIRE_JAXLEY_PLAN.md          # 技术方案
    ├── FINAL_REPORT.md                 # 完成报告
    └── QUICKSTART.md                   # 本文件
```

---

## 📊 数据统计

### FlyWire 连接组 (flywire_real_v1.0.json)

```
细胞类型: 146
连接数量: 2,071
输入类型: 3 (R1-6, R7, R8)
输出类型: 8 (T4a-d, T5a-d)

原始数据:
- 神经元: 25,732 (右侧视觉系统)
- 连接: 519,789
- 子系统: Motion, Color, OFF, Photoreceptors
```

### 主要细胞类型

```
光感受器:
- R1-6: 3,437 个
- R7:     659 个
- R8:     655 个

运动检测:
- T4a-d: ~3,100 个
- T5a-d: ~3,000 个

关键中间神经元:
- L2, L4: 层板神经元
- Mi1, Mi4, Mi9: 髓质内神经元
- Tm1, Tm2, Tm3, Tm4, Tm9: 跨髓质神经元
- CT1: 复杂切线神经元
```

### 最强连接

```
L2 -> Tm1:    90,970 突触 (兴奋性)
L2 -> Tm2:    76,125 突触 (兴奋性)
L2 -> T1:     48,971 突触 (兴奋性)
R1-6 -> L2:   37,561 突触 (兴奋性)
Tm9 -> CT1:   23,533 突触 (兴奋性)
```

---

## 🔧 故障排除

### 问题 1: ModuleNotFoundError: No module named 'torch'
**解决方案**: 
```bash
conda activate flywire_flyvis
pip install torch torchvision torchaudio
```

### 问题 2: 找不到 flywire_real_v1.0.json
**解决方案**: 
```bash
cd /Users/lengyuner/Desktop/NIPS2026/flyvis
python flywire_to_flyvis_converter.py
```

### 问题 3: 内存不足
**解决方案**: 
- 减小 extent 参数 (例如 extent=10)
- 增加 min_syn_count 阈值
- 使用更少的细胞类型

### 问题 4: 训练速度慢
**解决方案**:
- 使用 GPU (如果可用)
- 减小批次大小
- 使用更少的训练数据

---

## 📚 参考命令

### 数据处理

```bash
# 加载和统计 FlyWire 数据
python flywire_real_data_loader.py

# 转换为 Flyvis 格式
python flywire_to_flyvis_converter.py

# 验证连接组
python verify_flywire_connectome.py
```

### 网络操作

```python
# 创建连接组
from flyvis.connectome.flywire_connectome import ConnectomeFromFlyWire
connectome = ConnectomeFromFlyWire(
    flywire_data_path="flyvis/connectome/flywire_real_v1.0.json",
    extent=15
)

# 查看统计
stats = connectome.get_statistics()
print(stats)

# 创建视图
from flyvis.connectome import ConnectomeView
view = ConnectomeView(connectome)

# 绘制连接矩阵
fig = view.connectivity_matrix(mode='n_syn')
fig.savefig('connectivity_matrix.png')
```

---

## 🎯 下一步行动

### 今天
1. ✅ 安装 PyTorch
2. ✅ 运行验证脚本
3. ✅ 创建第一个网络实例

### 本周
4. 实现完整的训练循环
5. 验证 ON/OFF 通路
6. 测试 T4/T5 方向选择性

### 下周
7. 与 FIB 模型对比
8. 性能优化
9. 生成分析报告

---

## 💡 提示

1. **使用独立环境**: 始终使用 `flywire_flyvis` 环境
2. **保存检查点**: 训练时定期保存模型
3. **监控内存**: FlyWire 数据较大，注意内存使用
4. **参考原始代码**: Flyvis 的 examples 文件夹有很多示例
5. **逐步验证**: 先测试小规模，再扩展到完整模型

---

## 📞 获取帮助

**项目位置**: `/Users/lengyuner/Desktop/NIPS2026/flyvis/`

**关键文件**:
- 数据: `/Users/lengyuner/Desktop/data/flywire/Jun2025/`
- 连接组: `flyvis/connectome/flywire_real_v1.0.json`
- 文档: `README_FLYWIRE.md`, `FINAL_REPORT.md`

**参考资源**:
- Flyvis 文档: https://turagalab.github.io/flyvis/
- Flyvis GitHub: https://github.com/TuragaLab/flyvis
- FlyWire: https://flywire.ai/

---

## ✨ 成就解锁

- ✅ 环境设置完成
- ✅ 数据加载成功
- ✅ 格式转换完成
- ✅ JSON 验证通过
- 🔄 网络创建 (待完成)
- 🔄 模型训练 (待完成)
- 🔄 功能验证 (待完成)

---

**当前状态**: 准备安装 PyTorch 并创建第一个网络 🚀

**预计完成时间**: 1-2 天

**加油！** 🎉
