# 文档索引

**项目**: Connectome-Constrained Neural Network Framework  
**目标**: NeurIPS 2026  
**状态**: Phase 1 完成 ✅

---

## 📚 核心文档

### 入门指南
- **[README (English)](../README.md)** — 项目概述、快速开始、架构说明
- **[README_zh (中文)](README_zh.md)** — 中文版项目说明和使用指南

### 关键特性
- **[全神经元建模支持](all_neurons_support.md)** ⭐ — 如何加载和建模所有神经元
  - BANC: 115,151 neurons, 11,193 cell types
  - FAFB: 138,043 neurons, 8,772 cell types
  - 性能估算、最佳实践、代码示例
- **[全神经元验证报告](all_neurons_verification.md)** — 测试结果和性能基准

### 技术文档
- **[架构文档](architecture.md)** — 项目架构、模块设计、数据流
- **[实现总结](implementation_summary.md)** (English) — 技术实现细节、Bug 修复
- **[TODO 列表](todo.md)** — 开发路线图、待办事项

---

## 📊 可视化报告

### 已完成的可视化
1. **[01_connectome_and_network.ipynb](../notebooks/01_connectome_and_network.ipynb)**
   - T4/T5 → LC 通路演示
   - 695 neurons, 3 cell types
   - Voltage, LIF, HH 模型对比

2. **[02_full_connectome_visualization.ipynb](../notebooks/02_full_connectome_visualization.ipynb)**
   - 完整视叶连接组
   - 925 neurons, 25 cell types
   - 详细拓扑分析和功能分组
   - **报告**: [full_connectome_visualization_report.md](full_connectome_visualization_report.md)

3. **[03_fafb_full_brain.ipynb](../notebooks/03_fafb_full_brain.ipynb)**
   - FAFB v783 全脑连接组
   - 138,043 neurons, 8,772 cell types
   - Super-class 分析、视觉系统分析
   - **报告**: [fafb_full_brain_report.md](fafb_full_brain_report.md)

### 综合报告
- **[所有可视化完成报告](ALL_VISUALIZATION_COMPLETE.md)** — 三个笔记本的综合总结

---

## 🐛 调试和测试

### 调试报告
- **[笔记本调试报告](notebook_debug_report.md)** — LIF 模型输出修复、刺激强度调整

### 测试覆盖
- **[test_loader.py](../tests/test_loader.py)** — 22 个单元测试
  - BANC, Optic Lobe, FlyVis, FAFB 数据加载
  - 网络构建、前向传播、梯度流
  - ✅ 20 passed, 2 skipped

---

## 📈 项目状态

### Phase 1: 框架基础 ✅ (完成)
- **[Phase 1 完成报告](PHASE1_COMPLETE.md)** — 详细完成状态
- ✅ 连接组数据加载器 (4 个数据源)
- ✅ 神经元动力学模型 (Voltage, LIF, HH)
- ✅ 网络构建 (PyTorch, Jaxley)
- ✅ 单元测试 (22 tests)
- ✅ 演示笔记本 (3 notebooks, 14 figures)
- ✅ 完整文档

### Phase 2: 数据集成 (进行中)
- [ ] 加载 LC 钙成像真实数据
- [ ] 对齐时间序列（刺激 ↔ 神经活动）
- [ ] 数据预处理和归一化

### Phase 3: 训练方法 (待开始)
- [ ] 实现 Method A (knockout training)
- [ ] 实现 Method B (layer-wise training)
- [ ] 评估方向选择性 vs DMN baseline

### Phase 4: 分析和可视化 (待开始)
- [ ] 学习到的权重分析
- [ ] 方向选择性指数 (DSI)
- [ ] 与生物数据对比

---

## 📝 变更历史

- **[CHANGELOG.md](CHANGELOG.md)** — 详细的版本变更记录

### 最近更新 (2026-04-04)
- ✅ 验证全神经元建模支持
- ✅ 创建 `all_neurons_support.md` 文档
- ✅ 创建 `all_neurons_verification.md` 报告
- ✅ 更新 README 和 README_zh
- ✅ 性能基准测试 (7.8k, 35k, 138k neurons)

### 重要里程碑
- **2026-03-30**: Phase 1 完成
- **2026-04-04**: FAFB 全脑可视化完成
- **2026-04-04**: 全神经元建模支持验证

---

## 🔗 快速链接

### 常用文档
| 文档 | 用途 |
|------|------|
| [all_neurons_support.md](all_neurons_support.md) | 如何加载所有神经元 |
| [architecture.md](architecture.md) | 理解项目结构 |
| [todo.md](todo.md) | 查看待办事项 |
| [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) | 查看完成状态 |

### 代码示例
| 笔记本 | 内容 |
|--------|------|
| [01_connectome_and_network.ipynb](../notebooks/01_connectome_and_network.ipynb) | 基础演示 |
| [02_full_connectome_visualization.ipynb](../notebooks/02_full_connectome_visualization.ipynb) | 视叶分析 |
| [03_fafb_full_brain.ipynb](../notebooks/03_fafb_full_brain.ipynb) | 全脑分析 |

### 测试
| 文件 | 内容 |
|------|------|
| [test_loader.py](../tests/test_loader.py) | 数据加载和网络构建测试 |

---

## 📧 联系方式

**项目路径**: `/Users/lengyuner/Desktop/NIPS2026/neuro_framework`  
**数据路径**: `/Users/lengyuner/Desktop/data/flywire/Jun2025` (FAFB)

---

**最后更新**: 2026-04-04  
**文档版本**: v1.0
