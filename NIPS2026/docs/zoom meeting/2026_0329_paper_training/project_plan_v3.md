# 项目计划 v3.0 - 基于真实连接组的果蝇视觉神经网络

> 基于 2026年3月29日会议重大调整  
> 上一版本：`docs/zoom meeting/2026_0322/project_plan_v2.md`

---

## ⚠️ 重要更新：Paper 范围与技术路线调整

本次会议（Jizheng + Frank 小范围技术对接）对 paper 的核心目标做出了**重大调整**：

| 变更项 | 旧方案（v2.0） | 新方案（v3.0）|
|--------|---------------|---------------|
| Body model | 联合训练核心目标 | **仅作 Demo 展示，不进入训练** |
| Paper 核心 | 视觉 + 行为闭环 | **视觉系统（retina → LC）** |
| 代码框架 | PyTorch / FlyVis | **Jaxley（JAX-based）** |
| 训练方式 | 端到端闭环 | **DMN-style + 逐层渐进（两种方案并行）** |

---

## 项目概述

**项目名称**: 基于果蝇真实连接组的视觉系统神经网络建模  
**目标会议**: NeurIPS 2026

| 里程碑 | 日期 |
|--------|------|
| Abstract 截止 | 2026年5月4日 |
| Full Paper 截止 | 2026年5月6日 |
| 距今剩余时间 | ~5.5 周 |

**核心目标（更新后）**：
- 使用 **FAFB（FlyWire）真实 connectome 中的 LC 神经元**，替换 DMN 论文中的简化 LC 层
- 实现视觉输入 → T/Tm → **LC（真实连接组）** → DN 的端到端训练
- 系统性能优于 DMN baseline 和 RF baseline
- Body model（NeuroMechFly）以 Demo 形式展示（非训练核心）

---

## 技术架构（v3.0）

```
视觉输入（Retina / Visual Scene）
        ↓
T4/T5 细胞（运动检测，Motion Detection）
        ↓
Tm 细胞（Transmedullary Neurons）
        ↓
┌─────────────────────────────────────────┐
│  LC 细胞（Lobula Columnar Neurons）       │  ← 本次核心：替换为真实 FAFB connectome
│  - 使用真实突触连接（from FlyWire FAFB） │
│  - 带时间动力学的神经元模型（DMN-style） │
│  - 训练参数：突触权重、时间常数          │
└─────────────────────────────────────────┘
        ↓
DN（Descending Neurons，视觉系统输出）
        ↓
[可选展示] NeuroMechFly Body Model (MuJoCo)
        ↓
预测行为输出

训练方法 A：DMN-style knockout/silence（对单个 neuron type 做约束）
训练方法 B：逐层渐进训练（上游固定后再训练下游）
```

---

## 团队成员与分工（v3.0）

### Jizheng Dong（五竹）— 项目负责人 / 神经科学内容

**主要职责**：
1. **神经科学内容把关**
   - 确定需要哪些神经元层次（T4/T5 → Tm → LC → DN）
   - 指导如何从 FAFB 中提取和使用 LC connectome 数据
   - 设计训练方案（方案 A/B 的具体实现细节）

2. **Jaxley 框架填充**
   - 在 Frank 搭建的代码框架中，填充神经科学相关内容
   - 实现 DMN-style 训练（方案 A）
   - 设计逐层渐进训练流程（方案 B）

3. **Synapse 参数研究（长期）**
   - 研究通过调整 synapse 模型中的关键参数来解决 HH 模型 signal decay 问题
   - 探索基因数据推导突触参数的可能性（方案 C，独立研究方向）

4. **论文撰写与项目协调**
   - 负责论文主体架构和 NeurIPS 投稿
   - 在群里同步 3月29日会议决策给全体成员
   - 申请 GPU 计算资源

**近期任务**：
- [ ] 在群里同步本次技术路线调整
- [ ] 确认 LC calcium imaging 数据可用范围（哪些 cell type 有数据）
- [ ] 在 Frank 的框架上填充 Jaxley 实现细节

---

### Yi-De Tai（Frank）— 代码框架 / 连接组数据处理

**主要职责（本次会议确定的核心任务）**：
1. **Jaxley 代码框架搭建**（最高优先级）
   - 搭建整体代码骨架，包括以下功能模块：
     - 数据读取（从 FAFB 原始数据读取 connectome）
     - Connectivity matrix 构建
     - 神经网络模型接口（支持可替换的神经元模型：LIF / DMN-style / HH）
     - 训练循环框架
   - 框架设计原则：**神经元模型接口不固定死**，方便后续替换

2. **FAFB → Jaxley 数据转换**
   - 将 FlyWire FAFB connectome 数据转换为 Jaxley 支持的格式
   - 构建 visual system 相关层次（T/Tm/LC/DN）的 connectivity matrix
   - 同步准备 BANC 数据接口（格式类似，切换成本低）

3. **Jaxley 学习与推广**
   - 熟悉 Jaxley 框架，成为团队的 Jaxley 技术支持

**近期任务**：
- [ ] 本周：学习 Jaxley 框架，跑通基础示例
- [ ] 一周内：搭建代码框架骨架（数据读取 + connectivity matrix + 模型接口）
- [ ] 一周内：将 FAFB LC connectome 数据转换为 Jaxley 格式

---

### Vulcan Z（Alex）— Body Model 维护 / Demo 准备

**职责调整**：
- Body model 不再是本次 paper 的训练核心
- Vulcan 的主要任务转变为**维护 NeuroMechFly，为最终 Demo 做准备**

**主要职责**：
1. **NeuroMechFly 维护与完善**
   - 保持 NeuroMechFly 可正常运行的状态
   - 完善编程控制接口（如何通过代码控制各种动作）
   - 将功能文档化，方便其他人接入

2. **Demo 准备**
   - 等前端（视觉网络）训练出 DN 激活后，接入 NeuroMechFly 做行为展示
   - 探索如何从 DN 输出到 muscle signal 的映射

3. **FlyWire 数据辅助**
   - 已完成的约 100 个运动相关神经元筛选继续维护
   - 协助 Frank 的 connectome 数据处理工作

**近期任务**：
- [ ] 持续维护 NeuroMechFly，保持可用状态
- [ ] 整理 NeuroMechFly 输入输出接口文档
- [ ] 学习 Jaxley 框架（与全体成员同步）

---

### Sherry（Xuyi）— SNN / RF 模型参考

**主要职责**：
1. **RF 模型知识贡献**
   - 对 RF 模型（Shiu et al.）了解较深，提供 RF 模型参数设置参考
   - 协助 Frank 在 Jaxley 框架中配置合适的神经元参数

2. **SNN 背景应用**
   - 探索 SNN（LIF 模型）在 Jaxley 框架中的集成可能
   - 与 DMN-style 模型做对比

3. **代码开发支持**
   - 学习 Jaxley 框架
   - 参与训练流程开发

**近期任务**：
- [ ] 学习 Jaxley 框架
- [ ] 整理 RF 模型中可用于 Jaxley 的参数设置

---

## 训练方案详解

### 方案 A：DMN-style Knockout/Silence（优先推进）

```
数据来源：LC calcium imaging 数据（来自文献，约5种 LC type 有完整数据）

训练流程：
  1. 给定视觉输入（grating 或 natural scene）
  2. 前向传播：retina → T → Tm → LC（真实 connectome）→ DN
  3. Loss：每次只对一种 LC neuron type 的输出做约束（对比 calcium imaging label）
  4. 反向传播更新突触权重
  5. 可与 DMN 论文结果直接比较

优点：有 DMN 成熟方法参考，结果对比清晰
挑战：需要确认哪些 LC cell type 有 calcium imaging 数据
```

### 方案 B：逐层渐进训练

```
训练流程：
  第一阶段：只训练 T/Tm 层
    - 数据：T4/T5/Tm 的 calcium imaging 数据（文献中有全脑成像数据）
    - 训练完成后，冻结 T/Tm 参数
  
  第二阶段：在冻结的 T/Tm 基础上，训练 LC 层
    - 数据：LC calcium imaging 数据
    - 只更新 LC 相关参数
  
  第三阶段（可选）：加入 DN 层
    - 数据：DN 相关数据（如有）

优点：每次训练参数量小，计算更稳定；可逐步扩展到全脑
挑战：需要整合来自不同 paper 的多个数据集，对齐困难
```

### 方案 C：基因推导突触参数（Jizheng 长期研究，不纳入 NeurIPS 主线）

```
思路：
  - 用基因表达数据推导 synapse 模型中的关键参数
  - 解决 HH 模型在多层传递中的 signal decay 问题
  - 只需调对一个关键 synapse 参数，信号即可跨层传递

当前状态：Jizheng 私下研究中，需要专用算法，风险较高
后续：可能成为独立 paper 的主题
```

---

## 数据规划

| 数据类型 | 来源 | 用途 | 状态 |
|----------|------|------|------|
| FAFB connectome（视觉系统）| FlyWire | 构建 LC 网络结构 | 可用，Frank 处理 |
| BANC connectome | BANC | 备用，可扩展到 DN 层 | 接口准备中 |
| LC calcium imaging | 文献（*eLife* LC [21022](https://elifesciences.org/articles/21022)、*Nat. Neurosci.* 2025 等，见文末参考资料）| 训练 label | 约5种 type 确认，其余需核实 |
| T/Tm calcium imaging | 文献 | 方案 B 上游训练 | 待整理 |
| 39关键点行为数据 | Jizheng 实验室 | 后续 body model Demo 用 | 已有 |

---

## 时间线（v3.0）

```
2026年3月29日（本周）
  - Frank：学习 Jaxley，跑通示例
  - Jizheng：在群里同步路线调整，确认 LC 数据可用范围
  - 全体：熟悉 Jaxley 框架

2026年4月第1周（4月1日~7日）
  - Frank：完成代码框架骨架 + FAFB → Jaxley 数据转换
  - Jizheng：在框架中填充方案 A 实现
  - Vulcan：整理 NeuroMechFly 接口文档

2026年4月第2周（4月8日~14日）
  - 跑通方案 A 的基础训练
  - 评估方案 B 的可行性，准备多数据集对齐
  - 与 DMN baseline 对比初步结果

2026年4月第3~4周（4月15日~28日）
  - 完善训练，优化超参数
  - 开始方案 B 的逐层训练实验
  - 撰写论文 Methods 和 Results 部分

2026年5月第1周（4月29日~5月4日）
  - 完成实验结果
  - 完成 Abstract 并提交（5月4日截止）
  - NeuroMechFly Demo 准备（如有余力）

2026年5月5日~6日
  - Full Paper 最终润色与提交
```

---

## 关键技术风险

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| LC calcium imaging 数据不足 | 中 | 高 | 先确认数据范围；不足则缩小到有数据的 cell type |
| Jaxley 框架学习曲线陡 | 中 | 中 | Jizheng 有经验，可快速指导；Frank 专注于此 |
| 方案 A/B 训练效果不佳 | 中 | 高 | 两方案并行，任一成功即可；失败结果也可写对比 figure |
| 时间不足（5.5周）| 高 | 高 | 聚焦核心（LC替换），其余特性优先级降低 |
| FAFB 数据格式转换复杂 | 低 | 中 | Frank 有 connectome 经验，从原始数据读取 |

---

## 成功标准（v3.0）

### 最低标准（Must Have）
- [ ] 用 Jaxley 搭建完整的视觉系统网络（retina → LC）
- [ ] 使用真实 FAFB LC connectome 替换 DMN 中的简化 LC
- [ ] 方案 A 训练可收敛，结果可与 DMN baseline 比较
- [ ] 完成论文并提交 NeurIPS 2026

### 期望标准（Should Have）
- [ ] 方案 B 逐层渐进训练也可以跑通
- [ ] 在多种 LC cell type 上验证，结果优于 DMN baseline
- [ ] NeuroMechFly Demo 展示（DN 输出 → 行为动作）
- [ ] 论文被 NeurIPS 主会接收

### 理想标准（Nice to Have）
- [ ] BANC 版本（扩展到 LC → DN）
- [ ] 方案 C（synapse 基因参数）集成
- [ ] 代码开源

---

## 工具与资源

| 类别 | 工具/资源 |
|------|-----------|
| 核心框架 | **Jaxley**（JAX-based 神经仿真） |
| 补充框架 | PyTorch（训练辅助），FlyVis（参数参考） |
| 连接组数据 | FlyWire FAFB（主），BANC（备） |
| 即时通讯 | 微信群 |
| 代码协作 | GitHub |
| 会议 | Zoom |
| AI 辅助 | Cursor + Claude API（Jizheng 分享）|

---

## 参考资料

### 核心方法学与对比基准

| 论文 | 链接 | 重要性 | 说明 |
|------|------|--------|------|
| Connectome-constrained networks (DMN), *Nature* 2024 | [s41586-024-07939-3](https://www.nature.com/articles/s41586-024-07939-3) | 🔴 核心 | 任务约束 + connectome；直接对比基准 |
| Cowley et al., connectome + genetic perturbations / behavior, *Nature* 2024 | [s41586-024-07451-8](https://www.nature.com/articles/s41586-024-07451-8) | 🔴 核心 | knockout 式约束 baseline；与 DMN 同期对照阅读 |

### Connectome、活动预测与强连接（2025 起补充）

| 论文 | 链接 | 重要性 | 说明 |
|------|------|--------|------|
| Prediction of neural activity in connectome-constrained recurrent networks, *Nat. Neurosci.* 2025 | [s41593-025-02080-4](https://www.nature.com/articles/s41593-025-02080-4) | 🔴 关键 | 活动预测与约束；与 4/11 会讨论的「需 recording/activity 约束」直接对应 |
| Infrequent strong connections constrain connectomic predictions of neuronal function, *Cell* 2025 | [S0092-8674(25)00518-5](https://www.cell.com/cell/fulltext/S0092-8674(25)00518-5) | 🟡 重要 | 稀疏强连接对 connectome 功能推断的限制 |

### 视觉线路、细胞目录与 ON/OFF

| 论文 | 链接 | 重要性 | 说明 |
|------|------|--------|------|
| Neuronal parts list and wiring diagram for a visual system, *Nature* 2024 | [s41586-024-07981-1](https://www.nature.com/articles/s41586-024-07981-1) | 🔴 关键 | 视觉系统细胞目录与连线；[Figure 4](https://www.nature.com/articles/s41586-024-07981-1/figures/4) 等与 **ON/OFF** 协议对照 |

### T4/T5、LC 与 voltage→calcium

| 论文 | 链接 | 重要性 | 说明 |
|------|------|--------|------|
| Differential temporal filtering in the fly optic lobe (T4/T5), *J Comput Neurosci* 2025 | [10.1007/s10827-025-00914-5](https://link.springer.com/article/10.1007/s10827-025-00914-5) | 🟡 重要 | T4/T5 时间滤波与视叶动力学参考 |
| Visual projection neurons in the *Drosophila* lobula… (LC), *eLife* 2017 | [10.7554/eLife.21022](https://elifesciences.org/articles/21022) | 🔴 关键 | LC 特征检测与行为程序；可作 **voltage→calcium / 成像 readout** 与行为层面对照 |

### 工具、连接组数据与其他

| 论文 / 资源 | 链接 | 重要性 | 说明 |
|-------------|------|--------|------|
| Jaxley | 以官方论文 / GitHub 为准 | 🔴 核心 | 代码框架，Frank 优先阅读 |
| LC bottleneck / calcium imaging 综述与数据源 | 见上 *eLife* LC、*Nat. Neurosci.* 2025 等 | 🔴 关键 | 训练 label 与验证设计 |
| NeuroMechFly, *Nat. Methods* 2024 | 文献库检索 | 🟡 重要 | Vulcan 维护、Demo |
| FlyWire FAFB v783 数据集论文 | 文献库检索 | 🟡 重要 | connectome 数据来源 |

---

*最后更新：2026年4月12日（参考资料表扩展：Cell / Nat. Neurosci. / Nature parts list / T4–T5 / eLife LC 链接）*  
*版本：3.0（技术路线重大调整：聚焦视觉系统，采用 Jaxley 框架）*  
*基于：Jizheng Dong + Yi-De Tai 技术对接会议*
