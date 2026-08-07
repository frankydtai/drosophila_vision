# 会议纪要 - 2026年3月29日

## 参会人员

| 姓名 | 角色 |
|------|------|
| **Jizheng Dong (五竹)** | 项目负责人 |
| **Yi-De Tai (Frank)** | 神经网络模型开发 |

> 本次为 Jizheng + Frank 的小范围技术对接会，其他成员（Vulcan、Sherry）未参与。

---

## 会议主题

**明确 Paper 的技术路线和训练方案**：厘清现有数据、可训练的模型，以及接下来的具体分工。

---

## 核心决策与讨论

### 1. Paper 范围调整：聚焦视觉系统，暂时不做 Body Model

**背景**：上次会议（3月22日）讨论了 NeuroMechFly body model 联合训练的可能性。本次明确决定：

- **Body model（MuJoCo/NeuroMechFly）本篇 paper 不作为训练核心**
  - 加入 body model 后训练极其复杂，短期内无法完成
  - 目前相关团队（如 Turaga 组）在做类似工作，但我们没必要跟进
  - Body model 最终可能以「Demo 展示」的形式出现，而非训练目标
- **本次 paper 核心**：视觉神经网络（visual system）
  - 路径：视觉输入 → T 细胞（T4/T5）→ Tm 细胞 → **LC（Lobula Columnar）** → DN（Descending Neurons）
  - LC 是视觉系统的主要输出层，是本次工作的关键节点

---

### 2. Paper 核心思路：用真实 Connectome 替换 DMN 中的 LC 层

**DMN 论文现状**：
- DMN 团队已经用新的 FAFB 数据做过实验，但**没有发表** 
- DMN 论文中的 LC 是简化版，并非来自真实 connectome

**我们的方向**：
- 使用真实 FAFB connectome 数据中的 LC 神经元，**替换** DMN 中的简化 LC 层
- 保留相似的训练框架（knockout/silence 方法）
- 这个方向难度适中，且与 DMN 有明确的可比性

**网络层次（从视网膜到运动输出）**：
```
视觉输入（retina）
    ↓
T4/T5 细胞（motion detection）
    ↓
Tm 细胞（transmedullary neurons）
    ↓
LC 细胞（Lobula Columnar）← 本次重点替换为真实 connectome
    ↓
DN（Descending Neurons）← 视觉输出，连接中央脑
    ↓
（运动/行为 → 暂时不做完整闭环）
```

---

### 3. 训练方法讨论：三种方案

Jizheng 提出了三种可能的训练方法，讨论如下：

#### 方案 A：两种有参考基础的训练方法（可选其一或组合）

> 这两种方法来自两篇不同的 Nature 2024 论文，需要区分：

**方法 A1：DMN 任务约束训练**（来自 [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3)）
- 核心思路：连接组约束（connectome-constrained）+ 任务优化（task-optimized）
- 将真实 connectome 结构固定，对可训练参数（突触权重缩放因子 α、时间常数 τ、静息电位）做梯度优化
- 训练任务：让网络完成 **optic flow estimation**（光流估计），用自然视频（Sintel 数据集）作为输入
- 验证方式：与 26 篇实验文献中记录的神经元活动逐一比对，无需直接使用神经活动作为训练数据
- 关键结论：connectome 越稀疏 → 任务优化后预测结果与真实越接近（与 Jizheng 的 Teacher-Student 结论一致）
- **我们的延伸**：用 FAFB 真实 LC connectome 替换 DMN 原文中简化的 LC 层，保持相同训练框架，比较结果差异

**方法 A2：Knockout 训练**（来自 [Cowley et al., Nature 2024](https://www.nature.com/articles/s41586-024-07451-8)）
- 核心思路：用遗传学 **silence/knockout** 实验的行为数据作为训练约束，建立 model unit ↔ 真实神经元的一一对应
- 做法：训练时，若某种 LC neuron type 在真实实验中被 silence，则将模型中对应 unit 设为 0（knocked out）
- 训练数据：23 种 LC 类型各自 silence 后果蝇的行为记录（追求、鸣唱等）
- 目标：model unit 必须以与真实神经元相同的方式驱动行为 → 自动建立 1-to-1 mapping
- 验证方式：用 calcium imaging 验证模型预测的各 LC 神经元响应
- **与我们项目的关联**：提供了「用单个 neuron type perturbation 作为训练约束」的思路；若未来有 silence 实验数据可引入

**两种方法的关键区别**：

| | DMN 方法（A1） | Knockout 方法（A2） |
|--|--|--|
| 训练信号来源 | 视觉任务（optic flow） | 遗传扰动实验的行为数据 |
| 约束方式 | 任务误差梯度反传 | 对应 unit 在训练中设零 |
| 是否需要神经活动数据 | 不需要（验证时才用） | 不需要（验证时才用） |
| 可训练参数 | 突触权重 α、τ、静息电位 | 同 DMN + 决策网络权重 |
| 验证手段 | 与已发表神经活动数据比较 | 钙成像直接验证 |
| 适合场景 | 有 connectome + 视觉任务 | 有大量遗传扰动行为实验 |

- **本次 paper 主要参考 A1（DMN 方法）**：我们有 FAFB connectome 和视觉输入，缺少系统性的遗传扰动行为数据
- **A2（Knockout）的思路**可在未来有 silence 实验数据时引入，或用已有 silence 文献数据做部分验证

#### 方案 B：逐层渐进训练（Layered Progressive Training）
- 先用上游神经元（T4/T5/Tm）的 calcium imaging 数据训练上游部分
- 上游训练完后固定参数，再加入下游（LC）继续训练
- 优点：每次训练的参数量小，计算更可控；理论上可以扩展到全脑
- 挑战：需要整合来自不同 paper 的数据集
- **Jizheng 评价：理论可行，比较靠谱，后续可扩展到更多层**

#### 方案 C：基因推导突触参数（Jizheng 私下想法）
- 用基因数据推导/调整突触（synapse）模型中的关键参数
- **核心洞见（Jizheng 私下透露）**：HH 模型中 synapse 参数只需调对一个关键参数，信号就能在多层之间正常传递（解决 decay 问题）
- 这个参数目前需要用专门的算法来调整，难度较高，是 Jizheng 自己的研究课题
- 如果做出来，可能足以支撑一篇独立的 paper
- 优点：不依赖大量训练数据；缺点：风险较大，短期难以验证

**综合结论**：方案 A + 方案 B 为本次 paper 的主要路线；方案 C 为 Jizheng 的个人研究，暂不纳入 NeurIPS paper 主线。

---

### 4. 代码框架选型：Jaxley

**讨论背景**：需要选择一个可以支持反向传播（backprop）和神经动力学的代码框架。

**候选框架**：
- **RF 模型代码**（Brain to 系列）：无法支持反向传播，不适用
- **FlyVis（flyvis PyTorch）**：可以用，但 Jizheng 不想直接用（代码规范差）
- **Jaxley**（推荐）：
  - 基于 JAX 的神经网络仿真框架
  - 支持反向传播（可微分）
  - 支持真实神经元形态（morphology），可选择是否加入几何形状
  - Jizheng 近期常用，自称「推广大使」
  - 来自德国某研究组，与 FlyWire 有密切关联

**决定**：使用 Jaxley 作为主要代码框架。

**数据格式**：connectome 数据从 FAFB（FlyWire）原始数据读取，转换为 Jaxley 支持的 connectivity matrix 格式；不直接用 FlyVis 的处理结果。

---

### 5. 数据规划

**现有可用数据**：
1. **钙成像数据（calcium imaging）**：来自相关 paper（LC bottleneck 等）
   - 部分 LC cell type 有 calcium imaging 验证（约5种 LC type 有完整数据）
   - 其他 cell type 可能只有行为学数据（behavioral data）
2. **Connectome 数据**：
   - **FAFB（FlyWire）**：视觉系统部分（optic lobe）完整，精度高，优先使用
   - **BANC**：包含全脑（central brain + VNC），但 optic lobe 部分精度较低，暂时不用

**近期数据目标**：
- 先用 FAFB 做 visual → LC 的模型
- 同时准备好 BANC 的接口（两个数据格式类似，切换成本低）
- 不急于引入新数据，先把现有数据整合好

---

### 6. 神经元模型选型讨论

**问题**：DMN 使用的是简单的 machine learning 式神经元（无时间动力学），不适合生物真实性要求。

**讨论结论**：
- 至少要用类似 **DMN 那篇 paper** 的神经元模型：带时间常数（tau）的动力学模型
- 可使用 FlyVis 中的时间参数作为参考
- 更高级的选项：HH（Hodgkin-Huxley）模型，加上 synapse 模型
- Jaxley 框架支持多种神经元模型（LIF、HH 等），**模型接口设计时不固定死，保持可替换性**

**两个 connectome 版本的计划**：
- FAFB 版本：visual → T → Tm → LC（精度高）
- BANC 版本：可能延伸到 LC → DN（覆盖更多层）

---

### 7. 行为数据与 Body Model 的定位

- Body model（NeuroMechFly/MuJoCo）：目前由 Vulcan（Alex）维护，功能基本可用
- **本次 paper 策略**：body model 作为「展示 Demo」，不作为训练的一部分
  - 先做好视觉系统 → LC/DN 的训练
  - 拿到 DN 神经元的激活之后，再考虑接 body model 做动作展示
  - 这样可以避免梯度传播问题，降低训练难度
- 如果 closed-loop 能做出来，是加分项；做不出来，视觉系统部分本身也够发 paper

---

## 关键结论总结

1. **Paper 聚焦**：视觉系统（retina → T → Tm → **LC** → DN），用真实 FAFB connectome 替换 DMN 中的 LC 层
2. **Body model 延后**：本次不做 closed-loop 训练，body model 以 Demo 形式出现
3. **训练方案**：方案 A（DMN 式 knockout/silence）+ 方案 B（逐层渐进训练）并行推进
4. **代码框架**：Jaxley（JAX-based，支持 backprop 和神经动力学）
5. **神经元模型**：带时间动力学的模型（参考 DMN），接口设计保持可替换
6. **数据**：优先用 FAFB，同时准备 BANC 接口
7. **分工**：Frank 负责搭建 Jaxley 代码框架 + connectivity matrix；Jizheng 负责填充神经科学内容

---

## 下一步行动

| 优先级 | 任务 | 负责人 | 时间 |
|--------|------|--------|------|
| 🔴 高 | 用 Jaxley 搭建整体代码框架（数据读取、网络构建、接口设计） | Frank (Yi-De Tai) | 一周内 |
| 🔴 高 | 将 FAFB connectome 数据转换为 Jaxley connectivity matrix 格式 | Frank (Yi-De Tai) | 一周内 |
| 🔴 高 | 调研并熟悉 Jaxley 框架 | 全体 | 本周 |
| 🟡 中 | 在 Jaxley 框架中实现 DMN 式训练（方案 A） | Jizheng | 两周内 |
| 🟡 中 | 研究逐层渐进训练方案（方案 B）的数据对齐问题 | Jizheng + Frank | 两周内 |
| �� 中 | 确认 LC calcium imaging 数据的可用范围（哪些 cell type 有数据） | Jizheng | 本周 |
| 🟢 低 | Vulcan 继续维护 NeuroMechFly（为后续 Demo 做准备） | Vulcan (Alex) | 持续 |
| 🟢 低 | Jizheng 研究 synapse 参数基因推导方法（方案 C） | Jizheng | 长期 |

---

## 下次会议

- Jizheng 将在群里同步本次讨论结果给其他成员（Vulcan、Sherry）
- 下次会议时间待定 

---

*会议时间：2026年3月29日 22:03 - 22:53*  
*参与人员：Jizheng Dong, Yi-De Tai (Frank)*  
*版本：1.0*