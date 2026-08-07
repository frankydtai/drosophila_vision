# 会议纪要 - 2026年4月5日

## 参会人员

| 姓名 | 角色 |
|------|------|
| **Jizheng Dong（五竹）** | 项目负责人；主讲技术路线与参数调试进展 |
| **Yi-De Tai（Frank）** | 训练主线推进（FlyVis/DMN 路线 + FAFB 替换） |
| **Sherry（Xuyi）** | 代码实验与参数敏感性分析（CPU/GPU 两版） |
| **Alex（Vulcan）** | Body model / close-loop 探索（会中后段加入） |

> 说明：本次转录噪声较大（中英夹杂 + ASR 错词），以下纪要已结合 3/29 会议路线与近期讨论语境做术语还原。

---

## 会议主题

围绕 `project_plan_v3.md` 的主线，团队集中讨论：
1) 视觉网络主线应优先聚焦哪种训练策略；
2) Frank 当前 **FAFB 替换后训练不下降** 的卡点；
3) Sherry 的数据/代码流程与参数扫描；
4) Alex 的 close-loop 方向如何衔接（先 demo 还是先训练闭环）。

---

## 核心结论（先看）

1. **论文主线仍聚焦视觉系统，不做完整闭环训练**：先把 visual（retina/Tm/T4/T5/LC）训稳，再谈 motor/NeuroMechFly 的演示接入。  
2. **训练方法并行，但优先解决方法 A 的“loss 不降”**：Frank 已在 FlyVis/DMN 路线上替换 FAFB，出现几乎无信号与训练 loss 不下降。  
3. **数据选择上短期继续 FAFB，BANC 暂作备选**：BANC 可补到 motor 相关，但当前 completeness/quality 风险更高。  
4. **输入映射要按真实视觉几何处理**：不要强行沿用理想正六边形模板；优先真实坐标，再做可解释的裁剪/覆盖策略。  
5. **Sherry 继续做“可复现分析”，Frank 继续做“可收敛训练”**：两条线互相喂结果；减少重复踩坑。  
6. **协作方式改进**：强调日常群内持续 update，而非只在周会集中同步。

---

## 详细讨论

### 1) 技术路线再确认：四类训练思路并行评估

Jizheng 开场总结了当前考虑的 4 类路线（术语按上下文还原）：

- **方法 A：任务约束训练（DMN/FlyVis 风格）**  
  输入视频，目标 optic flow / 相关任务指标，端到端训练。

- **方法 B：扰动约束训练（knockout/silence 风格）**  
  对特定 neuron type 做失活，利用行为变化约束模型。

- **方法 C：活动数据约束（calcium/activity）逐层训练**  
  先上游（如 Tm/T4/T5）再下游（LC/DN），分层收敛。

- **方法 D：更生物细节的 neuron/synapse 参数化（HH/multi-compartment + adaptive threshold）**  
  Jizheng 在试，当前自评成功率约 60%~70%，但仍在调。

**会中共识**：由于距离 NeurIPS 截止仅约 1 个月，必须优先“能收敛、能复现、可写 paper”的路径。

---

### 2) 范围控制：为何不把 full close-loop 作为当前主训练目标

Jizheng 反复强调：

- 若直接把大规模 central brain + behavior 全部端到端训练，**中间参数太多且缺监督锚点**，可疑性高。  
- 行为回授会改变下一帧视觉输入，真实世界与模拟环境偏差会不断累积，导致监督困难。  
- 因此现阶段应 **focus 小区域先做扎实**（视觉系统），body model 作为后续 demo/接口验证更稳妥。

这与 3/29 的 `project_plan_v3.md` 决策一致：**视觉主线优先，闭环展示后置**。

---

### 3) Frank 汇报：当前三大卡点

#### 3.1 Connectome 数据切换：FAFB 可用，BANC 暂不优先

- Frank 提到若换 BANC 问题更大（例如早期看不到完整 receptor 位置信息）。  
- 会中共同查看后确认：BANC 有 skeleton/morphology 等数据，但完备性与 proofread 质量存在疑虑。  
- 最终建议：**短期继续用 FAFB 主线推进**，BANC 作为后续扩展（尤其 motor 方向）再引入。

#### 3.2 视觉输入几何映射问题（关键）

- Frank 展示了自己绘制的六边形/坐标映射图：FAFB 真实视觉坐标与 FlyVis 规则网格存在差异。  
- Jizheng 建议：**按真实几何优先**，必要时做“放大覆盖 / 中心裁剪 / 边缘舍弃”来与现有输入管线对齐，而不是强行把真实结构硬变成理想模板。  
- 结论：该问题可解，但要先固定一个一致映射策略，避免每次实验映射方式都变。

#### 3.3 训练不收敛：FAFB 替换后“几乎无信号 + loss 不下降”

- Frank 展示结果：用 FAFB 连接后，部分输出几乎全零，训练 loss 长时间在高位附近震荡，不呈持续下降。  
- Jizheng 观察：这不是简单多跑几轮就能解决，更可能是设置/结构层面问题。  
- 双方提到可能原因：
  - 真实连接下 recurrent/recruitment 链接更多，动力学特性变了；
  - 与原始 FlyVis 假设的规模、连接分布、输入映射不一致；
  - 前几层 ON/OFF 传递机制在真实结构里更“脆弱”，参数不对就会信号衰减或符号错位。

---

### 4) ON/OFF 机制复盘（用于解释为何“看起来没信号”）

Jizheng 用视觉通路（R1–R8、L1/L2、Tm、T4/T5、Mi 等）解释：

- 果蝇视觉前两层有大量 **抑制链路 + 双重抑制** 逻辑；  
- ON 与 OFF 通道依赖不同中间节点，符号/增益稍有偏差就会导致下游表现反常；  
- 因此“输出弱/反向/不稳定”在早期并不罕见，要结合 pathway 逐层诊断，不宜只盯最终 loss。

会中也提到一篇视觉系统 wiring 图（含 ON/OFF bar 相关图示）作为对照。

Neuronal parts list and wiring diagram for a visual system
https://www.nature.com/articles/s41586-024-07981-1	

---

### 5) Sherry 汇报：数据下载、运行流程与参数敏感性

Sherry 当前工作重点：

- 从 FAFB 数据与子集（如 R1-R6）出发做可运行实验；
- 比较 CPU 与 GPU 版本代码路径（GPU/PyTorch 版本显著加速，CPU 版本耗时过长）；
- 关注 synapse 相关参数对输出的影响（会中多次提到 `w_scale` 对活跃度影响最大）。

Jizheng 给出的建议：

- 先把 FAFB 关键目录完整下载到本地再实验（减少在线读数据造成的不稳定）；
- 不必在本会把所有文件格式都看完，先保证可复现；
- 对中间结果（尤其 pandas 序列化后的文件）用 notebook 脚本化读取，不要手工猜格式。

---

### 6) 神经元模型与参数讨论：Voltage / HH / Synapse

会中后段进入模型参数讨论，重点包括：

- 可能以 **Voltage 模型** 先跑通，再逐步替换为 **HH + synapse**；
- HH 参数中 Na/K/Leak 及阈值相关参数耦合较强，难以直接端到端稳定优化；
- Jizheng 展示自己当前参数化：
  - 一部分描述 neuron 内部动力学（HH）；
  - 一部分描述 synapse 开关概率、衰减与 postsynaptic 作用；
- 提议可借鉴已有训练后参数做初始化，以提高收敛概率（对 Frank 线有参考价值）。

---

### 7) Alex 汇报：close-loop 下一步

Alex（后段恢复语音）表示：

- 本周暂无实质新结果；
- 下一步希望探索“已有视觉网络 output 时，如何接入 fly/body module 并闭环”。

Jizheng 回复：

- 可先参考两类方案：
  1) 论文里用较简化 decision network（如 MLP）从视觉输出到动作；
  2) 看 FlyGym / 相关项目如何把 connectome 输出接到控制层；
- 若追求更完整 motor 细节，可能要依赖 BANC 等数据，但质量与工程复杂度更高。  

共识：**Alex 先做方案调研与小规模验证，不阻塞视觉训练主线**。

---

## 任务分工与行动项

| 优先级 | 任务 | 负责人 | 说明 |
|--------|------|--------|------|
| 🔴 高 | 继续排查 FAFB 替换后 loss 不下降根因 | Frank | 固定输入映射策略 + 检查前层通路与参数设置 |
| 🔴 高 | 做可复现实验与参数敏感性记录（含 `w_scale` 等） | Sherry | 侧重分析，不与 Frank 主训练线重复 |
| 🔴 高 | 持续调 HH + synapse 参数，并沉淀可借鉴初始化 | Jizheng | 供 Frank/Sherry 复用 |
| 🟡 中 | 调研 visual output → behavior 的简化闭环接法 | Alex | 先 paper/框架调研，再小规模实现 |
| 🟡 中 | 统一代码协作习惯（branch/文档/中间结果共享） | 全体 | 避免“谁改了哪版”不清楚 |
| 🟢 低 | BANC 作为后续扩展做数据可用性补查 | Jizheng + Frank | 不影响当前 FAFB 主线 |

---

## 下次会议与协作机制

- 下次会议时间暂未完全固定（涉及值班与时区）。  
- 大家一致同意：**平时在群里持续 update**，减少周会冗长同步。  
- Frank 表示会继续在群内发结果；Sherry 有问题可直接在群里问；Jizheng 继续提供参数与方向校准。

---

## 转录噪声校正（本次高频）

| 转录词 | 推测原意 |
|--------|----------|
| fireways / fly with / flybase | **FlyVis**（或其代码仓） |
| faa / fab / ffb | **FAFB** |
| bank | **BANC** |
| rc / rlc | **LC**（lobula columnar）或相关层级缩写 |
| dna | **DN**（descending neurons） |
| multiple empowerment model | **multi-compartment model** |
| chh | **HH**（Hodgkin-Huxley） |
| synaps / sign up | **synapse** |
| recruit 链接 | **recurrent**（递归连接）语境更可能 |
| 训/劝 | **训练** |
| magical / 墨竹口 | **MuJoCo** |
| close这个loop | **close the loop** |
| neon/new | **neuron** |
| virtual/virtial column | **visual column** |

---

*会议时间：2026年4月5日 09:02 - 11:10（以转录时间戳为准）*  
*整理依据：`meeting_saved_closed_caption.txt`，并参考 `2026_0322`、`2026_0329_paper_training` 纪要与 `project_plan_v3.md` 对齐术语与路线*  
*版本：1.0*
