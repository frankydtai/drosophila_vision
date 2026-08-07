# 会议纪要 - 2026年4月11日

## 参会人员

| 姓名 | 角色 |
|------|------|
| **Jizheng Dong（五竹）** | 项目负责人；本次中途因值班接电话，前半段以语音参与 |
| **Yi-De Tai（Frank）** | 模型复现与训练（FAFB / DMN 式管线） |

> **说明**：转录中 Sherry 被提及（Jizheng 需与其通话、值班），实际技术讨论主要在 **Jizheng + Frank** 之间进行，与 3 月 29 日「小范围对接会」形式一致。

---

## 会议主题

围绕 **NeurIPS 2026 视觉主线**（见 `project_plan_v3.md`）：Frank 复现 **DMN 式 connectome-constrained 训练** 时遇到的 **不收敛 / 梯度不下降** 问题；Jizheng 从 **初始化、归一化、网络暂态、刺激时序、数据子集** 等角度给建议；Jizheng 同步 **optic lobe / mcHH 笔记本与加载器** 进展；补充 **钙成像 / 频率响应** 类公开数据作为后续 **activity 约束训练** 的参考。

---

## 核心讨论（按主题）

### 1. 文献结论是否「矛盾」：DMN（Nature 2024）vs 强调 activity 约束的工作

**Frank 关切**：DMN 类工作强调 connectome + 任务即可；与 **connectome-constrained 下神经活动预测** 类工作对照时，易出现「仅结构 + 任务仍 **local minima 多、activity 不准**，需 **recording 类约束**」的表述——表面结论相反。  
**对应文献（已固定链接）**：*Nat. Neurosci.* 2025，[Prediction of neural activity in connectome-constrained recurrent networks](https://www.nature.com/articles/s41593-025-02080-4)。更完整的文献表见 `project_plan_v3.md`「参考资料」。

**Jizheng 看法**：不必看成完全矛盾；总体上仍是 **需要加入更多约束（constraints）**；DMN 结果本身也体现 **多单元中部分准、部分不准**，与「无 activity 约束时易不稳」可并存理解。

**对 Nature 2024（DMN）任务设置的批评（Jizheng）**  
- 并非直接 decode 神经活动，而是接 **MLP decoder** 再做下游 CV 任务，**生物学可解释性 / 与真实 readout 对齐** 方面较弱。  
- Frank 补充：当前更棘手的是 **连梯度都难以下降**，不仅是 decode「对不对」的问题。

---

### 2. Frank 侧训练问题：FAFB 替换后梯度不下降、与论文设定差异

**现象（Frank）**  
- 使用论文原始 connectivity 设定时 **尚可**；换成 **FAFB** 后 **loss 不下降**。  
- 已尝试：对 activation 做 **缩放 / sigmoid**（因数值动辄很大）、调整 synaptic count cap（如 **>100 截断为 100**）等，**仍未训练出稳定下降**。

**Jizheng 对机制的解释**  
- 若初始活动与真实分布差很远，**BCE 类 loss** 易长期贴在 0/1 两端 → **梯度饱和变小**。  
- **Synapse count 与权重初始化**：论文式「按 count 缩放」在 **真实 connectome 度分布极不均匀**（有的 presynaptic 几十上百条边、有的极少）时，与「各 type 数量大致均衡」的简化设定不同，直接 **除以 synapse count** 可能放大 **imbalance**。  
- **建议**：按 **incoming 权重在 presynaptic 上的占比** 做归一化（举例：多路输入按 0.2 / 0.4 / 0.4 分配相对强度），或对权重矩阵做 **除以总和 / softmax 式** 重整，使极端高度数节点不会 dominate；必要时 **在权重端加 gating / 额外非线性**，而不只改突触后 **f**（Frank 已把部分 **ReLU** 换成 **sigmoid** 以压 T4 等过强响应，Jizheng 认为 **f 用 ReLU 未必是主因**，更优先检查 **W 的初始化与归一化**）。

**Frank 对论文公式的质疑（转录校正：synapse count、符号 ±1、可学习 α）**  
- 按文中写法，易导致 **每条边上初始有效权重几乎相同**，**connectome 强弱先验被冲掉**，再靠 **α** 从头学，与真实 **count/strength** 信息利用方式不一致。  
- Frank 已改为 **每条边统一初值（如 0.05）** 等尝试，**仍未解决不下降**。  
- Jizheng：**不必拘泥原文每一项实现**；数据集中 **synapse count 本身也可能不准**，应 **以能优化为先** 改设定。

---

### 3. 网络暂态与刺激时序：先「跑稳」再给训练刺激

**问题**：训练用的是 **时间窗 / interval 类刺激**（转录作「Intel」→ **interval**），**一上来就刺激 + 反传**，可能仍在 **剧烈瞬态** 上。

**Jizheng 经验**  
- 初始网络常 **强烈振荡**；可 **先不加学习或只前向**，用 **噪声 / 零输入 / 固定 latent** 等让网络 **运行一段时间至相对稳态**，再 **施加视觉刺激** 并记录用于 loss 的窗口。  
- 也可 **先跑稳一次，把稳态后的 voltage 等存成之后每次初始化的 warm start**。  
- Frank 观察到 **初期 activity 特别大、后期较稳**，与此一致。

**可诊断步骤**：先看 **flash 响应** 在时间轴上 **多久进入可观测的稳定响应**，再对齐训练 protocol。

---

### 4. 视觉子图 neuron type 列表：旧 subset 缺 Lamina 层

**Frank**：早期 Jizheng 提供的 **visual subset**（转录「subsistence」→ **subset**）按某篇分类只含 **motion / color OFF / photoreceptor** 等，**缺少 Lamina 重要类型**（如 **L2、L3**），训练时会 **漏细胞类型**。  
**Frank**：已自行 **补全列表**，覆盖应有类型。  
**Jizheng**：该段代码自己 **近期较少直接使用**；本周主要工作在 **notebook / loader** 展示（见下节）。

---

### 5. Frank 的下一步计划

- 继续排查 **FAFB + 当前 loss** 下 **不下降** 的根因。  
- 参考另一篇 **Nature** 系方法，将 **部分 calcium / recording 条件** 并入训练（不仅是自然视频任务）；将 **flash ON/OFF** 等 **协议化刺激** 纳入 **多任务 / 多约束** 训练。

---

### 6. Jizheng 本周进展（屏幕分享 / notebook 脉络；与 `neuro_framework` 一致）

**加载与范围**  
- 在先前工作基础上扩展：**尽可能加载全量相关类型**；支持 **visual 区域 / optic lobe / 部分神经元 / 全脑** 等粒度（转录「vivo」→ **visual**，「牛肉」→ **neuron**）。  
- **FAFB optic lobe**：给出 **synapse 数量级、neuron type 数量**（转录「两百多个」保留为数量级描述，以代码为准）。  
- **默认路径**：**T4 → LC** 相关 pathway（转录「default t，four，lc」）。  
- **模型**：当前仍以 **voltage 驱动** 的简化模型为主，**HH（mcHH）** 可在 loader / notebook 中切换配置。

**形态简化（SWC / morphology）**  
- **简化 compartment / 树突结构**（转录「customer」→ **compartment**；「讲话」→ **剪掉 branch / 剪枝**）：把复杂 morphology **压到约两 compartment（两点式）** 等，并统计 **体积 / 表面积、连通性**；承认 **当前简化算法仍偏激进（删太多）或保留不足**，**后续会继续改**。

**刺激与验证笔记本**  
- **Photoreceptor step / slab current** 示例：展示 **刺激前基线 vs 给光后** 的明显差别，强调 **要等暂态过后再读响应**。  
- 另建 **flash**、**moving edge** 等文件夹做标准协议。  
- **Tuning curve**、**不同方向 motion** 等已有初步图；**moving edge 极坐标图** 的画法 Jizheng **尚需再核对**。  
- **与 FAFB 网格对齐**：真实数据 **非理想六边形网格**，已做 **normalization**；**方向 tuning 结果仍偏弱、各向差异小**，**待调参或改为训练目标优化**。

**实现备注**  
- **两 compartment** 为当前 T4 pathway 默认简化。  
- **各 neuron type 的 HH / 生理参数**：主要依 **Jizheng 以往经验** **分 type 设置**（非全体共用一套）。  
- **R1–R6、L1/L5 等** 数量级与文献 **大致对得上**，但未做系统扫参。  
- **Mi 等以 M 开头的 motion 相关类型**：**尚未全部接入**（工作量仍大）。

**工具链**  
- **Cursor 处理含大量图的 notebook 较弱**；建议 **先导出 .py 再跑** 或 **notebook 内少嵌图**。  
- **FlyWire API 偶发不可用**。

---

### 7. 公开数据推荐：钙成像 + 多刺激（frequency / tuning）

Jizheng 分享 **一篇以「frequency」为关键词的数据型论文**（结论认为 **偏片面、模型简**，但 **数据质量好**），内含 **多种 visual stimulus 下多 neuron type 的记录**，可用于：  
- 将 **model voltage → calcium-like readout** 的映射（文中或附录常给 **Ca 成像与生理量的关系**）。  

**使用建议**  
- 若暂未引入 **calcium / activity** 约束（转录「ilc」疑为 **calcium**），可 **从该数据集入门**。  
- **不要一次用光所有条件**：**一部分用于训练、一部分留作测试**，注意 **样本量仍偏少**。  
- Frank 指出：若只有 **单一 cell type** 但有 **多 column / 多方向 / 多 orientation**，仍可通过 **条件维度扩增** 有效数据量；数据里若含 **均值等统计量** 也比「只有 ±1 标签」更易对齐。

**协作**  
- Jizheng **重发/整理文档链接**（含 **T4/T5、Ca 成像、call site** 等条目，转录不清处已按「方法学与数据源目录」理解）；Frank **将链接丢到群 / Chat** 备查。

---

### 8. 协作与时间表

- Jizheng：**一两天内** 整理好 **可跑通的 notebook / 入口文件** 发给 Frank；Frank **可先** 按本次会议调整 **synapse weight 初始化与归一化、数据集检查、暂态 protocol**。  
- 若 Frank 能 **直接跑通 Jizheng 侧已 warm 过的 HH 网络**，可作为 **对照基线**（再谈 **K 通道等 HH 参数** 的训练范围）。  
- **NeurIPS 2026**：距 **Abstract（约 5 月初）** 不足一月，共识为 **尽力而为**；若赶不上，再考虑 **其他顶会**（转录未明确另两个会名，保留为「其他年会选项」）。

---

## 关键结论摘要

1. **文献**：DMN 与「需 activity 约束」类工作可在 **「多约束、多数据」** 框架下统一理解，不必简单对立。  
2. **训练卡点**：Frank 在 **FAFB + 原 DMN 式设定** 上出现 **梯度不下降**；优先排查 **BCE 饱和、W 与 synapse-count 归一化、度分布 imbalance**，而非仅改突触后 **f**。  
3. **动力学**：训练协议宜加入 **settling / warm start**，与 **flash 诊断** 对齐。  
4. **子图**：视觉子集需包含 **Lamina（L2/L3 等）** 等完整类型列表。  
5. **工程**：Jizheng 侧 **loader + 简化形态 + flash/moving edge** 已搭出雏形；**Mi 等类型未全**；**tuning 与网格对齐** 仍待加强。  
6. **数据**：推荐 **高质量钙成像 + 多刺激论文** 作 **activity 约束** 与 **voltage→calcium** 映射参考；**train/test 拆分**、**勿一次用尽条件**。  
7. **下一步**：Jizheng **整理并发入口**；Frank **调初始化/归一化/时序** 并 **试跑对方模型**。

---

## 行动项（Action Items）

| 优先级 | 任务 | 负责人 | 备注 |
|--------|------|--------|------|
| 🔴 高 | 调整 synapse weight 初始化、按度分布/总和归一化；检查数据集与 loss 尺度 | Frank | 对照本节 2、3 |
| 🔴 高 | 整理可复现的 **notebook / 脚本路径** 并发给 Frank | Jizheng | 约定「一两个晚上内」 |
| 🔴 高 | Frank **试跑 Jizheng 侧 HH 初始化网络**，确认能否作为训练基线 | Frank | 见会议末段 |
| 🟡 中 | 将 **flash ON/OFF** 与 **recording 约束** 纳入训练设计 | Frank | 对齐 `project_plan_v3` 方案 A/B |
| 🟡 中 | 继续改进 **SWC / compartment 简化**（避免删太过或保留不足） | Jizheng | |
| 🟡 中 | 核对 **moving edge 极坐标图** 绘制逻辑 | Jizheng | |
| 🟢 低 | 群文档中沉淀 **钙成像数据论文** 链接与使用注意 | 双方 | |

---

## 转录噪声对照（备查）

以下为本次逐字稿中 **明显 ASR 错误** 与 **据上下文与项目用语** 所做的还原，便于日后对照原档。

| 转录 | 推测原意 |
|------|----------|
| neuroscians | Nature Neuroscience（期刊/语境） |
| taraga | Turaga（Srinivas Turaga，Janelia 等） |
| constream | constraints |
| 量 minimal / local minimal | local minima |
| Fb | FAFB |
| Sigmoe / sequenoid | sigmoid |
| snap / cynic counter | synapse count |
| 学校 / 相当于… | 相当于 |
| wait | weight |
| 串 / 劝 | 训练 / 权重 |
| 阿宝 | α（alpha，可学习缩放） |
| sms / 改成一样 | same / 统一初值 |
| Intel | interval（时间窗刺激） |
| recruitment | recurrent（递归连接） |
| subsistence | subset |
| led三 / L1跟L3 | Lamina **L2/L3**（层别编号以文献为准） |
| vivo / 牛肉 / 胎 | visual / neuron / type |
| bank | **BANC**（全脑数据，与 FAFB 对举时） |
| password | pathway |
| customer | compartment（区室化形态） |
| 讲话 | 剪掉 / 剪枝（branch） |
| slap current | step/slab current |
| Turning curve | tuning curve |
| 背上 | **NeurIPS**（投稿会议语境） |
| m开头的…mile | **Mi** 等 motion 相关类型 |
| 年假 | **notebook**（「三个/四个 notebook」） |
| causing | calcium |
| 持沙倒过来 | 时差倒过来 |
| ilc | 疑为 **calcium**（钙成像约束），非独立脑区缩写 |

---

## 下次会议

- 未在转录中固定时间；建议 **Jizheng 发代码后 3–5 天内** 短同步 **训练是否开始下降**。  
- Sherry 与值班事宜由 Jizheng **线下协调**。

---

## 相关文献链接（2026-04-12 补充）

与本次会议「活动预测约束 / T4–T5 / LC / ON–OFF / voltage→calcium」讨论直接相关的条目如下（全文与分表见 `docs/zoom meeting/2026_0329_paper_training/project_plan_v3.md`「参考资料」）。

- **Infrequent strong connections constrain connectomic predictions of neuronal function**（*Cell* 2025）  
  https://www.cell.com/cell/fulltext/S0092-8674(25)00518-5  

- **Differential temporal filtering in the fly optic lobe**（T4/T5，*J Comput Neurosci* 2025）  
  https://link.springer.com/article/10.1007/s10827-025-00914-5  

- **Visual projection neurons in the *Drosophila* lobula link feature detection to distinct behavioral programs**（LC，*eLife*；voltage / calcium 与行为）  
  https://elifesciences.org/articles/21022  

- **Neuronal parts list and wiring diagram for a visual system**（*Nature* 2024；含 ON/OFF 等线路）  
  https://www.nature.com/articles/s41586-024-07981-1  
  - Figure 4：https://www.nature.com/articles/s41586-024-07981-1/figures/4  

- **Prediction of neural activity in connectome-constrained recurrent networks**（*Nat. Neurosci.* 2025）  
  https://www.nature.com/articles/s41593-025-02080-4  

---

*会议时间：2026年4月11日 约 21:02–21:52（以 Zoom 转录时间戳为准）*  
*整理说明：依据 `meeting_saved_closed_caption copy.txt` 与 `2026_0329` 会议摘要、`project_plan_v3.md` 对齐术语与路线*  
*版本：1.1（补充文献链接与 §1 中 Nat. Neurosci. 2025 对应关系）*
