# 项目计划 v2.0 - 果蝇全脑闭环控制系统

> 基于 2026年3月22日会议更新  
> 上一版本：`docs/zoom meeting/2026_0316/project_plan.md`

---

## 项目概述

**项目名称**: 基于果蝇全脑连接组的闭环神经行为控制系统  
**目标会议**: NeurIPS 2026

| 里程碑 | 日期 |
|--------|------|
| Abstract 截止 | 2026年5月4日 |
| Full Paper 截止 | 2026年5月6日 |
| 距今剩余时间 | ~6.5 周 |

**核心目标**：
- 使用 DMN（Connectome-constrained network）替换 RF 模型中的神经网络部分
- 整合果蝇 social behavior 行为数据（39关键点 3D 坐标）
- 实现视觉输入 → 神经网络 → 运动控制 的端到端系统
- 探索与 NeuroMechFly body model 的联合训练方案

---

## 团队成员与分工

### Jizheng Dong（五竹）— 项目负责人 / 神经模型

**背景**：计算神经科学，开发了 multi-compartment Hodgkin-Huxley 模型

**主要职责**：
1. **DMN 模型维护与指导**
   - 维护并完善可训练的 DMN 神经网络模型
   - 将视觉输入（optical flow）映射到神经活动
   - 指导 Frank 接手和修改模型代码

2. **行为数据提供**
   - 整理 social behavior 场景下的 39 关键点 3D 时序数据
   - 进行降维处理，将行为分类（grooming, local motion, wing extension, turning 等）
   - 提供标注好的数据集给团队使用

3. **论文规划与技术决策**
   - 确定最终技术路线（body model 联合训练方案）
   - 参考清华大学相关 paper，研究其 body model 训练方法
   - 负责论文主体撰写和技术架构决策

4. **资源协调**
   - 申请 GPU 计算资源（联系合作老师）
   - 分享 Claude/AI API 给团队成员
   - 发 when2meet 安排下周会议时间

**本周任务**：
- [ ] 将 DMN 模型代码和相关资料 share 给 Frank
- [ ] 与 Frank 进行代码对接（明天或本周内）
- [ ] 发 when2meet 确定下周会议时间

---

### Vulcan Z（Alex / Alice）— 行为模拟与系统集成

**背景**：已成功运行 NeuroMechFly，完成初步 FlyWire 数据下载

**主要职责**：
1. **NeuroMechFly 深度研究**
   - 完全搞清楚 body model 的输入输出接口
   - 研究如何通过编程控制各种具体动作（左转、右转、翅膀动作等）
   - 理解 MuJoCo 模型文件结构

2. **Body Model 训练可行性研究**（关键问题）
   - 研究在加入 MuJoCo body model 后，如何进行梯度反传
   - 参考清华 paper 中使用 body model 进行训练的方法
   - 评估 differentiable simulation 或 RL 方案的可行性

3. **系统集成**
   - 将 DMN 输出（运动控制信号）与 NeuroMechFly 接口对接
   - 构建端到端训练流程
   - 与 Sherry 协作完成 NeuroMechFly 接口研究

4. **FlyWire 数据处理**
   - 已筛选约 100 个运动相关神经元（FAFB v783）
   - 继续完善神经元选择和数据处理流程

**近期里程碑**：
- [ ] 本周：完整文档化 NeuroMechFly 输入输出格式
- [ ] 本周：调研 body model + neural network 联合训练方案
- [ ] 下周：实现 DMN 输出 → NeuroMechFly 控制信号的接口
- [ ] Week 4-5：完成基础闭环系统第一版

**每日投入**：4-5 小时，充分使用 Claude/Cursor 加速开发

---

### Yi-De Tai（Frank）— 神经网络模型开发（新成员）

**背景**：BCI 研究（墨尔本大学博士），熟悉神经信号处理，有 connectome 相关经验，人在台湾

**主要职责**：
1. **接手和扩展 DMN 模型**
   - 接收 Jizheng 的代码和资料
   - 理解并在此基础上进行修改和扩展
   - 负责模型的进一步开发和优化

2. **神经网络训练**
   - 参与设计训练策略
   - 利用行为数据训练模型参数
   - 调试和优化训练过程

3. **Background 贡献**
   - 利用台湾团队的 connectome 相关联系
   - BCI 经验有助于理解神经信号的编码和解码

**入门任务**（本周）：
- [ ] 接收 Jizheng share 的代码和资料
- [ ] 在 B 站学习相关背景知识（LIF/HH 模型原理）
- [ ] 阅读 DMN paper（Connectome-constrained networks, Nature 2024）
- [ ] 与 Jizheng 进行代码对接会议

**后续任务**：
- [ ] Week 2-3：理解并运行 DMN 代码
- [ ] Week 3-4：在代码基础上进行扩展修改
- [ ] Week 5-6：配合训练流程，优化模型

---

### Sherry — 神经仿真模型研究（新成员）

**背景**：SNN（脉冲神经网络）专家，计算机背景，现为软件工程师

**主要职责**：
1. **NeuroMechFly 代码研究**
   - 深入研究 NeuroMechFly 的 GitHub 代码（不只是 README）
   - 阅读其技术文档（blog、technical docs）
   - 理解 ecosystem 运行逻辑

2. **SNN 模型贡献**
   - 利用 SNN 专业知识，协助神经仿真部分
   - 研究 LIF 模型与 HH 模型的接口兼容性
   - 协助实现可扩展的神经元仿真模块

3. **可视化开发**（潜在方向）
   - 利用软件工程背景开发可视化工具
   - 展示神经元连接、电压变化、行为输出等

**入门任务**（本周）：
- [ ] 深入阅读 NeuroMechFly GitHub 技术文档
- [ ] 理解 NeuroMechFly 的完整输入输出接口
- [ ] 参考蔡老师团队代码，理解 LIF 模型与 NeuroMechFly 的关系
- [ ] 尝试运行 NeuroMechFly ecosystem

**后续任务**：
- [ ] Week 2-3：完成 NeuroMechFly 接口文档整理
- [ ] Week 3-4：协助实现神经仿真与 body model 的接口
- [ ] Week 5+：根据团队需求，负责可视化或仿真模块

---

## 技术架构

```
[Social Behavior Scene]
        ↓
[3D Pose Reconstruction (39 keypoints)]
        ↓
[Visual Input Generation (MuJoCo rendering)]
        ↓
[DMN Neural Network (FlyWire connectome-constrained)]
   - 可训练参数：突触权重、神经递质参数
   - 约束：Dale's principle、稀疏性
        ↓
[Motor Decoder → Muscle Signals]
        ↓
[NeuroMechFly Body Model (MuJoCo)]
        ↓
[Predicted Behavior (39 keypoints)]
        ↓
[Loss: MPJPE + velocity + physics + neural]
        ↓
[Backprop → Update DMN parameters]
        ↑________________________|
              (closed loop)
```

### 关键技术问题（待解决）

| 问题 | 状态 | 负责人 |
|------|------|--------|
| NeuroMechFly 输入输出接口格式 | 进行中 | Vulcan + Sherry |
| Body model 联合训练（梯度反传）| 待调研 | Vulcan |
| DMN 模型与行为数据的对接 | 待开始 | Frank + Jizheng |
| 清华 paper 训练方法参考 | 待阅读 | 全体 |
| GPU 资源申请 | 待进行 | Jizheng |

---

## 整体时间线

```
3月22日 ─── 4月5日 ─── 4月19日 ─── 5月4日 ─── 5月6日
   │             │             │            │           │
 当前          Phase 1       Phase 2     Abstract    Full Paper
               完成          系统集成      截止         截止
```

### Phase 1（3月22日 - 4月5日）：基础搭建
- Vulcan：NeuroMechFly 接口完全搞清楚，body model 训练方案调研
- Frank：接手 Jizheng 代码，成功运行 DMN 模型
- Sherry：NeuroMechFly 代码深度研究，整理接口文档
- Jizheng：代码 share，行为数据整理，参考清华 paper

### Phase 2（4月5日 - 4月19日）：系统集成
- DMN 输出 → NeuroMechFly 控制信号接口实现
- 端到端训练流程第一版
- 申请并获得 GPU 资源
- 初步训练结果

### Phase 3（4月19日 - 5月4日）：训练与优化
- 正式训练，调整超参数
- 与 RF baseline 对比
- 实验结果分析
- Abstract 撰写提交

### Phase 4（5月4日 - 5月6日）：论文完成
- 完成 Full Paper 撰写
- 图表和可视化
- 最终提交

---

## 风险管理

| 风险 | 可能性 | 影响 | 应对方案 |
|------|--------|------|----------|
| Body model 无法梯度反传 | 高 | 高 | 改用 RL 方案，或跳过 body model 先做 feedforward |
| GPU 资源申请延迟 | 中 | 高 | 先用小规模模型验证，必要时用云服务 |
| 团队新成员上手慢 | 中 | 中 | 提前 share 资料，安排 1-on-1 对接 |
| 时间不够 | 高 | 高 | 优先完成核心流程，简化实验，考虑投 Workshop |
| NeurIPS 未录用 | 中 | 中 | 准备 ICLR 2027 或其他顶会备选 |

---

## 沟通机制

### 定期会议
- **每周例会**：根据 when2meet 确定时间（Jizheng 会提前发链接）
- **代码对接**：Jizheng + Frank，本周内安排
- **紧急讨论**：随时，微信群

### 工具与资源

| 用途 | 工具 |
|------|------|
| 即时通讯 | 微信群 |
| 代码协作 | GitHub |
| 会议 | Zoom |
| AI 辅助开发 | Cursor + Claude API（Jizheng 分享）|
| 文档 | 本项目 docs 目录 |

### AI 工具使用
- Jizheng 可分享 Claude API 给大家
- 淘宝共享 Cursor 会员方案约 26-28 元/月/人
- 推荐通过 Cursor 调用 Claude/GPT，大幅加速开发

---

## 参考资料

### 关键论文
| 论文 | 重要性 | 负责阅读 |
|------|--------|----------|
| Connectome-constrained networks, Nature 2024 (DMN) | 🔴 核心 | 全体 |
| Shiu et al., Nature 2024 (RF Model) | 🔴 核心 | 全体 |
| NeuroMechFly, Nature Methods 2024 | 🔴 核心 | Vulcan + Sherry |
| 清华大学 body model 训练相关 paper | 🔴 关键 | 全体（下周前） |
| FlyWire FAFB v783 数据集论文 | 🟡 重要 | Vulcan |

### 数据与代码
- **FlyWire**: https://codex.flywire.ai/
- **NeuroMechFly GitHub**: [需查找链接，查看技术文档]
- **DMN 代码**: Jizheng share
- **行为数据**: Jizheng 提供

---

## 成功标准

### 最低标准 (Must Have)
- [ ] 成功将 DMN 输出与 NeuroMechFly 接口对接
- [ ] 实现基础的 feedforward 预测（视觉输入 → 行为输出）
- [ ] 系统性能优于 RF baseline
- [ ] 完成论文并提交（NeurIPS 或 Workshop）

### 期望标准 (Should Have)
- [ ] 实现完整的闭环训练流程
- [ ] 行为重建误差（MPJPE）显著低于 baseline
- [ ] 神经生物学约束（Dale's principle、稀疏性）得到满足
- [ ] 论文被 NeurIPS 主会接收

### 理想标准 (Nice to Have)
- [ ] 实现 Vulcan 提出的精细目标导向控制
- [ ] 多种 social behavior 场景的泛化验证
- [ ] 精美的神经活动可视化
- [ ] 代码开源，推动领域发展

---

*最后更新：2026年3月22日*  
*版本：2.0（新增 Frank、Sherry 两位成员）*