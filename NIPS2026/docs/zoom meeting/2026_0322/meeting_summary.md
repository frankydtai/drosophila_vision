# 会议纪要 - 2026年3月22日

## 参会人员

| 姓名 | 角色 | 背景 |
|------|------|------|
| **Jizheng Dong (五竹)** | 项目负责人 | 神经科学/计算神经科学 |
| **Alex (Vulcan)** | 行为模拟 | 已有 NeuroMechFly 运行经验 |
| **Yi-De Tai (Frank)** | 新成员 | PhD @ 墨尔本大学，BCI研究（羊脑植入），希望转型 Neural×AI |
| **Sherry (Xuyi)** | LIF | 本科+硕士，脉冲神经网络(SNN)背景，软件工程师，计划申请博士 |

---

## 新成员自我介绍

### Yi-De Tai (Frank)
- 台湾人，人在台湾
- 本科在美国就读，博士在墨尔本大学
- 研究方向：BCI（脑机接口），将电极植入羊脑，做低频神经信号研究
- 转型动机：AI 进展远快于 hardware，希望在 neuroscience 和 AI 之间找到平衡点
- 曾与台湾团队合作 connectome 相关研究，早期做过 FlyWire connector

### Sherry (Xuyi)
- 计算机背景，本科+硕士
- 专业方向：脉冲神经网络 (Spiking Neural Network, SNN)
- 目前在做软件工程工作，计划申请博士
- 认为 SNN 与连接组研究有相通之处，对该方向很感兴趣

---

## 核心技术讨论

### 1. DMN 模型关键结论介绍（Jizheng 讲解）

**Teacher-Student Network 实验结论**：
- 创建两种网络：稠密连接（接近全连接）vs 稀疏连接
- 将 teacher 网络训练好后，给 student 网络部分信息（连接方式、突触强度符号等）
- 重新训练 student 网络，比较其参数与 teacher 的相似度
- **关键发现**：网络越稀疏 → 重训练后参数相似度越高
- **实际意义**：果蝇大脑本身是稀疏连接的，这对我们的训练策略有重要启示
- 已知信息越多（连接+符号+突触强度）→ 训练结果与真实越相近

**DMN 模型架构回顾**：
- 类似卷积核结构，从上层传递到下层
- 红色标注的参数（突触权重）是可训练的
- 输入：视频；输出：optical flow（光流/运动方向）
- 与 CNN 的区别：有求导项 + synapse 连接部分

### 2. 现有行为数据介绍（Jizheng）

**实验场景**：
- Social behavior：两只雌性果蝇 + 一只雄性果蝇
- 雄性果蝇追逐雌性，做求偶行为
- 已采集 39 个关键点的 3D 坐标时序数据

**行为建模**：
- 已做降维处理，将时间序列行为降维为点
- 可观测到多种行为：梳理 (grooming)、局部运动 (local motion)、翅膀展开 (wing extension)、向左/右转等
- 计划将行为数据与神经网络模型对接（line up）

### 3. Alex (Vulcan) 进度汇报

**已完成**：
- 成功运行 NeuroMechFly 基础示例
- 已有 MuJoCo 模型文件
- 从 FlyWire FAFB v783 下载数据
- 从数据中筛选了约 100 个神经元（带有运动相关标注的类别）

**NeuroMechFly 接口发现**：
- 有 API，但目前 API 主要用于数据抓取
- 控制方式：写测试文件，通过 MuJoCo 模型接口控制
- 需要进一步研究如何通过编程控制具体动作

**Alex (Vulcan) 的想法（讨论）**：
- 提议：让果蝇通过视觉感知目标物体，然后主动控制工具去捕获目标
- 超越现有论文只做简单移动的局限，实现精细的 closed-loop 控制
  - **Jizheng 反馈**：
  - 这个方向很有趣，但实现难度很大
  - Closed-loop 意味着非常长的 time series，时间尺度大，训练复杂度高
  - 现有 Nature 论文也只做到了对应行为（feedforward），没有实现完整的 feedback
  - 建议先把基础跑通，再看有没有好的结论
  - Yi-De 补充：即便是 Nature 顶刊也只是对应行为，要做 feedback 是前沿挑战

### 4. Sherry 模型演示（LIF 模型代码走读）

**Sherry 正在研究的模型（蔡老师团队）**：
- Leaky Integrate-and-Fire (LIF) 模型
- 将脉冲 (spike) 转化为频率/发放率
- 泊松分布 (Poisson distribution) 生成输入时间序列

**关键技术讨论（Jizheng 讲解 LIF vs Hodgkin-Huxley）**：
- LIF 模型的衰减问题：信号经过多层传递后急剧衰减
  - LIF 中衰减由公式本身性质决定，难以简单调参解决
  - 传几层后，100Hz 输入几乎消失
- Jizheng 自己的模型（multi-compartment HH）的优势：
  - Synapse 释放概率由 presynaptic 决定
  - Post-synapse 接收信号后有主动的 spike 过程
  - 只需将膜电位拽过阈值，神经元会自主继续上升（像大坝蓄水后自然溢出）
  - 信号可以多层传递，不会像 LIF 那样快速衰减
- **I_silence 实验**：
  - 给网络加入 silence（抑制电流）后观察神经元发放率变化
  - 结果：有些神经元率降低，有些反而增加（代偿效应）
  - 网络连接复杂，具体机制待进一步分析

### 5. 任务分工讨论（会议后半段）

**接下来一周的任务**：

| 成员 | 任务 |
|------|------|
| **Jizheng** | 下周比较忙；将代码 share 给 Frank；之后与 Frank 对接 |
| **Frank (Yi-De)** | 接收 Jizheng 的代码和资料，在此基础上修改；建议先看 B 站相关教程了解原理 |
| **Sherry (Xuyi)** | 继续研究 NeuroMechFly 模型，看懂所有输入输出接口；查看 GitHub 技术文档（不止 README） |
| **Alex (Vulcan)** | 继续研究 NeuroMechFly body model，搞清楚如何控制各种动作；研究 body model 如何与训练结合 |

**关键问题（待解决）**：
- 加入 body model (MuJoCo) 后训练如何进行？微分是否可行？
- 参考清华大学的相关 paper，看他们如何用 body model 做训练
- Cursor/Claude API 共享使用方案（Jizheng 分享 API key，或共享账号，淘宝28元/30天3人共享）

### 6. AI 工具使用讨论

- Alex (Vulcan) 正在使用 Claude（已购买包年）
- Jizheng 有 Claude API，可以分享给大家使用
- 可以通过 Cursor 调用 Claude/GPT 等不同模型的 API
- 淘宝有共享会员方案（约 26-28 元/30天/3人）
- 建议：如果共享不能用就退货，改用 API

---

## 关键结论总结

1. **团队扩充**：新加入 Yi-De Tai (Frank)  
2. **Alex (Vulcan) 已成功运行** NeuroMechFly 基础示例，已有初步成果
3. **稀疏网络训练结论**：果蝇连接组稀疏性是优势，已知约束信息越多，训练结果越接近真实
4. **LIF vs HH**：Jizheng 的 HH 模型在多层信号传递上优于简单 LIF
5. **主要技术难点**：body model 与 neural network 的联合训练，closed-loop 的 time series 问题
6. **参考对象**：清华大学相关 paper（使用 body model 做训练的方法）

---

## 下一步行动

| 优先级 | 任务 | 负责人 | 时间 |
|--------|------|--------|------|
| 🔴 高 | 将代码和资料 share 给 Frank | Jizheng | 本周内 |
| 🔴 高 | 研究清华 paper 的 body model 训练方法 | 全体 | 下周前 |
| 🔴 高 | 搞清楚 NeuroMechFly 输入输出接口和控制方式 | Alex (Vulcan) + Sherry (Xuyi) | 下周前 |
| 🟡 中 | Frank 在 Jizheng 代码基础上修改运行 | Frank | 下周 |
| 🟡 中 | 调查如何让 body model 支持梯度反传（训练） | Alex (Vulcan) | 下周 |
| 🟢 低 | 确定 Cursor/Claude API 共享方案 | 全体 | 本周 |

---

## 下次会议

- Jizheng 将发 **when2meet** 链接，根据大家时间安排
- Frank 周六（国内时间）有空，周日没空
- 预计周四/周五先沟通一次

---

*会议时间：2026年3月22日 22:03 - 23:55*   
*版本：1.0*