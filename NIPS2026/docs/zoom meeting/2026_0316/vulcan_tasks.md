# 个人任务清单 - Vulcan (Alex)

## 项目概述
- **项目**: 果蝇全脑闭环控制系统
- **角色**: 行为模拟与系统集成
- **时间**: 2026年3月16日 - 2026年5月6日 (7周)
- **投入**: 4-5小时/天

---

## Week 1-2: 学习与环境搭建 (3月16日 - 3月29日)

### 主要目标
✅ 理解NeuroMechFly架构和MuJoCo基础
✅ 运行第一个示例程序
✅ 熟悉FlyWire数据结构

### 具体任务

#### Day 1-2: NeuroMechFly入门
- [ ] 阅读NeuroMechFly论文 (Nature Methods 2024)
- [ ] 查看GitHub仓库和文档
- [ ] 安装MuJoCo和相关依赖
- [ ] 运行官方示例代码
- [ ] 理解输入输出接口

**学习资源**:
- 论文: https://www.nature.com/articles/s41592-024-02497-y
- 代码: [GitHub链接]
- MuJoCo文档: https://mujoco.readthedocs.io/

#### Day 3-4: MuJoCo深入学习
- [ ] 学习MuJoCo基础概念
- [ ] 理解物理引擎参数
- [ ] 尝试修改简单参数
- [ ] 运行不同的示例场景
- [ ] 记录关键API和函数

**重点理解**:
- 物理模拟循环
- 传感器和执行器
- 状态空间和动作空间
- 渲染和可视化

#### Day 5-7: FlyWire数据探索
- [ ] 访问FlyWire网站
- [ ] 理解数据集结构
- [ ] 下载示例数据
- [ ] 阅读数据文档
- [ ] 理解神经元类型分类

**需要下载的数据**:
- Cell Types (consolidated_cell_types.csv.gz)
- Classification (classification.csv.gz)
- Connections (connections_princeton.csv.gz)
- Visual Neuron Annotations (visual_neuron_types.csv.gz)

#### Day 8-10: 筛选Motion相关神经元
- [ ] 理解视觉-运动通路
- [ ] 筛选相关神经元类型
- [ ] 提取连接信息
- [ ] 创建神经元子集
- [ ] 验证数据完整性

**关键神经元类型**:
- Visual neurons (optic lobe)
- Motor neurons
- Descending neurons
- Central complex neurons

#### Day 11-14: 整合与测试
- [ ] 编写数据加载脚本
- [ ] 测试NeuroMechFly不同配置
- [ ] 记录遇到的问题
- [ ] 准备Week 2总结报告
- [ ] 与Jizheng同步进展

**Week 1-2 交付物**:
- ✅ 可运行的NeuroMechFly环境
- ✅ FlyWire数据下载和初步处理脚本
- ✅ 学习笔记和问题清单
- ✅ 进度报告

---

## Week 3-4: 数据处理与接口设计 (3月30日 - 4月12日)

### 主要目标
✅ 完成FlyWire数据处理
✅ 设计DMN-NeuroMechFly接口
✅ 实现基础数据流

### 具体任务

#### Day 15-17: 深度数据处理
- [ ] 解析神经元连接矩阵
- [ ] 提取突触权重
- [ ] 处理神经递质信息
- [ ] 创建神经元-肌肉映射
- [ ] 数据格式转换

**数据处理重点**:
- 连接矩阵稀疏化
- 权重归一化
- 神经元ID映射
- 数据验证

#### Day 18-21: 接口设计
- [ ] 定义DMN输出格式
- [ ] 定义NeuroMechFly输入格式
- [ ] 设计转换函数
- [ ] 实现数据转换层
- [ ] 单元测试

**接口规范**:
```python
# DMN输出: 神经元激活值
dmn_output = {
    'neuron_ids': [...],
    'activations': [...],
    'timestamp': ...
}

# NeuroMechFly输入: 肌肉控制信号
nmf_input = {
    'muscle_ids': [...],
    'control_signals': [...],
    'timestamp': ...
}
```

#### Day 22-25: 模拟数据测试
- [ ] 生成模拟DMN输出
- [ ] 测试数据转换
- [ ] 运行NeuroMechFly with模拟输入
- [ ] 验证行为输出
- [ ] 调试和优化

#### Day 26-28: 文档和准备
- [ ] 编写接口文档
- [ ] 准备集成测试计划
- [ ] 与Jizheng确认DMN输出格式
- [ ] Week 3-4总结报告

**Week 3-4 交付物**:
- ✅ 处理后的FlyWire数据集
- ✅ 接口设计文档
- ✅ 数据转换代码
- ✅ 测试报告

---

## Week 5-6: 系统集成 (4月13日 - 4月26日)

### 主要目标
✅ 集成DMN模型输出
✅ 构建闭环系统
✅ 实现反馈机制

### 具体任务

#### Day 29-32: DMN集成
- [ ] 接收Jizheng的DMN模型输出
- [ ] 实现实时数据转换
- [ ] 连接DMN和NeuroMechFly
- [ ] 测试单向数据流
- [ ] 性能优化

**集成架构**:
```
Visual Input → DMN Model → Neural Activations 
    ↓
Conversion Layer → Muscle Signals → NeuroMechFly
    ↓
Behavior Output → Sensory Feedback → (loop back)
```

#### Day 33-36: 闭环实现
- [ ] 实现感觉反馈
- [ ] 连接视觉输入
- [ ] 构建完整循环
- [ ] 测试闭环稳定性
- [ ] 调整参数

**闭环关键点**:
- 时间步同步
- 延迟处理
- 状态管理
- 错误处理

#### Day 37-40: 测试和调试
- [ ] 运行完整闭环测试
- [ ] 记录行为输出
- [ ] 分析系统性能
- [ ] 识别瓶颈
- [ ] 优化代码

#### Day 41-42: 文档和演示
- [ ] 编写系统文档
- [ ] 准备演示视频
- [ ] Week 5-6总结报告
- [ ] 与团队分享结果

**Week 5-6 交付物**:
- ✅ 可运行的闭环系统
- ✅ 系统架构文档
- ✅ 测试结果和日志
- ✅ 演示视频

---

## Week 7: 优化与论文支持 (4月27日 - 5月6日)

### 主要目标
✅ 系统优化
✅ 实验数据收集
✅ 支持论文撰写

### 具体任务

#### Day 43-45: 性能优化
- [ ] 代码重构
- [ ] 性能profiling
- [ ] 优化瓶颈
- [ ] 提高稳定性
- [ ] 最终测试

#### Day 46-48: 实验和数据
- [ ] 运行多组实验
- [ ] 收集性能数据
- [ ] 生成可视化
- [ ] 对比baseline
- [ ] 整理结果

#### Day 49: 论文支持
- [ ] 准备方法部分内容
- [ ] 提供系统图表
- [ ] 协助结果分析
- [ ] Review论文草稿

**Week 7 交付物**:
- ✅ 优化后的系统
- ✅ 实验数据和图表
- ✅ 方法部分文字
- ✅ 最终代码

---

## 技术栈和工具

### 编程语言
- Python 3.8+
- 可能需要C++（MuJoCo扩展）

### 主要库
- MuJoCo
- NumPy, SciPy
- Pandas (数据处理)
- Matplotlib, Seaborn (可视化)
- PyTorch/TensorFlow (如需要)

### 开发工具
- VS Code / PyCharm
- Jupyter Notebook
- Git / GitHub
- Claude / GPT (AI辅助)

### 数据工具
- CSV处理
- HDF5 (大数据)
- 可能需要数据库

---

## 学习资源

### 必读论文
1. NeuroMechFly (Nature Methods 2024)
2. FlyWire connectome papers
3. RF Model (Shiu et al., Nature 2024)
4. DMN Model (Nature 2024)

### 在线资源
- MuJoCo官方文档
- FlyWire数据门户
- 相关GitHub仓库
- 神经科学基础教程

### AI辅助
- 使用Claude理解复杂概念
- 使用GPT生成代码框架
- 使用AI调试问题

---

## 问题和风险

### 技术问题
- [ ] MuJoCo安装和配置
- [ ] 数据格式兼容性
- [ ] 性能瓶颈
- [ ] 接口不匹配

### 解决策略
- 提前测试和验证
- 保持与Jizheng沟通
- 准备备选方案
- 及时寻求帮助

---

## 沟通计划

### 与Jizheng
- **频率**: 每2-3天
- **内容**: 
  - 进度更新
  - 技术问题讨论
  - 接口确认
  - 数据交换

### 团队会议
- **每周例会**: 周一或周五
- **准备内容**:
  - 本周完成任务
  - 下周计划
  - 遇到的问题
  - 需要的支持

---

## 自我评估

### 每周检查
- [ ] 是否完成计划任务？
- [ ] 遇到什么困难？
- [ ] 需要调整计划吗？
- [ ] 学到了什么？

### 关键指标
- 代码提交频率
- 功能完成度
- 测试覆盖率
- 文档完整性

---

## 备注

- **灵活性**: 根据实际情况调整任务优先级
- **求助**: 遇到困难及时沟通，不要拖延
- **记录**: 保持良好的笔记和文档习惯
- **效率**: 充分利用AI工具提高生产力

---

**创建日期**: 2026年3月16日
**最后更新**: 2026年3月16日
**负责人**: Vulcan (Alex)
