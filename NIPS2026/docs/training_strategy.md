# 模型训练策略

## 问题定义

### 输入数据
1. **视觉输入**: 
   - 来源：Social behavior中多只果蝇的3D pose重构
   - 处理：通过MuJoCo模型模拟每只果蝇的视觉输入
   - 本质：从其他果蝇的3D pose生成当前果蝇的视网膜图像

2. **行为数据**:
   - 39个关键点的3D坐标
   - 时间序列数据（已降维，~100帧/点）
   - 包含多只果蝇的交互行为

### 输出目标
- 预测果蝇的运动行为
- 控制NeuroMechFly模型产生相应动作
- 实现闭环控制

---

## 训练框架设计

### 整体架构

```
Social Behavior Scene (多只果蝇)
    ↓
[3D Pose Reconstruction] → 其他果蝇的3D poses
    ↓
[Visual Input Generation] → 当前果蝇的视觉输入 (通过MuJoCo渲染)
    ↓
[DMN Model] → 神经元激活模式
    ↓
[Motor Control] → 肌肉控制信号
    ↓
[NeuroMechFly] → 预测的行为输出
    ↓
[Loss Computation] ← 真实的行为数据 (39个关键点)
```

---

## 可训练参数

### 1. DMN模型参数

#### 1.1 突触权重 (Synaptic Weights)
```python
# 连接矩阵的权重
W_syn = {
    'visual_to_central': trainable,  # 视觉→中央复合体
    'central_to_motor': trainable,   # 中央复合体→运动神经元
    'recurrent': trainable,          # 循环连接
}
```

**为什么训练**:
- FlyWire提供的是连接拓扑（谁连谁）
- 但突触强度（权重）需要学习
- 不同行为可能需要不同的权重配置

**约束条件**:
- 保持连接拓扑不变（有连接的地方才能有权重）
- 可以加入生物学约束（如Dale's principle：兴奋性/抑制性）
- 权重范围限制（避免过大或过小）

#### 1.2 神经递质参数
```python
# 神经递质相关参数
NT_params = {
    'release_probability': trainable,  # 释放概率
    'time_constant': trainable,        # 时间常数
    'reversal_potential': fixed,       # 反转电位（可固定）
}
```

**为什么训练**:
- 调节突触传递的动态特性
- 影响网络的时间响应特性

#### 1.3 神经元参数（如果使用复杂模型）
```python
# 如果使用Hodgkin-Huxley类模型
Neuron_params = {
    'leak_conductance': trainable,     # 漏电导
    'threshold': trainable,            # 阈值
    'time_constant': trainable,        # 时间常数
}
```

**注意**: 如果参数太多，可能难以训练，建议从简单开始

---

### 2. 视觉处理参数

#### 2.1 视觉编码器
```python
# 将渲染的视觉输入转换为神经元激活
Visual_encoder = {
    'receptive_fields': trainable,     # 感受野参数
    'gain': trainable,                 # 增益
    'adaptation': trainable,           # 适应参数
}
```

**为什么需要**:
- 原始视觉输入需要转换为视觉神经元的激活
- 可能需要学习如何从场景中提取相关特征

#### 2.2 注意力机制（可选）
```python
# 关注场景中的重要部分
Attention = {
    'spatial_weights': trainable,      # 空间注意力
    'feature_weights': trainable,      # 特征注意力
}
```

---

### 3. 运动解码器参数

#### 3.1 神经元到肌肉的映射
```python
# 将运动神经元激活转换为肌肉控制信号
Motor_decoder = {
    'neuron_to_muscle_weights': trainable,  # 神经元→肌肉权重
    'activation_function_params': trainable, # 激活函数参数
}
```

**为什么需要**:
- 神经元激活和肌肉控制信号之间需要映射
- 这个映射可能不是简单的线性关系

---

## 训练目标和损失函数

### 主要目标
**预测未来的行为轨迹，使其与真实行为数据匹配**

### 损失函数设计

#### 1. 行为重构损失 (Behavior Reconstruction Loss)
```python
# L1: 关键点位置损失
L_keypoint = MSE(predicted_keypoints, true_keypoints)

# 权重不同的关键点
weights = {
    'head': 2.0,      # 头部更重要
    'legs': 1.5,      # 腿部
    'body': 1.0,      # 身体
}

L_weighted_keypoint = Σ(weights[i] * MSE(pred[i], true[i]))
```

#### 2. 速度和加速度损失
```python
# L2: 运动平滑性
L_velocity = MSE(predicted_velocity, true_velocity)
L_acceleration = MSE(predicted_acceleration, true_acceleration)
```

**为什么需要**:
- 只匹配位置可能导致不自然的运动
- 速度和加速度约束保证运动的平滑性

#### 3. 物理约束损失
```python
# L3: 物理可行性
L_physics = {
    'joint_limits': penalty_for_invalid_joint_angles,
    'ground_contact': penalty_for_penetration,
    'balance': penalty_for_unstable_poses,
}
```

#### 4. 神经生物学约束损失（可选）
```python
# L4: 神经活动的生物学合理性
L_neural = {
    'firing_rate': penalty_for_unrealistic_rates,
    'sparsity': L1_regularization_on_activations,
    'dale_principle': penalty_for_violating_excitatory_inhibitory,
}
```

#### 5. 总损失
```python
L_total = α₁*L_keypoint + α₂*L_velocity + α₃*L_acceleration + 
          α₄*L_physics + α₅*L_neural

# 超参数
α₁ = 1.0   # 位置最重要
α₂ = 0.5   # 速度
α₃ = 0.2   # 加速度
α₄ = 0.3   # 物理约束
α₅ = 0.1   # 神经约束
```

---

## 训练策略

### 阶段1: 预训练 (Week 3-4)

#### 目标
学习基本的视觉-运动映射

#### 数据
- 使用简单场景（单只果蝇或简单交互）
- 短时间序列（10-20帧）

#### 训练参数
- 只训练突触权重
- 固定其他参数

#### 损失
- 主要关注 L_keypoint
- 简单的物理约束

```python
# 伪代码
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        visual_input = generate_visual_input(batch['other_flies_poses'])
        neural_activity = DMN(visual_input)
        muscle_signals = motor_decoder(neural_activity)
        predicted_behavior = NeuroMechFly(muscle_signals)
        
        # 计算损失
        loss = MSE(predicted_behavior, batch['true_behavior'])
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

---

### 阶段2: 精细调整 (Week 5)

#### 目标
提高预测精度，加入更多约束

#### 数据
- 复杂场景（多只果蝇交互）
- 较长时间序列（50-100帧）

#### 训练参数
- 突触权重
- 视觉编码器参数
- 运动解码器参数

#### 损失
- 完整的损失函数
- 加入速度、加速度约束
- 物理约束

---

### 阶段3: 闭环训练 (Week 6)

#### 目标
实现真正的闭环控制

#### 方法
```python
# 闭环训练
for step in range(max_steps):
    # 1. 当前状态
    current_state = get_current_state()
    
    # 2. 生成视觉输入（基于其他果蝇的位置）
    visual_input = render_visual_scene(other_flies_poses)
    
    # 3. 模型预测
    neural_activity = DMN(visual_input, current_state)
    action = motor_decoder(neural_activity)
    
    # 4. 执行动作
    next_state = NeuroMechFly.step(action)
    
    # 5. 计算奖励/损失
    reward = compute_reward(next_state, target_behavior)
    
    # 6. 更新模型
    update_model(reward)
```

#### 特点
- 模型的输出会影响下一步的输入
- 需要考虑长期依赖
- 可能需要强化学习技术

---

## 数据处理流程

### 1. 视觉输入生成

```python
def generate_visual_input(scene_data, focal_fly_id):
    """
    从social behavior场景生成单只果蝇的视觉输入
    
    Args:
        scene_data: 包含所有果蝇3D poses的数据
        focal_fly_id: 当前关注的果蝇ID
    
    Returns:
        visual_input: 视网膜图像或视觉神经元激活
    """
    # 1. 获取其他果蝇的3D poses
    other_flies = [fly for fly in scene_data if fly.id != focal_fly_id]
    
    # 2. 获取focal fly的位置和朝向
    focal_pose = scene_data[focal_fly_id]
    camera_pos = focal_pose.head_position
    camera_orientation = focal_pose.head_orientation
    
    # 3. 在MuJoCo中设置场景
    scene = setup_mujoco_scene(other_flies)
    
    # 4. 渲染视觉输入
    # 选项A: 渲染RGB图像
    rgb_image = render_from_fly_perspective(
        scene, camera_pos, camera_orientation
    )
    
    # 选项B: 直接计算视觉神经元激活
    # 基于其他果蝇的位置、大小、运动等
    visual_features = compute_visual_features(
        other_flies, focal_pose
    )
    
    return visual_features
```

### 2. 行为数据预处理

```python
def preprocess_behavior_data(raw_data):
    """
    预处理行为数据
    """
    # 1. 归一化关键点坐标
    normalized_keypoints = normalize_keypoints(raw_data.keypoints)
    
    # 2. 计算速度和加速度
    velocity = compute_velocity(normalized_keypoints)
    acceleration = compute_acceleration(velocity)
    
    # 3. 降维（如果需要）
    reduced_data = dimensionality_reduction(
        normalized_keypoints, n_frames_per_point=100
    )
    
    # 4. 分割训练/验证/测试集
    train, val, test = split_data(reduced_data)
    
    return {
        'keypoints': normalized_keypoints,
        'velocity': velocity,
        'acceleration': acceleration,
        'reduced': reduced_data,
        'splits': (train, val, test)
    }
```

---

## 评估指标

### 1. 行为重构精度
```python
# 关键点位置误差
MPJPE = Mean_Per_Joint_Position_Error(predicted, true)

# 不同身体部位的误差
errors = {
    'head': MPJPE(predicted.head, true.head),
    'thorax': MPJPE(predicted.thorax, true.thorax),
    'legs': MPJPE(predicted.legs, true.legs),
    'wings': MPJPE(predicted.wings, true.wings),
}
```

### 2. 运动质量
```python
# 运动平滑性
smoothness = compute_jerk(predicted_trajectory)

# 物理可行性
physics_score = check_physics_constraints(predicted_behavior)

# 与真实行为的相似度
similarity = compute_trajectory_similarity(predicted, true)
```

### 3. 神经活动合理性
```python
# 神经元激活率
firing_rates = compute_firing_rates(neural_activity)

# 激活模式的稀疏性
sparsity = compute_sparsity(neural_activity)

# 与已知神经生理学数据的一致性
consistency = compare_with_physiology_data(neural_activity)
```

---

## 最终目标

### 短期目标 (NeurIPS 2026)

#### 定量目标
1. **行为重构精度**
   - MPJPE < 5mm (果蝇体长约3mm)
   - 关键关节角度误差 < 10°

2. **时间预测**
   - 能够预测未来0.5-1秒的行为
   - 预测误差随时间增长缓慢

3. **闭环稳定性**
   - 闭环运行至少10秒不发散
   - 能够完成简单的行为序列

#### 定性目标
1. **行为自然性**
   - 生成的行为看起来自然
   - 符合物理规律
   - 与真实果蝇行为相似

2. **模型可解释性**
   - 能够分析哪些神经元对特定行为重要
   - 神经活动模式与已知生理学一致

3. **泛化能力**
   - 在未见过的场景中表现合理
   - 对不同的社交情境有适当反应

---

### 长期目标 (后续研究)

1. **完整的社交行为**
   - 求偶行为
   - 攻击行为
   - 跟随行为

2. **多模态整合**
   - 视觉 + 嗅觉
   - 视觉 + 触觉

3. **学习和适应**
   - 在线学习新行为
   - 适应新环境

4. **神经机制理解**
   - 发现控制行为的关键神经回路
   - 理解社交行为的神经基础

---

## 实验设计建议

### 实验1: Baseline对比
**目的**: 证明我们的方法优于现有方法

**对比模型**:
- RF Model (简单感受野模型)
- 纯数据驱动的深度学习模型（LSTM/Transformer）
- 不使用连接组的模型

**评估**:
- 在相同数据上训练
- 使用相同的评估指标
- 分析各自的优缺点

### 实验2: 消融实验
**目的**: 验证各个组件的重要性

**变体**:
1. 不使用连接组约束
2. 不使用神经递质信息
3. 简化的神经元模型
4. 不同的损失函数组合

### 实验3: 泛化测试
**目的**: 测试模型的泛化能力

**测试场景**:
1. 未见过的果蝇个体
2. 不同的社交情境
3. 不同的环境条件

---

## 技术挑战和解决方案

### 挑战1: 计算复杂度
**问题**: 139,255个神经元，计算量巨大

**解决方案**:
1. 只使用相关的神经元子集（视觉-运动通路）
2. 使用稀疏矩阵运算
3. GPU加速
4. 简化神经元模型

### 挑战2: 训练不稳定
**问题**: 梯度消失/爆炸，训练难以收敛

**解决方案**:
1. 梯度裁剪
2. 学习率调度
3. 批归一化
4. 残差连接
5. 分阶段训练

### 挑战3: 数据对齐
**问题**: 视觉输入和行为数据的时间对齐

**解决方案**:
1. 仔细的时间戳对齐
2. 考虑感觉-运动延迟
3. 使用滑动窗口

### 挑战4: 评估困难
**问题**: 如何评估生成行为的质量

**解决方案**:
1. 多个定量指标
2. 人工评估（视频）
3. 与生物学家合作验证
4. 与真实数据的统计对比

---

## 代码框架示例

```python
class FlyBrainModel(nn.Module):
    def __init__(self, connectome, config):
        super().__init__()
        
        # 1. 视觉编码器
        self.visual_encoder = VisualEncoder(
            n_visual_neurons=config.n_visual
        )
        
        # 2. DMN核心网络
        self.dmn = DMN(
            connectome=connectome,
            n_neurons=config.n_neurons,
            neuron_model=config.neuron_model
        )
        
        # 3. 运动解码器
        self.motor_decoder = MotorDecoder(
            n_motor_neurons=config.n_motor,
            n_muscles=config.n_muscles
        )
        
    def forward(self, visual_input, state):
        # 视觉处理
        visual_activity = self.visual_encoder(visual_input)
        
        # 神经网络动力学
        neural_activity = self.dmn(visual_activity, state)
        
        # 运动输出
        muscle_signals = self.motor_decoder(neural_activity)
        
        return muscle_signals, neural_activity

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate
        )
        
    def train_step(self, batch):
        # 生成视觉输入
        visual_input = self.generate_visual_input(
            batch['scene_data'],
            batch['focal_fly_id']
        )
        
        # 前向传播
        predicted_muscles, neural_activity = self.model(
            visual_input,
            batch['initial_state']
        )
        
        # 使用NeuroMechFly模拟
        predicted_behavior = self.neuromechfly.simulate(
            predicted_muscles,
            n_steps=batch['n_steps']
        )
        
        # 计算损失
        loss = self.compute_loss(
            predicted_behavior,
            batch['true_behavior'],
            neural_activity
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        return loss.item()
    
    def compute_loss(self, pred, true, neural_activity):
        # 行为损失
        L_keypoint = F.mse_loss(pred.keypoints, true.keypoints)
        L_velocity = F.mse_loss(pred.velocity, true.velocity)
        
        # 物理约束
        L_physics = self.compute_physics_loss(pred)
        
        # 神经约束
        L_neural = self.compute_neural_loss(neural_activity)
        
        # 总损失
        total_loss = (
            self.config.alpha_keypoint * L_keypoint +
            self.config.alpha_velocity * L_velocity +
            self.config.alpha_physics * L_physics +
            self.config.alpha_neural * L_neural
        )
        
        return total_loss
```

---

## 总结

### 核心思路
1. **输入**: 从social behavior场景生成视觉输入
2. **模型**: 使用基于连接组的DMN处理
3. **输出**: 预测行为，控制NeuroMechFly
4. **训练**: 最小化预测行为与真实行为的差异

### 关键创新点
1. 使用真实的果蝇连接组
2. 从social behavior生成视觉输入
3. 端到端的闭环训练
4. 结合神经科学和深度学习

### 成功的关键
1. 合理的模型简化（不能太复杂）
2. 充足的训练数据
3. 合适的损失函数设计
4. 稳定的训练策略
5. 与Jizheng密切合作

---

**文档创建**: 2026年3月16日  
**版本**: 1.0  
**作者**: AI Assistant  
**审阅**: 待Jizheng确认
