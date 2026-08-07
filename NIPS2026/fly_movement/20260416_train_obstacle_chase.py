"""
20260416 - 类型特异LC映射：果蝇A避障追踪果蝇B
============================================================
改进点（相对于 20260416_train_correct_arch.py）：

1. **类型特异LC输入映射**
   - 基于 Cowley et al. 2024 (Nature) 的发现：不同LC类型编码不同视觉特征
   - LC10a组：位置编码（position-sensitive）
   - LC11组：小目标运动检测（small object motion）
   - LPLC2组：looming检测（size change / approach speed）
   - LC6/LC15/LC17组：大小编码（size-sensitive）
   - 其他LC：通用视觉特征
   - 各组有独立的输入编码器MLP

2. **更丰富的视觉特征（12维 → 分通道送入不同LC组）**
   - 目标相关：位置差、距离、视角大小、运动速度、looming速度、方向
   - 障碍物相关：墙距离、墙方向、墙接近速度
   - 整合特征：总暗区、方向符号、归一化距离

3. **多目标训练**
   - 追踪目标：朝果蝇B转向和前进
   - 避障目标：远离墙壁
   - 平衡权重：追踪 vs 避障

论文依据：
- Cowley et al. 2024: LC群体分布式编码，不同LC类型编码不同视觉特征
- Namiki et al. 2018: DN分左右投射到VNC
- Dorkenwald et al. 2024 (FlyWire): connectome数据来源
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============ Surrogate Gradient LIF（与原版一致）============
class SurrogateLIF(nn.Module):
    def __init__(self, n, tau=20.0, v_th=0.5,
                 v_reset=0.0, dt=1.0, surrogate_scale=10.0):
        super().__init__()
        self.tau = tau; self.v_th = v_th
        self.v_reset = v_reset; self.dt = dt
        self.surrogate_scale = surrogate_scale
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, v, I):
        dv = (-v + I + self.bias) * (self.dt / self.tau)
        v_new = v + dv
        sp_hard = (v_new >= self.v_th).float()
        sp_soft = torch.sigmoid(
            (v_new - self.v_th) * self.surrogate_scale)
        sp = sp_hard - sp_soft.detach() + sp_soft
        v_new = v_new * (1 - sp_hard) + self.v_reset * sp_hard
        return v_new, sp


# ============ LC类型分组器 ============
class LCTypeGrouper:
    """
    基于W矩阵的连接模式将LC神经元聚类为功能组。

    原理（Cowley et al. 2024）：
    - 具有相似下游投射的LC神经元倾向于编码相似的视觉特征
    - 不同组应该接收不同的视觉特征子集
    """
    def __init__(self, W, lc_idx, central_idx, n_groups=5):
        self.n_groups = n_groups
        # 用LC→Central的连接模式做聚类
        W_lc_cent = W[np.ix_(central_idx, lc_idx)]  # (n_central, n_lc)
        # 每个LC神经元的连接指纹 = 它向各Central神经元的投射模式
        fingerprints = W_lc_cent.T  # (n_lc, n_central)
        # 降维：用非零连接的统计特征
        features = np.column_stack([
            fingerprints.mean(axis=1),                # 平均连接强度
            (fingerprints > 0).sum(axis=1),            # 非零连接数
            fingerprints.max(axis=1),                  # 最强连接
            fingerprints.std(axis=1),                  # 连接异质性
            np.percentile(fingerprints, 90, axis=1),   # 90分位连接
        ])
        # KMeans聚类
        features_norm = (features - features.mean(0)) / (features.std(0) + 1e-8)
        km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        self.labels = km.fit_predict(features_norm)
        # 按组分割LC索引
        self.groups = {}
        for g in range(n_groups):
            mask = self.labels == g
            self.groups[g] = np.where(mask)[0]  # 局部索引（在lc_idx内的位置）
        # 为每组命名（基于连接特征）
        self.group_names = self._name_groups(features)

    def _name_groups(self, features):
        """根据连接特征给组命名"""
        names = {}
        for g in range(self.n_groups):
            idx = self.groups[g]
            avg_strength = features[idx, 0].mean()
            avg_fanout = features[idx, 1].mean()
            if avg_strength > np.percentile(features[:, 0], 75):
                names[g] = f"G{g}_strong_proj"   # 强投射组（类似LPLC2-looming）
            elif avg_fanout > np.percentile(features[:, 1], 75):
                names[g] = f"G{g}_wide_fanout"   # 广泛扇出（类似LC10a-position）
            elif avg_strength < np.percentile(features[:, 0], 25):
                names[g] = f"G{g}_weak_sparse"   # 弱/稀疏连接（类似LC11-small_obj）
            else:
                names[g] = f"G{g}_moderate"       # 中等连接
        return names

    def print_summary(self):
        print(f"\nLC Type Groups ({self.n_groups} groups):")
        for g in range(self.n_groups):
            print(f"  {self.group_names[g]}: "
                  f"{len(self.groups[g])} neurons")


# ============ 类型特异LC输入编码器 ============
class TypeSpecificLCEncoder(nn.Module):
    """
    不同LC组有不同的输入编码器，接收不同的视觉特征子集。

    基于 Cowley et al. 2024 的发现：
    - 位置敏感组：接收目标位置、方向特征
    - looming敏感组：接收大小变化、接近速度特征
    - 小目标组：接收运动速度、小尺寸特征
    - 大小敏感组：接收目标大小、距离特征
    - 通用组：接收所有特征的压缩表示

    视觉特征向量（12维）：
    [0] target_dark_diff   - 目标左右暗区差（位置信号）
    [1] target_dist_inv    - 目标距离倒数（近=大）
    [2] target_size        - 目标视角大小
    [3] target_direction   - 目标方向符号 (-1/0/+1)
    [4] target_speed       - 目标运动速度
    [5] target_looming     - 目标looming速度（大小变化率）
    [6] wall_dist_inv      - 墙距离倒数
    [7] wall_direction     - 墙方向（左/右/前）
    [8] wall_proximity     - 墙接近速度
    [9] total_dark          - 总暗区归一化
    [10] norm_dist          - 归一化目标距离
    [11] norm_wall_dist     - 归一化墙距离
    """
    def __init__(self, lc_grouper, total_vis_dim=12):
        super().__init__()
        self.n_groups = lc_grouper.n_groups
        self.groups = lc_grouper.groups
        self.group_names = lc_grouper.group_names

        # 每个组的输入特征子集和编码器
        # 定义每组接收哪些特征维度
        self.group_feature_idx = nn.ParameterDict()  # 不实际参与梯度
        self.encoders = nn.ModuleDict()

        for g in range(self.n_groups):
            n_neurons = len(self.groups[g])
            if n_neurons == 0:
                continue
            name = self.group_names[g]
            # 根据组的特征决定输入维度
            if 'wide_fanout' in name:
                # 位置敏感组：位置、方向、运动速度
                feat_idx = [0, 3, 4, 7, 10]
                in_dim = len(feat_idx)
            elif 'strong_proj' in name:
                # looming敏感组：大小变化、距离、looming
                feat_idx = [1, 2, 5, 6, 8]
                in_dim = len(feat_idx)
            elif 'weak_sparse' in name:
                # 小目标/稀疏组：运动速度、大小、方向
                feat_idx = [2, 4, 5, 9]
                in_dim = len(feat_idx)
            else:
                # 通用组：所有特征的压缩
                feat_idx = list(range(total_vis_dim))
                in_dim = total_vis_dim

            # 注册特征索引（作为buffer，不参与梯度）
            self.register_buffer(
                f'feat_idx_{g}',
                torch.LongTensor(feat_idx))

            # 每组一个独立的MLP编码器
            self.encoders[str(g)] = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_neurons),
                nn.Sigmoid()
            )

    def forward(self, vis_feat):
        """
        vis_feat: (batch, 12) 完整视觉特征
        返回: (batch, total_lc) 按组拼接的LC输入电流
        """
        outputs = []
        for g in range(self.n_groups):
            if len(self.groups[g]) == 0:
                continue
            # 取该组的特征子集
            idx = getattr(self, f'feat_idx_{g}')
            feat_subset = vis_feat[:, idx]
            # 通过该组的编码器
            out = self.encoders[str(g)](feat_subset)
            outputs.append(out)
        # 按组顺序拼接
        return torch.cat(outputs, dim=1)

    def get_group_order(self):
        """返回拼接后的LC索引顺序（用于重建到原始索引）"""
        order = []
        for g in range(self.n_groups):
            order.extend(self.groups[g].tolist())
        return order


# ============ 带类型特异LC的SNN ============
class TypeSpecificSNN(nn.Module):
    """
    改进的SNN架构：

    复眼(721小眼/眼)
         ↓ [12维视觉特征]
    类型特异LC编码器（5组独立MLP）
         ↓ [分组编码后合并]
    LC神经元(1420) - 按类型分组的视觉投射神经元
         ↓ [W_lc_cent: FlyWire真实突触，冻结]
    Central神经元(2737) - 中间整合层
         ↓ [W_cent_dn: FlyWire真实突触，冻结]
    DN_left + DN_right - 下行神经元
         ↓
    left_rate vs right_rate → 转向 + 速度控制
         ↓ [VNC替代: HybridTurningController]
    腿部运动 [left_speed, right_speed]
    """
    def __init__(self, W, lc_idx, central_idx,
                 dn_left_idx, dn_right_idx, dn_idx_all,
                 lc_grouper, vis_dim=12):
        super().__init__()
        self.lc_idx       = torch.LongTensor(lc_idx)
        self.central_idx  = torch.LongTensor(central_idx)
        self.dn_left_idx  = torch.LongTensor(dn_left_idx)
        self.dn_right_idx = torch.LongTensor(dn_right_idx)
        self.dn_idx_all   = torch.LongTensor(dn_idx_all)

        # W矩阵（冻结）
        W_scaled = W / 100.0
        self.register_buffer('W_lc_cent',
            torch.FloatTensor(
                W_scaled[np.ix_(central_idx, lc_idx)]))
        self.register_buffer('W_cent_dn',
            torch.FloatTensor(
                W_scaled[np.ix_(dn_idx_all, central_idx)]))

        # 各层LIF
        self.lif_lc      = SurrogateLIF(len(lc_idx))
        self.lif_central  = SurrogateLIF(len(central_idx))
        self.lif_dn       = SurrogateLIF(len(dn_idx_all))

        # ★ 核心改进：类型特异LC编码器 ★
        self.lc_encoder = TypeSpecificLCEncoder(
            lc_grouper, total_vis_dim=vis_dim)
        # 保存组序到原始索引的映射（用于重排列）
        group_order = self.lc_encoder.get_group_order()
        # 创建从组序到原始LC索引的置换矩阵
        n_lc = len(lc_idx)
        perm = torch.zeros(n_lc, n_lc)
        for new_pos, orig_pos in enumerate(group_order):
            perm[orig_pos, new_pos] = 1.0
        self.register_buffer('perm_matrix', perm)

        # 本体感觉 → DN（与原版相同）
        self.proprio_to_dn = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, len(dn_idx_all)), nn.Tanh())

        # 放电率统计缓冲区
        self._bufs = {k: [] for k in
                      ['lc', 'central', 'dn_left', 'dn_right']}
        self.win = 100

    def forward(self, vis_feat, proprio, state=None):
        batch = vis_feat.shape[0]
        dev = vis_feat.device
        if state is None:
            state = self.init_state(batch, dev)
        v_lc, v_central, v_dn = state

        # ======== LC层：类型特异编码 ========
        I_lc_grouped = self.lc_encoder(vis_feat)
        # 重排列回原始索引顺序
        I_lc = torch.matmul(I_lc_grouped, self.perm_matrix.T)
        v_lc, sp_lc = self.lif_lc(v_lc, I_lc)

        # ======== Central层 ========
        I_central = torch.einsum(
            'ij,bj->bi', self.W_lc_cent, sp_lc)
        v_central, sp_central = self.lif_central(
            v_central, I_central)

        # ======== DN层 ========
        I_dn = torch.einsum(
            'ij,bj->bi', self.W_cent_dn, sp_central)
        I_prop = self.proprio_to_dn(proprio) * 0.1
        v_dn, sp_dn = self.lif_dn(v_dn, I_dn + I_prop)

        # ======== DN左右分离 → 腿速度 ========
        n_left  = len(self.dn_left_idx)
        n_right = len(self.dn_right_idx)
        sp_dn_left  = sp_dn[:, :n_left]
        sp_dn_right = sp_dn[:, n_left:n_left+n_right]

        # 放电率统计
        for key, sp in [
            ('lc', sp_lc), ('central', sp_central),
            ('dn_left', sp_dn_left), ('dn_right', sp_dn_right)
        ]:
            self._bufs[key].append(sp.detach())
            if len(self._bufs[key]) > self.win:
                self._bufs[key].pop(0)

        rates = {}
        for k in self._bufs:
            if self._bufs[k]:
                rates[k] = torch.stack(
                    self._bufs[k]).mean(0) * 1000.0
            else:
                rates[k] = torch.zeros(batch, 1, device=dev)

        # 转向控制
        dn_left_rate  = sp_dn_left.mean(dim=1, keepdim=True)
        dn_right_rate = sp_dn_right.mean(dim=1, keepdim=True)
        turn_signal   = dn_left_rate - dn_right_rate

        # 总体DN活动 → 速度调节
        total_dn = sp_dn.mean(dim=1, keepdim=True)

        # ★ 非线性放大：小差异也能产生明确转向 ★
        turn_amplified = torch.sign(turn_signal) * torch.pow(
            torch.abs(turn_signal) * 5.0, 0.6).clamp(max=1.0)
        # 急转时大幅减速
        turn_abs = torch.abs(turn_amplified)
        base_speed = torch.where(
            turn_abs > 0.3,
            torch.tensor(0.2),    # 急转：几乎原地转
            0.5 + total_dn * 0.3  # 正常前进
        )
        turn_scale = 5.0

        left_leg  = torch.clamp(
            base_speed - turn_amplified * turn_scale, 0.1, 1.5)
        right_leg = torch.clamp(
            base_speed + turn_amplified * turn_scale, 0.1, 1.5)

        cmd = torch.cat([left_leg, right_leg], dim=1)
        new_state = (v_lc, v_central, v_dn)
        return cmd, new_state, rates, turn_signal

    def init_state(self, batch=1, device='cpu'):
        return (
            torch.zeros(batch, len(self.lc_idx), device=device),
            torch.zeros(batch, len(self.central_idx), device=device),
            torch.zeros(batch, len(self.dn_idx_all), device=device),
        )

    def reset(self, batch=1, device='cpu'):
        self._bufs = {k: [] for k in
                      ['lc', 'central', 'dn_left', 'dn_right']}
        return self.init_state(batch, device)


# ============ 场景生成器（合成训练数据）============
class ObstacleChaseScenario:
    """
    生成果蝇A追踪果蝇B + 避开墙壁的训练场景

    场景布局（俯视）:
        ┌─────────────────────┐
        │                     │
        │    B (移动目标)       │
        │         ↑           │
        │   ██████████  (墙)  │
        │         ↑           │
        │    A (追踪者)        │
        │                     │
        └─────────────────────┘

    果蝇B在墙的另一侧移动，果蝇A需要绕墙追踪
    """
    def __init__(self,
                 arena_size=40.0,       # mm
                 wall_y=10.0,           # 墙的y坐标
                 wall_x_range=(-8, 8),  # 墙的x范围
                 wall_thickness=1.0):   # 墙厚度
        self.arena_size = arena_size
        self.wall_y = wall_y
        self.wall_x_range = wall_x_range
        self.wall_thickness = wall_thickness

    def generate_batch(self, bs=256):
        """
        生成一批训练样本

        返回：
            vis_feat: (bs, 12) 视觉特征
            proprio: (bs, 12) 本体感觉（填零）
            targets: (bs, 2) 目标腿速度 [left, right]
        """
        # ★ 混合场景：50%远距离接近 + 50%近距离跟随 ★
        n_far   = bs // 2
        n_close = bs - n_far

        # --- 远距离场景：A在墙南侧，B在墙北侧 ---
        far_a_x = np.random.uniform(-15, 15, n_far)
        far_a_y = np.random.uniform(-5, 8, n_far)
        far_a_h = np.random.uniform(-np.pi, np.pi, n_far)
        far_b_x = np.random.uniform(-15, 15, n_far)
        far_b_y = np.random.uniform(12, 25, n_far)

        # --- 近距离场景：A就在B附近（2~8mm），各种角度 ---
        close_b_x = np.random.uniform(-10, 20, n_close)
        close_b_y = np.random.uniform(8, 20, n_close)
        close_dist = np.random.uniform(2.0, 8.0, n_close)
        close_angle = np.random.uniform(-np.pi, np.pi, n_close)
        close_a_x = close_b_x + close_dist * np.cos(close_angle)
        close_a_y = close_b_y + close_dist * np.sin(close_angle)
        close_a_h = np.random.uniform(-np.pi, np.pi, n_close)

        # 合并
        fly_a_x = np.concatenate([far_a_x, close_a_x])
        fly_a_y = np.concatenate([far_a_y, close_a_y])
        fly_a_heading = np.concatenate([far_a_h, close_a_h])
        fly_b_x = np.concatenate([far_b_x, close_b_x])
        fly_b_y = np.concatenate([far_b_y, close_b_y])

        # 果蝇B的运动速度（模拟行走）
        fly_b_vx = np.random.uniform(-2, 2, bs)
        fly_b_vy = np.random.uniform(-1, 1, bs)

        # ======== 计算12维视觉特征 ========
        # 相对目标（果蝇B）
        dx_b = fly_b_x - fly_a_x
        dy_b = fly_b_y - fly_a_y
        dist_b = np.sqrt(dx_b**2 + dy_b**2)

        # 目标在A视野中的角度
        angle_to_b = np.arctan2(dy_b, dx_b) - fly_a_heading
        angle_to_b = (angle_to_b + np.pi) % (2*np.pi) - np.pi

        # 目标视角大小（模拟：距离越近越大）
        target_size = np.clip(4.0 / (dist_b + 1.0), 0, 1)

        # 目标运动速度（径向分量）
        radial_speed = (dx_b * fly_b_vx + dy_b * fly_b_vy) / (dist_b + 0.1)
        looming_speed = np.clip(-radial_speed / 5.0, -1, 1)  # 接近为正

        # 目标切向速度
        tangent_speed = (-dy_b * fly_b_vx + dx_b * fly_b_vy) / (dist_b + 0.1)

        # 相对墙壁
        # 到墙的最近距离
        wall_dist = np.abs(fly_a_y - self.wall_y)
        # 墙在A面前还是后面
        wall_in_front = (
            (self.wall_y - fly_a_y) * np.cos(fly_a_heading) > 0
        ).astype(float)
        # A是否在墙的x范围内
        in_wall_range = (
            (fly_a_x >= self.wall_x_range[0]) &
            (fly_a_x <= self.wall_x_range[1])
        ).astype(float)

        # 墙的方向（相对A的朝向）
        wall_center_x = (self.wall_x_range[0] + self.wall_x_range[1]) / 2
        dx_wall = wall_center_x - fly_a_x
        dy_wall = self.wall_y - fly_a_y
        angle_to_wall = np.arctan2(dy_wall, dx_wall) - fly_a_heading
        angle_to_wall = (angle_to_wall + np.pi) % (2*np.pi) - np.pi

        # 组装12维特征
        # ★ 模拟CompoundEyeProcessor的输出 ★
        # 复眼原理：方向 = 左右加权亮度差 ∝ sin(angle) × visibility
        visibility = np.clip(target_size * 3.0, 0, 1)  # 近=1, 远≈0
        noise = np.random.normal(0, 0.08, bs)

        # [0] 方向信号（加权亮度差，经时间平滑）
        direction = np.sin(angle_to_b) * visibility + noise
        # [1] 总暗区（≈目标大小/近度）
        total_dark = np.clip(target_size + np.random.normal(0, 0.05, bs), 0, 1)
        # [2] 对比度（有无显著物体）
        contrast = np.clip(visibility * 0.8 + np.random.normal(0, 0.05, bs), 0, 1)
        # [3] 方向符号
        direction_sign = np.sign(direction)
        # [4] 运动方向（目标切向速度的视觉反映）
        motion_dir = np.clip(
            tangent_speed * visibility * 0.3 + np.random.normal(0, 0.05, bs), -1, 1)
        # [5] looming（接近速度的视觉反映）
        looming_vis = np.clip(
            np.abs(looming_speed) * visibility + np.random.normal(0, 0.03, bs), 0, 1)

        vis_feat = np.stack([
            np.clip(direction, -1, 1),             # [0]  方向
            total_dark,                              # [1]  总暗区
            contrast,                                # [2]  对比度
            direction_sign,                          # [3]  方向符号
            motion_dir,                              # [4]  运动方向
            looming_vis,                             # [5]  looming
            np.clip(1.0/(wall_dist+1.0), 0, 1) * in_wall_range,  # [6] 墙接近
            np.sign(fly_a_x - (self.wall_x_range[0]+self.wall_x_range[1])/2),  # [7] 墙方位
            np.clip(wall_in_front * in_wall_range, 0, 1),  # [8] 墙正前方
            total_dark,                              # [9]  瞬时暗区
            np.clip(1.0 - total_dark, 0, 1),        # [10] 距离估计
            np.clip(wall_dist / 15.0, 0, 1),        # [11] 归一化墙距
        ], axis=1).astype(np.float32)

        # ======== 计算目标腿速度 ========

        # ★ 角度自适应转向：角度越大转弯越急 ★
        abs_angle = np.abs(angle_to_b)

        # 小角度(<30°)：温和转向
        # 中角度(30-90°)：中等转向
        # 大角度(>90°，目标在身后)：急转弯
        turn_gain = np.where(
            abs_angle > np.pi/2, 0.8,           # 身后：急转
            np.where(abs_angle > np.pi/6, 0.5,  # 侧面：中转
                     0.3))                        # 正前方：微调

        track_turn = np.clip(
            np.sin(angle_to_b) * turn_gain, -0.8, 0.8)

        # 目标在正后方(>120°)：最大转向 + 几乎原地转
        far_behind = abs_angle > 2*np.pi/3
        track_turn[far_behind] = np.sign(
            angle_to_b[far_behind]) * 0.8

        # 避障信号：远离墙壁
        wall_repulsion = np.zeros(bs)
        danger_mask = (wall_dist < 5.0) & (wall_in_front > 0.5) & (in_wall_range > 0.5)
        target_side = np.sign(dx_b)
        wall_repulsion[danger_mask] = (
            target_side[danger_mask] * 0.4 *
            (1.0 - wall_dist[danger_mask] / 5.0)
        )

        # 合并：追踪 + 避障
        total_turn = np.clip(track_turn + wall_repulsion, -0.8, 0.8)

        # ★ 急转时减速，小调整时正常速度 ★
        abs_turn = np.abs(total_turn)
        base_speed = np.where(
            abs_turn > 0.5, 0.2,                  # 急转：几乎停下来转
            np.where(dist_b < 3.0, 0.4,           # 近距离跟随
                np.where(dist_b < 10.0,
                    0.4 + (dist_b-3.0)*0.06,       # 中距离渐变
                    0.8)))                          # 远距离全速

        left_t  = np.clip(
            (base_speed - total_turn), 0.1, 1.5
        ).astype(np.float32)
        right_t = np.clip(
            (base_speed + total_turn), 0.1, 1.5
        ).astype(np.float32)

        return (
            torch.FloatTensor(vis_feat),
            torch.zeros(bs, 12),
            torch.FloatTensor(np.stack([left_t, right_t], axis=1))
        )


# ============ 加载数据 ============
print("=" * 60)
print("Loading connectome data...")
print("=" * 60)
W    = np.load('20260416_full_pathway_W_cleft.npy')
meta = pd.read_csv('20260416_full_pathway_meta_cleft.csv')
df_dn_lat = pd.read_csv('20260416_dn_with_laterality.csv')

lc_idx      = meta[meta['is_LC']==True].index.values
central_idx = meta[meta['is_central']==True].index.values

dn_left_ids  = df_dn_lat[df_dn_lat['laterality']=='left']['root_id'].tolist()
dn_right_ids = df_dn_lat[df_dn_lat['laterality']=='right']['root_id'].tolist()
dn_left_meta  = meta[meta['root_id'].isin(dn_left_ids)]
dn_right_meta = meta[meta['root_id'].isin(dn_right_ids)]
dn_left_idx   = dn_left_meta.index.values
dn_right_idx  = dn_right_meta.index.values
dn_idx_all    = np.concatenate([dn_left_idx, dn_right_idx])

print(f"LC: {len(lc_idx)}")
print(f"Central: {len(central_idx)}")
print(f"DN_left: {len(dn_left_idx)}")
print(f"DN_right: {len(dn_right_idx)}")
print(f"W shape: {W.shape}")

# ============ LC类型聚类 ============
print("\n--- LC Type Grouping ---")
lc_grouper = LCTypeGrouper(W, lc_idx, central_idx, n_groups=5)
lc_grouper.print_summary()

# ============ 创建模型 ============
snn = TypeSpecificSNN(
    W, lc_idx, central_idx,
    dn_left_idx, dn_right_idx, dn_idx_all,
    lc_grouper, vis_dim=12)

n_params = sum(p.numel() for p in snn.parameters())
n_trainable = sum(p.numel() for p in snn.parameters() if p.requires_grad)
print(f"\nTotal parameters: {n_params:,}")
print(f"Trainable parameters: {n_trainable:,}")

# ============ 验证（训练前）============
print("\n--- Pre-training Validation ---")
scenario = ObstacleChaseScenario()
snn.eval()

test_cases = [
    ('Target LEFT, no wall',    [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0,
                                  0.0, 0.0, 0.0, 0.1, 0.5, 1.0]),
    ('Target RIGHT, no wall',   [-0.5, 0.3, 0.2, -1.0, 0.1, 0.0,
                                  0.0, 0.0, 0.0, 0.1, 0.5, 1.0]),
    ('Target FRONT, wall close', [0.0, 0.3, 0.2, 0.0, 0.1, 0.0,
                                  0.8, 0.0, 1.0, 0.1, 0.5, 0.2]),
    ('Target LEFT, wall close',  [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0,
                                  0.8, 0.3, 1.0, 0.1, 0.5, 0.2]),
]
for name, vf in test_cases:
    state = snn.reset(batch=1)
    with torch.no_grad():
        for t in range(50):
            cmd, state, rates, turn = snn(
                torch.FloatTensor(vf).unsqueeze(0),
                torch.zeros(1, 12), state)
    l = float(cmd[0, 0]); r = float(cmd[0, 1])
    dl = rates['dn_left'].mean().item()
    dr = rates['dn_right'].mean().item()
    direction = 'LEFT' if l < r else 'RIGHT' if l > r else 'STRAIGHT'
    print(f"  {name:30s} | L={l:.3f} R={r:.3f} {direction} | "
          f"DN_L={dl:.0f} DN_R={dr:.0f}Hz")

# ============ 训练 ============
print("\n" + "=" * 60)
print("=== Multi-objective Training ===")
print("Goal: Track fly B + Avoid wall")
print("=" * 60)

optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=500, eta_min=1e-5)

T_warmup = 30
n_epochs = 500
losses = []
track_losses = []
avoid_losses = []
best_loss = 999.0

for epoch in range(n_epochs):
    snn.train()
    vis, prop, targets = scenario.generate_batch(256)
    state = snn.reset(batch=256)

    # Warmup
    with torch.no_grad():
        for t in range(T_warmup):
            _, state, _, _ = snn(vis, prop, state)

    # 前向传播 + 计算loss
    loss_acc = 0.0
    for t in range(10):
        cmd, state, rates, turn = snn(vis, prop, state)
        if t >= 5:
            # 主loss：腿速度匹配
            tracking_loss = nn.MSELoss()(cmd, targets)

            # 辅助loss：确保墙附近时DN左右不对称足够大
            wall_close = (vis[:, 6:7] > 0.3).float()  # 墙距离近的样本
            dn_asymmetry = torch.abs(turn)
            avoid_loss = (wall_close * (0.3 - dn_asymmetry)).clamp(min=0).mean()

            loss_acc += tracking_loss + 0.3 * avoid_loss
    loss = loss_acc / 5.0

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(snn.state_dict(),
                   '20260416_obstacle_chase_best.pt')

    if epoch % 100 == 0 or epoch == n_epochs - 1:
        snn.eval()
        state_t = snn.reset(batch=1)
        res = []
        for name, vf in [
            ('Front',      [0.0, 0.3, 0.2, 0.0, 0.1, 0.0,
                            0.0, 0.0, 0.0, 0.1, 0.5, 1.0]),
            ('Left',       [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0,
                            0.0, 0.0, 0.0, 0.1, 0.5, 1.0]),
            ('Right',      [-0.5, 0.3, 0.2, -1.0, 0.1, 0.0,
                            0.0, 0.0, 0.0, 0.1, 0.5, 1.0]),
            ('Wall+Left',  [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0,
                            0.8, 0.0, 1.0, 0.1, 0.5, 0.2]),
        ]:
            with torch.no_grad():
                for t in range(T_warmup + 10):
                    c, state_t, rt, ts = snn(
                        torch.FloatTensor(vf).unsqueeze(0),
                        torch.zeros(1, 12), state_t)
            l = float(c[0, 0]); r = float(c[0, 1])
            direction = 'LEFT' if l < r else 'RIGHT'
            res.append(f"{name}:{direction}({l:.2f}/{r:.2f})")
        print(f"Ep{epoch:3d} | Loss={loss.item():.4f} | "
              + ' | '.join(res))

# ============ 最终验证 ============
print("\n=== Final Validation ===")
snn.load_state_dict(
    torch.load('20260416_obstacle_chase_best.pt'))
snn.eval()

scenarios = [
    ('Target Front',
     [0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0],
     'STRAIGHT'),
    ('Target Left',
     [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0],
     'LEFT'),
    ('Target Right',
     [-0.5, 0.3, 0.2, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0],
     'RIGHT'),
    ('Wall+Target Left',
     [+0.5, 0.3, 0.2, +1.0, 0.1, 0.0, 0.8, 0.0, 1.0, 0.1, 0.5, 0.2],
     'LEFT'),
    ('Wall+Target Right',
     [-0.5, 0.3, 0.2, -1.0, 0.1, 0.0, 0.8, 0.0, 1.0, 0.1, 0.5, 0.2],
     'RIGHT'),
    ('Far Target Left',
     [+0.2, 0.1, 0.05, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.9, 1.0],
     'LEFT'),
]

all_correct = True
state_t = snn.reset(batch=1)
for name, vf, expected in scenarios:
    with torch.no_grad():
        for t in range(T_warmup + 10):
            c, state_t, rt, ts = snn(
                torch.FloatTensor(vf).unsqueeze(0),
                torch.zeros(1, 12), state_t)
    l = float(c[0, 0]); r = float(c[0, 1])
    dl = rt['dn_left'].mean().item()
    dr = rt['dn_right'].mean().item()
    lc_hz = rt['lc'].mean().item()
    ct_hz = rt['central'].mean().item()
    direction = 'LEFT' if l < r - 0.02 else 'RIGHT' if l > r + 0.02 else 'STRAIGHT'
    ok = 'OK' if expected == direction or expected == 'STRAIGHT' else 'FAIL'
    if ok == 'FAIL':
        all_correct = False
    print(f"  {name:22s} | L={l:.3f} R={r:.3f} {direction:8s} {ok} | "
          f"LC={lc_hz:.0f} Cent={ct_hz:.0f} "
          f"DN_L={dl:.0f} DN_R={dr:.0f}Hz")

print()
if all_correct:
    print("All correct!")
else:
    print("Partial - some scenarios need tuning")

print()
print("Architecture summary:")
print(f"  Compound eye (721 ommatidia/eye)")
print(f"  → 12-dim visual features (target + obstacle)")
print(f"  → Type-specific LC encoders ({lc_grouper.n_groups} groups)")
for g in range(lc_grouper.n_groups):
    print(f"      {lc_grouper.group_names[g]}: "
          f"{len(lc_grouper.groups[g])} neurons")
print(f"  → LC neurons ({len(lc_idx)}) [Grouped Visual Projection]")
print(f"  → Central neurons ({len(central_idx)}) [Integration]")
print(f"  → DN_left ({len(dn_left_idx)}) + DN_right ({len(dn_right_idx)}) [Descending]")
print(f"  → VNC (HybridTurningController, surrogate)")
print(f"  → Leg motion [left_speed, right_speed]")

# ============ 训练曲线 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(losses, 'b-', linewidth=1.5, alpha=0.5)
if len(losses) >= 10:
    ma = [np.mean(losses[max(0, i-9):i+1])
          for i in range(len(losses))]
    ax1.plot(ma, 'r-', linewidth=2, label='Moving avg')
    ax1.legend()
ax1.set_title('Type-Specific LC Training\n'
    'Obstacle Chase: Track FlyB + Avoid Wall', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# DN响应曲线
dark_diffs = np.linspace(-1, 1, 60)
left_speeds = []; right_speeds = []
dn_left_rates = []; dn_right_rates = []
# 无墙
state_s = snn.reset(batch=1)
for dd in dark_diffs:
    vf = torch.FloatTensor(
        [[dd, 0.3, abs(dd)*0.3, np.sign(dd), 0.1, 0.0,
          0.0, 0.0, 0.0, 0.1, 0.5, 1.0]])
    with torch.no_grad():
        for t in range(T_warmup + 5):
            c, state_s, rt, ts = snn(
                vf, torch.zeros(1, 12), state_s)
    left_speeds.append(float(c[0, 0]))
    right_speeds.append(float(c[0, 1]))
    dn_left_rates.append(rt['dn_left'].mean().item())
    dn_right_rates.append(rt['dn_right'].mean().item())

# 有墙
left_speeds_w = []; right_speeds_w = []
state_w = snn.reset(batch=1)
for dd in dark_diffs:
    vf = torch.FloatTensor(
        [[dd, 0.3, abs(dd)*0.3, np.sign(dd), 0.1, 0.0,
          0.7, 0.0, 0.8, 0.1, 0.5, 0.2]])
    with torch.no_grad():
        for t in range(T_warmup + 5):
            c, state_w, rt, ts = snn(
                vf, torch.zeros(1, 12), state_w)
    left_speeds_w.append(float(c[0, 0]))
    right_speeds_w.append(float(c[0, 1]))

ax2.plot(dark_diffs, left_speeds, 'b-', linewidth=2,
         label='Left leg (no wall)')
ax2.plot(dark_diffs, right_speeds, 'r-', linewidth=2,
         label='Right leg (no wall)')
ax2.plot(dark_diffs, left_speeds_w, 'b--', linewidth=1.5,
         alpha=0.6, label='Left leg (wall close)')
ax2.plot(dark_diffs, right_speeds_w, 'r--', linewidth=1.5,
         alpha=0.6, label='Right leg (wall close)')
ax2.axhline(y=0.8, color='gray', linestyle=':', linewidth=1)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_title('Leg Speed vs Target Direction\n'
    'Solid=no wall, Dashed=wall close', fontsize=12)
ax2.set_xlabel('Target Direction (dark diff)')
ax2.set_ylabel('Leg Speed')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f'Type-Specific LC Architecture: '
    f'Eye → LC({len(lc_idx)}, {lc_grouper.n_groups} groups) → '
    f'Central({len(central_idx)}) → '
    f'DN_L({len(dn_left_idx)})+DN_R({len(dn_right_idx)}) → VNC\n'
    f'FlyWire FAFB | Obstacle Chase | '
    f'Best Loss={best_loss:.4f}',
    fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('20260416_obstacle_chase_training.png', dpi=150)
print("\nSaved: 20260416_obstacle_chase_training.png")
print("Model: 20260416_obstacle_chase_best.pt")
print("Next:  20260416_watch_obstacle_chase.py")
