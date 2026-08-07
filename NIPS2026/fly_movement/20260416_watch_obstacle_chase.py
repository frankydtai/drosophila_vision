"""
20260416 - 观看脚本：果蝇A避障追踪果蝇B
============================================================
使用类型特异LC映射训练的模型，在NeuroMechFly物理仿真中运行：
- 果蝇A（NeuroMechFly）：由SNN控制，追踪果蝇B并避开墙壁
- 果蝇B（MuJoCo body）：沿预设路径行走的目标
- 墙壁：场景中的物理障碍物

信号通路：
  复眼(721) → 12维特征 → 类型特异LC编码器(5组)
  → LC(1420) → Central(2737) → DN_L(632)+DN_R(621)
  → VNC → 腿部运动
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import flygym
import flygym.arena
from flygym.examples.locomotion import HybridTurningController
import mujoco
import mujoco.viewer
import imageio

# ============ 导入模型定义（与训练脚本一致）============
# 这里直接内联定义，确保与训练完全一致
from sklearn.cluster import KMeans

class SurrogateLIF(nn.Module):
    def __init__(self, n, tau=20.0, v_th=0.5,
                 v_reset=0.0, dt=1.0, surrogate_scale=10.0):
        super().__init__()
        self.tau=tau; self.v_th=v_th
        self.v_reset=v_reset; self.dt=dt
        self.surrogate_scale=surrogate_scale
        self.bias=nn.Parameter(torch.zeros(n))
    def forward(self, v, I):
        dv=(-v+I+self.bias)*(self.dt/self.tau)
        v_new=v+dv
        sp_hard=(v_new>=self.v_th).float()
        sp_soft=torch.sigmoid(
            (v_new-self.v_th)*self.surrogate_scale)
        sp=sp_hard-sp_soft.detach()+sp_soft
        v_new=v_new*(1-sp_hard)+self.v_reset*sp_hard
        return v_new, sp

class LCTypeGrouper:
    def __init__(self, W, lc_idx, central_idx, n_groups=5):
        self.n_groups = n_groups
        W_lc_cent = W[np.ix_(central_idx, lc_idx)]
        fingerprints = W_lc_cent.T
        features = np.column_stack([
            fingerprints.mean(axis=1),
            (fingerprints > 0).sum(axis=1),
            fingerprints.max(axis=1),
            fingerprints.std(axis=1),
            np.percentile(fingerprints, 90, axis=1),
        ])
        features_norm = (features - features.mean(0)) / (features.std(0) + 1e-8)
        km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        self.labels = km.fit_predict(features_norm)
        self.groups = {}
        for g in range(n_groups):
            mask = self.labels == g
            self.groups[g] = np.where(mask)[0]
        self.group_names = self._name_groups(features)
    def _name_groups(self, features):
        names = {}
        for g in range(self.n_groups):
            idx = self.groups[g]
            avg_strength = features[idx, 0].mean()
            avg_fanout = features[idx, 1].mean()
            if avg_strength > np.percentile(features[:, 0], 75):
                names[g] = f"G{g}_strong_proj"
            elif avg_fanout > np.percentile(features[:, 1], 75):
                names[g] = f"G{g}_wide_fanout"
            elif avg_strength < np.percentile(features[:, 0], 25):
                names[g] = f"G{g}_weak_sparse"
            else:
                names[g] = f"G{g}_moderate"
        return names
    def print_summary(self):
        print(f"\nLC Type Groups ({self.n_groups} groups):")
        for g in range(self.n_groups):
            print(f"  {self.group_names[g]}: "
                  f"{len(self.groups[g])} neurons")

class TypeSpecificLCEncoder(nn.Module):
    def __init__(self, lc_grouper, total_vis_dim=12):
        super().__init__()
        self.n_groups = lc_grouper.n_groups
        self.groups = lc_grouper.groups
        self.group_names = lc_grouper.group_names
        self.encoders = nn.ModuleDict()
        for g in range(self.n_groups):
            n_neurons = len(self.groups[g])
            if n_neurons == 0: continue
            name = self.group_names[g]
            if 'wide_fanout' in name:
                feat_idx = [0, 3, 4, 7, 10]
            elif 'strong_proj' in name:
                feat_idx = [1, 2, 5, 6, 8]
            elif 'weak_sparse' in name:
                feat_idx = [2, 4, 5, 9]
            else:
                feat_idx = list(range(total_vis_dim))
            self.register_buffer(f'feat_idx_{g}',
                                 torch.LongTensor(feat_idx))
            in_dim = len(feat_idx)
            self.encoders[str(g)] = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, n_neurons), nn.Sigmoid())
    def forward(self, vis_feat):
        outputs = []
        for g in range(self.n_groups):
            if len(self.groups[g]) == 0: continue
            idx = getattr(self, f'feat_idx_{g}')
            feat_subset = vis_feat[:, idx]
            out = self.encoders[str(g)](feat_subset)
            outputs.append(out)
        return torch.cat(outputs, dim=1)
    def get_group_order(self):
        order = []
        for g in range(self.n_groups):
            order.extend(self.groups[g].tolist())
        return order

class TypeSpecificSNN(nn.Module):
    def __init__(self, W, lc_idx, central_idx,
                 dn_left_idx, dn_right_idx, dn_idx_all,
                 lc_grouper, vis_dim=12):
        super().__init__()
        self.lc_idx       = torch.LongTensor(lc_idx)
        self.central_idx  = torch.LongTensor(central_idx)
        self.dn_left_idx  = torch.LongTensor(dn_left_idx)
        self.dn_right_idx = torch.LongTensor(dn_right_idx)
        self.dn_idx_all   = torch.LongTensor(dn_idx_all)
        W_scaled = W / 100.0
        self.register_buffer('W_lc_cent',
            torch.FloatTensor(W_scaled[np.ix_(central_idx, lc_idx)]))
        self.register_buffer('W_cent_dn',
            torch.FloatTensor(W_scaled[np.ix_(dn_idx_all, central_idx)]))
        self.lif_lc      = SurrogateLIF(len(lc_idx))
        self.lif_central = SurrogateLIF(len(central_idx))
        self.lif_dn      = SurrogateLIF(len(dn_idx_all))
        self.lc_encoder  = TypeSpecificLCEncoder(lc_grouper, total_vis_dim=vis_dim)
        group_order = self.lc_encoder.get_group_order()
        n_lc = len(lc_idx)
        perm = torch.zeros(n_lc, n_lc)
        for new_pos, orig_pos in enumerate(group_order):
            perm[orig_pos, new_pos] = 1.0
        self.register_buffer('perm_matrix', perm)
        self.proprio_to_dn = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, len(dn_idx_all)), nn.Tanh())
        self._bufs = {k: [] for k in ['lc','central','dn_left','dn_right']}
        self.win = 100

    def forward(self, vis_feat, proprio, state=None):
        batch = vis_feat.shape[0]; dev = vis_feat.device
        if state is None: state = self.init_state(batch, dev)
        v_lc, v_central, v_dn = state
        I_lc_grouped = self.lc_encoder(vis_feat)
        I_lc = torch.matmul(I_lc_grouped, self.perm_matrix.T)
        v_lc, sp_lc = self.lif_lc(v_lc, I_lc)
        I_central = torch.einsum('ij,bj->bi', self.W_lc_cent, sp_lc)
        v_central, sp_central = self.lif_central(v_central, I_central)
        I_dn = torch.einsum('ij,bj->bi', self.W_cent_dn, sp_central)
        I_prop = self.proprio_to_dn(proprio) * 0.1
        v_dn, sp_dn = self.lif_dn(v_dn, I_dn + I_prop)
        n_left = len(self.dn_left_idx); n_right = len(self.dn_right_idx)
        sp_dn_left = sp_dn[:,:n_left]; sp_dn_right = sp_dn[:,n_left:n_left+n_right]
        for key,sp in [('lc',sp_lc),('central',sp_central),
                        ('dn_left',sp_dn_left),('dn_right',sp_dn_right)]:
            self._bufs[key].append(sp.detach())
            if len(self._bufs[key])>self.win: self._bufs[key].pop(0)
        rates={}
        for k in self._bufs:
            if self._bufs[k]:
                rates[k]=torch.stack(self._bufs[k]).mean(0)*1000.0
            else:
                rates[k]=torch.zeros(batch,1,device=dev)
        dn_left_rate  = sp_dn_left.mean(dim=1, keepdim=True)
        dn_right_rate = sp_dn_right.mean(dim=1, keepdim=True)
        turn_signal   = dn_left_rate - dn_right_rate
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
        left_leg  = torch.clamp(base_speed - turn_amplified*turn_scale, 0.1, 1.5)
        right_leg = torch.clamp(base_speed + turn_amplified*turn_scale, 0.1, 1.5)
        cmd = torch.cat([left_leg, right_leg], dim=1)
        return cmd, (v_lc, v_central, v_dn), rates, turn_signal

    def init_state(self, batch=1, device='cpu'):
        return (torch.zeros(batch,len(self.lc_idx),device=device),
                torch.zeros(batch,len(self.central_idx),device=device),
                torch.zeros(batch,len(self.dn_idx_all),device=device))
    def reset(self, batch=1, device='cpu'):
        self._bufs={k:[] for k in ['lc','central','dn_left','dn_right']}
        return self.init_state(batch, device)


# ============ 参数 ============
# 果蝇B的运动路径（椭圆轨迹）
FLY_B_CENTER = np.array([15.0, 10.0])  # 中心位置
FLY_B_RADIUS_X = 10.0
FLY_B_RADIUS_Y = 6.0
FLY_B_SPEED = 0.00008  # 角速度（弧度/步）——比果蝇A慢

# 墙壁位置
WALL_POS_Y = 5.0       # 墙的y坐标
WALL_X_MIN = -5.0
WALL_X_MAX = 10.0
WALL_HEIGHT = 3.0

N_STEPS        = 40000
RENDER_EVERY   = 50
EYE_EVERY      = 100


# ============ 纯复眼视觉处理器 ============
class CompoundEyeProcessor:
    """
    纯复眼视觉特征提取器 —— 不使用任何几何先验

    所有关于果蝇B的信息完全从复眼的 721×2 像素值推导：
    - 方向：左右眼加权亮度差
    - 距离/大小：暗区像素总面积
    - 运动：帧间亮度变化的左右差异
    - looming：帧间暗区总量变化

    设计原理（对应真实果蝇视觉通路）：
    - 视网膜 → R1-R6光感受器 → 亮度信号
    - Lamina → 时间高通滤波（运动检测）
    - Medulla → 方向选择性（Hassenstein-Reichardt检测器）
    - Lobula/LP → LC神经元（特征检测）
    """
    def __init__(self, ema_alpha=0.15):
        """
        ema_alpha: 指数移动平均的平滑系数（越小越平滑）
        """
        self.prev_left = None
        self.prev_right = None
        # 平滑后的特征（模拟神经回路的时间常数）
        self.smooth_direction = 0.0
        self.smooth_direction_sign = 0.0
        self.smooth_total_dark = 0.0
        self.smooth_motion_dir = 0.0
        self.smooth_looming = 0.0
        self.smooth_contrast = 0.0
        self.alpha = ema_alpha

    def _ema(self, old, new):
        """指数移动平均"""
        return old * (1 - self.alpha) + new * self.alpha

    def process(self, vision, fly_pos, wall_y, wall_x_range):
        """
        从复眼原始数据提取12维特征

        参数：
            vision: (2, 721, 2) 复眼数据
            fly_pos: 果蝇A自身位置（用于墙壁感知，墙是已知环境）
            wall_y, wall_x_range: 墙的位置（固定环境信息）

        返回：
            features: (12,) 特征向量
            dark_diff: 原始暗区差（用于显示）
        """
        left = vision[0, :, 0].copy()    # 721个小眼，左眼
        right = vision[1, :, 0].copy()   # 721个小眼，右眼

        # ======== 1. 方向检测（左右亮度不对称）========
        # 加权亮度差：不是简单计数暗像素，而是累加"暗的程度"
        # 这样近处大目标（很暗、占很多小眼）信号强
        # 远处小目标（稍暗、占少数小眼）信号弱但非零
        ref = 0.5  # 参考亮度（中灰）
        left_darkness  = np.sum(np.clip(ref - left, 0, None))
        right_darkness = np.sum(np.clip(ref - right, 0, None))
        raw_direction = (left_darkness - right_darkness)
        # 归一化到 ±1（经验值：最大差约50）
        norm_direction = np.clip(raw_direction / 30.0, -1, 1)

        # 暗区计数（用于显示和符号判断）
        left_dark_count  = (left < 0.3).sum()
        right_dark_count = (right < 0.3).sum()
        dark_diff = float(left_dark_count - right_dark_count)

        # ======== 2. 目标大小/距离（总暗区面积）========
        total_darkness = (left_darkness + right_darkness)
        norm_total = np.clip(total_darkness / 100.0, 0, 1)

        # ======== 3. 对比度（场景中是否有显著物体）========
        left_var  = np.var(left)
        right_var = np.var(right)
        contrast = np.clip((left_var + right_var) * 10, 0, 1)

        # ======== 4. 运动检测（帧间变化）========
        if self.prev_left is not None:
            # 左右眼的帧间亮度变化量
            left_motion  = np.mean(np.abs(left - self.prev_left))
            right_motion = np.mean(np.abs(right - self.prev_right))
            # 运动方向：哪只眼变化更大 = 目标在向那边运动
            raw_motion_dir = (left_motion - right_motion)
            motion_dir = np.clip(raw_motion_dir * 50, -1, 1)
            # Looming：总变化量增大 = 目标接近中
            total_motion = left_motion + right_motion
            raw_looming = np.clip(total_motion * 20, 0, 1)
        else:
            motion_dir = 0.0
            raw_looming = 0.0

        # 保存当前帧
        self.prev_left = left.copy()
        self.prev_right = right.copy()

        # ======== 5. 时间平滑（模拟神经回路时间常数）========
        self.smooth_direction      = self._ema(self.smooth_direction, norm_direction)
        self.smooth_direction_sign = np.sign(self.smooth_direction)
        self.smooth_total_dark     = self._ema(self.smooth_total_dark, norm_total)
        self.smooth_motion_dir     = self._ema(self.smooth_motion_dir, motion_dir)
        self.smooth_looming        = self._ema(self.smooth_looming, raw_looming)
        self.smooth_contrast       = self._ema(self.smooth_contrast, contrast)

        # ======== 6. 墙壁信息（已知固定环境）========
        wall_dist = abs(fly_pos[1] - wall_y)
        in_wall_range = float(wall_x_range[0] <= fly_pos[0] <= wall_x_range[1])
        wall_proximity = np.clip(1.0/(wall_dist+1.0), 0, 1) * in_wall_range

        # 墙方向（通过复眼也能看到墙——大片灰色区域）
        # 但灰色墙和暗色目标难以区分，用已知位置更可靠
        wall_front_factor = float(fly_pos[1] < wall_y) * in_wall_range
        wall_norm_dist = np.clip(wall_dist / 15.0, 0, 1)

        # ======== 组装12维特征 ========
        features = np.array([
            self.smooth_direction,           # [0] 复眼方向信号（时间平滑）
            self.smooth_total_dark,          # [1] 总暗区（≈目标近/大）
            self.smooth_contrast,            # [2] 对比度（有无物体）
            self.smooth_direction_sign,      # [3] 方向符号
            self.smooth_motion_dir,          # [4] 运动方向（帧间差异）
            self.smooth_looming,             # [5] looming（帧间变化量）
            wall_proximity,                   # [6] 墙接近度
            float(np.sign(fly_pos[0] - (wall_x_range[0]+wall_x_range[1])/2)),  # [7] 墙方位
            wall_front_factor,               # [8] 墙在前方
            norm_total,                       # [9] 瞬时总暗区
            np.clip(1.0 - self.smooth_total_dark, 0, 1),  # [10] 距离估计（暗区少=远）
            wall_norm_dist,                   # [11] 归一化墙距离
        ], dtype=np.float32)

        return features, dark_diff


def get_fly_b_position(step):
    """果蝇B沿椭圆路径运动"""
    theta = step * FLY_B_SPEED
    x = FLY_B_CENTER[0] + FLY_B_RADIUS_X * np.cos(theta)
    y = FLY_B_CENTER[1] + FLY_B_RADIUS_Y * np.sin(theta)
    return np.array([x, y])

def get_fly_b_velocity(step):
    """果蝇B的速度"""
    theta = step * FLY_B_SPEED
    vx = -FLY_B_RADIUS_X * FLY_B_SPEED * np.sin(theta)
    vy =  FLY_B_RADIUS_Y * FLY_B_SPEED * np.cos(theta)
    return np.array([vx, vy])


# ============ 复眼帧生成 ============
def make_eye_frame(vision, step, dist, dark_diff,
                   left, right, id_map,
                   dn_left_hz, dn_right_hz,
                   fly_b_pos, wall_dist):
    H_map, W_map = id_map.shape

    # ★ 先用有效小眼填充，再用最近邻插值填补间隙 ★
    left_img = np.full((H_map, W_map), np.nan)
    right_img = np.full((H_map, W_map), np.nan)
    for row in range(H_map):
        for col in range(W_map):
            oid = id_map[row, col]
            if 0 <= oid < 721:
                left_img[row, col] = vision[0, oid, 0]
                right_img[row, col] = vision[1, oid, 0]

    # 用最近邻插值填补 NaN（小眼间隙）
    from scipy.ndimage import distance_transform_edt
    for img in [left_img, right_img]:
        mask = np.isnan(img)
        if mask.any() and (~mask).any():
            # 找每个NaN像素最近的有效像素
            _, nearest_idx = distance_transform_edt(
                mask, return_distances=True, return_indices=True)
            img[mask] = img[tuple(nearest_idx[:, mask])]

    fig = plt.figure(figsize=(14, 4), facecolor='black')
    ax1 = fig.add_axes([0.02, 0.15, 0.30, 0.75])
    ax2 = fig.add_axes([0.34, 0.15, 0.30, 0.75])
    ax3 = fig.add_axes([0.67, 0.15, 0.12, 0.75])
    ax4 = fig.add_axes([0.82, 0.15, 0.16, 0.75])

    ax1.imshow(left_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Left Eye', color='white', fontsize=10)
    ax1.axis('off')

    ax2.imshow(right_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Right Eye', color='white', fontsize=10)
    ax2.axis('off')

    # DN柱状图
    bars = ax3.bar(['DN_L', 'DN_R'],
                    [dn_left_hz, dn_right_hz],
                    color=['blue', 'red'], alpha=0.8)
    ax3.set_ylim(0, 200)
    ax3.set_title('DN Hz', color='white', fontsize=9)
    ax3.tick_params(colors='white', labelsize=7)
    ax3.set_facecolor('black')
    for spine in ax3.spines.values():
        spine.set_color('white')

    # 小地图
    ax4.set_facecolor('black')
    ax4.set_xlim(-5, 35)
    ax4.set_ylim(-10, 25)
    # 墙
    ax4.fill_between([WALL_X_MIN, WALL_X_MAX],
                      WALL_POS_Y - 0.5, WALL_POS_Y + 0.5,
                      color='gray', alpha=0.8)
    ax4.text((WALL_X_MIN+WALL_X_MAX)/2, WALL_POS_Y+1.5,
             'WALL', color='gray', ha='center', fontsize=7)
    # 果蝇B
    ax4.scatter(fly_b_pos[0], fly_b_pos[1], s=100, c='orange',
                marker='o', zorder=5, label='Fly B')
    ax4.set_title('Map', color='white', fontsize=9)
    ax4.tick_params(colors='white', labelsize=6)
    ax4.legend(fontsize=6, loc='upper left',
               facecolor='black', labelcolor='white')
    for spine in ax4.spines.values():
        spine.set_color('white')

    direction = ('LEFT' if dark_diff > 5
                 else 'RIGHT' if dark_diff < -5
                 else 'FRONT')
    leg_str = 'TURN LEFT' if left < right else 'TURN RIGHT'
    fig.text(0.5, 0.03,
        f"Step:{step:5d} | Dist:{dist:.1f}mm | "
        f"WallDist:{wall_dist:.1f}mm | "
        f"FlyB:{direction} | {leg_str} | "
        f"L={left:.2f} R={right:.2f}",
        ha='center', color='yellow',
        fontsize=9, fontfamily='monospace')
    fig.text(0.5, 0.97,
        'Type-Specific LC: Eye → LC(5 groups) → Central → '
        'DN_L+DN_R → VNC → Legs | Obstacle Chase',
        ha='center', color='white',
        fontsize=9, fontweight='bold')

    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf


# ============ 加载模型 ============
print("Loading type-specific architecture model...")
W    = np.load('20260416_full_pathway_W_cleft.npy')
meta = pd.read_csv('20260416_full_pathway_meta_cleft.csv')
df_dn_lat = pd.read_csv('20260416_dn_with_laterality.csv')

lc_idx      = meta[meta['is_LC']==True].index.values
central_idx = meta[meta['is_central']==True].index.values
dn_left_ids  = df_dn_lat[df_dn_lat['laterality']=='left']['root_id'].tolist()
dn_right_ids = df_dn_lat[df_dn_lat['laterality']=='right']['root_id'].tolist()
dn_left_meta = meta[meta['root_id'].isin(dn_left_ids)]
dn_right_meta = meta[meta['root_id'].isin(dn_right_ids)]
dn_left_idx  = dn_left_meta.index.values
dn_right_idx = dn_right_meta.index.values
dn_idx_all   = np.concatenate([dn_left_idx, dn_right_idx])

lc_grouper = LCTypeGrouper(W, lc_idx, central_idx, n_groups=5)
snn = TypeSpecificSNN(W, lc_idx, central_idx,
                      dn_left_idx, dn_right_idx, dn_idx_all,
                      lc_grouper, vis_dim=12)
snn.load_state_dict(
    torch.load('20260416_obstacle_chase_best.pt'))
snn.eval()

print(f"LC: {len(lc_idx)} ({lc_grouper.n_groups} groups)")
print(f"Central: {len(central_idx)}")
print(f"DN_L: {len(dn_left_idx)}  DN_R: {len(dn_right_idx)}")
lc_grouper.print_summary()


# ============ 创建场景 ============
print(f"\nFly B path: ellipse center={FLY_B_CENTER}")
print(f"Wall: y={WALL_POS_Y}, x=[{WALL_X_MIN}, {WALL_X_MAX}]")

fly = flygym.Fly(
    enable_adhesion=True, init_pose='tripod',
    enable_vision=True, vision_refresh_rate=500,
    contact_sensor_placements=[
        f"{lg}{sg}"
        for lg in ["LF","LM","LH","RF","RM","RH"]
        for sg in ["Tibia","Tarsus1","Tarsus2",
                   "Tarsus3","Tarsus4","Tarsus5"]
    ],
)
arena = flygym.arena.FlatTerrain()

# ======== 添加墙壁 ========
wall_body = arena.root_element.worldbody.add(
    'body', name='wall',
    pos=[(WALL_X_MIN+WALL_X_MAX)/2, WALL_POS_Y, WALL_HEIGHT/2])
wall_body.add('geom', name='wall_geom',
    type='box',
    size=[(WALL_X_MAX-WALL_X_MIN)/2, 0.5, WALL_HEIGHT/2],
    rgba=[0.5, 0.5, 0.5, 1.0],
    contype='1', conaffinity='1')

# ======== 添加果蝇B（移动体，5倍大，确保复眼可靠检测）========
# 初始位置
fly_b_init = get_fly_b_position(0)
fly_b_body = arena.root_element.worldbody.add(
    'body', name='fly_b',
    pos=[fly_b_init[0], fly_b_init[1], 2.0])  # 抬高，在天空背景前更显眼
# 果蝇B的身体（纯黑，5倍大小）
fly_b_body.add('geom', name='fly_b_thorax',
    type='ellipsoid', size=[4.0, 2.0, 1.5],
    rgba=[0.02, 0.01, 0.01, 1.0],
    contype='0', conaffinity='0')
fly_b_body.add('geom', name='fly_b_head',
    type='sphere', size=[1.5],
    pos=[4.5, 0, 0.5],
    rgba=[0.03, 0.01, 0.01, 1.0],
    contype='0', conaffinity='0')
# 翅膀（5倍，深色不透明）
fly_b_body.add('geom', name='fly_b_wing_l',
    type='ellipsoid', size=[3.0, 0.75, 0.25],
    pos=[-1.0, 2.5, 1.0],
    rgba=[0.05, 0.05, 0.05, 0.9],
    contype='0', conaffinity='0')
fly_b_body.add('geom', name='fly_b_wing_r',
    type='ellipsoid', size=[3.0, 0.75, 0.25],
    pos=[-1.0, -2.5, 1.0],
    rgba=[0.05, 0.05, 0.05, 0.9],
    contype='0', conaffinity='0')

# 路径标记点（可视化果蝇B的路径）
for theta_mark in np.linspace(0, 2*np.pi, 12, endpoint=False):
    mx = FLY_B_CENTER[0] + FLY_B_RADIUS_X * np.cos(theta_mark)
    my = FLY_B_CENTER[1] + FLY_B_RADIUS_Y * np.sin(theta_mark)
    marker = arena.root_element.worldbody.add(
        'body', name=f'path_mark_{int(np.degrees(theta_mark))}',
        pos=[mx, my, 0.05])
    marker.add('geom', type='sphere', size=[0.15],
        rgba=[1, 0.6, 0, 0.3],
        contype='0', conaffinity='0')

cam = flygym.Camera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_zoomout",
    play_speed=0.15,
    output_path="./20260416_obstacle_chase.mp4")
ctrl = HybridTurningController(
    fly=fly, arena=arena, cameras=cam, timestep=1e-4)
obs, _ = ctrl.reset()

id_map = fly.retina.ommatidia_id_map
print(f"Retina: {fly.retina.num_ommatidia_per_eye} ommatidia/eye")

eye_writer = imageio.get_writer(
    './20260416_obstacle_chase_eye.mp4', fps=20, quality=7)


# ============ 主循环 ============
state = snn.reset(batch=1)
positions = []; dist_hist = []; wall_dist_hist = []
left_hist = []; right_hist = []
dn_left_hist = []; dn_right_hist = []
fly_b_positions = []
min_dist = 999.0
close_count = 0

# ★ 纯复眼视觉处理器 ★
eye_processor = CompoundEyeProcessor(ema_alpha=0.12)

print(f"\nRunning {N_STEPS} steps...")
print("  ★ All fly-B features derived from compound eye ONLY ★")

with mujoco.viewer.launch_passive(
    ctrl.unwrapped.physics.model.ptr,
    ctrl.unwrapped.physics.data.ptr,
) as viewer:
    viewer.cam.azimuth   = 45
    viewer.cam.elevation = -35
    viewer.cam.distance  = 60.0

    for i in range(N_STEPS):
        if not viewer.is_running():
            break

        fly_pos = obs['fly'][0][:2]
        positions.append(fly_pos.copy())
        vision = obs['vision']

        # 更新果蝇B位置（场景管理，非视觉作弊）
        fly_b_pos = get_fly_b_position(i)
        fly_b_vel = get_fly_b_velocity(i)
        fly_b_positions.append(fly_b_pos.copy())

        # 移动果蝇B的MuJoCo体
        try:
            physics = ctrl.unwrapped.physics
            physics.named.model.body_pos['fly_b'] = [
                fly_b_pos[0], fly_b_pos[1], 1.5]
            speed = np.linalg.norm(fly_b_vel)
            if speed > 1e-6:
                heading_b = np.arctan2(fly_b_vel[1], fly_b_vel[0])
                quat = [np.cos(heading_b/2), 0, 0, np.sin(heading_b/2)]
                physics.named.model.body_quat['fly_b'] = quat
        except Exception as e:
            try:
                model_ptr = ctrl.unwrapped.physics.model.ptr
                body_id = mujoco.mj_name2id(
                    model_ptr, mujoco.mjtObj.mjOBJ_BODY, 'fly_b')
                model_ptr.body_pos[body_id] = [
                    fly_b_pos[0], fly_b_pos[1], 1.5]
            except Exception as e2:
                if i == 0:
                    print(f"Warning: Cannot move fly B: {e2}")

        # ★ 纯复眼特征提取 —— 不传入fly_b_pos/vel ★
        vis_feat, dark_diff = eye_processor.process(
            vision, fly_pos,
            WALL_POS_Y, (WALL_X_MIN, WALL_X_MAX))

        # 距离仅用于日志和绘图，不送入SNN
        dist = np.sqrt(((fly_pos - fly_b_pos)**2).sum())
        wall_dist = abs(fly_pos[1] - WALL_POS_Y)
        dist_hist.append(dist)
        wall_dist_hist.append(wall_dist)
        min_dist = min(min_dist, dist)
        if dist < 5.0:
            close_count += 1

        # 本体感觉
        joints = obs.get('joints', np.zeros((3, 42)))[0]
        proprio = (joints[:12] / np.pi).astype(np.float32)
        vis_t  = torch.FloatTensor(vis_feat).unsqueeze(0)
        prop_t = torch.FloatTensor(proprio).unsqueeze(0)

        with torch.no_grad():
            cmd, state, rates, turn = snn(vis_t, prop_t, state)

        left  = float(np.clip(cmd[0, 0].detach(), 0.1, 1.5))
        right = float(np.clip(cmd[0, 1].detach(), 0.1, 1.5))
        left_hist.append(left)
        right_hist.append(right)

        dl = rates['dn_left'].mean().item()
        dr = rates['dn_right'].mean().item()
        dn_left_hist.append(dl)
        dn_right_hist.append(dr)

        try:
            obs, _, _, _, _ = ctrl.step(
                np.array([left, right], dtype=np.float32))
        except Exception as e:
            print(f"Error: {e}")
            break

        if i % RENDER_EVERY == 0:
            ctrl.render()
        if i % EYE_EVERY == 0:
            frame = make_eye_frame(
                vision, i, dist, dark_diff,
                left, right, id_map, dl, dr,
                fly_b_pos, wall_dist)
            eye_writer.append_data(frame)
        viewer.sync()

        if i % 5000 == 0:
            feat0 = vis_feat[0]  # direction feature sent to SNN
            print(f"  Step {i:5d} | "
                  f"dist_B={dist:.1f}mm | "
                  f"wall={wall_dist:.1f}mm | "
                  f"dark_diff={dark_diff:+.0f} feat[0]={feat0:+.2f} | "
                  f"DN_L={dl:.0f} DN_R={dr:.0f}Hz | "
                  f"L={left:.2f} R={right:.2f} | "
                  f"{'LEFT' if left<right else 'RIGHT'}")

eye_writer.close()
ctrl.close()

print(f"\nMin dist to fly B: {min_dist:.2f}mm")
print(f"Close count (<5mm): {close_count} steps "
      f"({100*close_count/N_STEPS:.1f}%)")


# ============ 结果图 ============
positions = np.array(positions)
fly_b_positions = np.array(fly_b_positions)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- 轨迹图 ----
colors = plt.cm.plasma(np.linspace(0, 1, len(positions)))
for k in range(len(positions)-1):
    axes[0,0].plot(positions[k:k+2,0], positions[k:k+2,1],
                   color=colors[k], linewidth=1.5)
# 果蝇B轨迹
for k in range(0, len(fly_b_positions)-1, 10):
    axes[0,0].plot(fly_b_positions[k:k+2,0],
                   fly_b_positions[k:k+2,1],
                   color='orange', linewidth=0.5, alpha=0.3)
# 墙
axes[0,0].fill_between(
    [WALL_X_MIN, WALL_X_MAX],
    WALL_POS_Y - 0.5, WALL_POS_Y + 0.5,
    color='gray', alpha=0.8, label='Wall')
axes[0,0].scatter(*positions[0], s=300, c='lime',
                   marker='*', zorder=6, label='Fly A Start')
axes[0,0].scatter(*positions[-1], s=300, c='blue',
                   marker='D', zorder=6, label='Fly A End')
axes[0,0].scatter(*fly_b_positions[-1], s=300, c='orange',
                   marker='o', zorder=7, label='Fly B (last)')

sm = plt.cm.ScalarMappable(
    cmap='plasma', norm=plt.Normalize(0, len(positions)))
sm.set_array([])
plt.colorbar(sm, ax=axes[0,0], label='Time', shrink=0.8)
axes[0,0].set_title(
    'Fly A Trajectory (chase Fly B, avoid wall)\n'
    'Type-Specific LC → Central → DN(L/R) → Legs',
    fontsize=11)
axes[0,0].set_xlabel('X (mm)')
axes[0,0].set_ylabel('Y (mm)')
axes[0,0].legend(fontsize=7)
axes[0,0].axis('equal')
axes[0,0].grid(True, alpha=0.3)

# ---- 距离曲线 ----
axes[0,1].plot(dist_hist, color='purple', linewidth=1, label='Dist to Fly B')
axes[0,1].plot(wall_dist_hist, color='gray', linewidth=1,
               alpha=0.6, label='Dist to Wall')
axes[0,1].axhline(y=5.0, color='orange', linestyle='--',
                   linewidth=1.5, label='Close threshold (5mm)')
axes[0,1].axhline(y=min_dist, color='green', linestyle=':',
                   linewidth=1.5, label=f'Best={min_dist:.1f}mm')
axes[0,1].set_title('Distances', fontsize=12)
axes[0,1].set_xlabel('Step')
axes[0,1].set_ylabel('mm')
axes[0,1].legend(fontsize=8)
axes[0,1].grid(True, alpha=0.3)

# ---- DN发放率 ----
axes[1,0].plot(dn_left_hist, color='blue',
               linewidth=0.8, alpha=0.7, label='DN_left')
axes[1,0].plot(dn_right_hist, color='red',
               linewidth=0.8, alpha=0.7, label='DN_right')
axes[1,0].set_title('DN Left vs Right Firing Rate', fontsize=12)
axes[1,0].set_xlabel('Step')
axes[1,0].set_ylabel('Hz')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# ---- 腿速度 ----
axes[1,1].plot(left_hist, color='blue',
               linewidth=0.8, alpha=0.7, label='Left leg')
axes[1,1].plot(right_hist, color='red',
               linewidth=0.8, alpha=0.7, label='Right leg')
axes[1,1].axhline(y=0.8, color='gray', linestyle='--',
                   linewidth=1, label='Baseline')
axes[1,1].set_title('Leg Speed Commands', fontsize=12)
axes[1,1].set_xlabel('Step')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle(
    f'Type-Specific LC Obstacle Chase: '
    f'Eye → LC({len(lc_idx)}, {lc_grouper.n_groups} groups) → '
    f'Central({len(central_idx)}) → '
    f'DN_L({len(dn_left_idx)})+DN_R({len(dn_right_idx)}) → VNC\n'
    f'FlyWire FAFB | MinDist={min_dist:.1f}mm | '
    f'CloseTime={100*close_count/N_STEPS:.1f}%',
    fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('20260416_obstacle_chase_result.png', dpi=150)
print("\nSaved: 20260416_obstacle_chase_result.png")
print()
print("Videos:")
print("  vlc ./20260416_obstacle_chase.mp4")
print("  vlc ./20260416_obstacle_chase_eye.mp4")
