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
DARK_THRESHOLD = 0.3   # 更敏感，更多像素被判为"暗"
MAX_DARK_DIFF  = 10.0  # 缩小归一化范围，放大方向信号（原50→10）


# ============ 视觉特征提取（12维）============
def extract_visual_features_12d(vision, fly_pos, fly_heading,
                                 fly_b_pos, fly_b_vel,
                                 wall_y, wall_x_range,
                                 prev_target_size=0.0):
    """
    从复眼图像和场景信息提取12维视觉特征

    参数：
        vision: (2, 721, 2) 复眼图像
        fly_pos: 果蝇A的位置 (x,y)
        fly_heading: 果蝇A的朝向角
        fly_b_pos: 果蝇B的位置 (x,y)
        fly_b_vel: 果蝇B的速度 (vx,vy)
        wall_y: 墙的y坐标
        wall_x_range: 墙的x范围 (min, max)
        prev_target_size: 上一步的目标大小（用于计算looming）
    """
    left = vision[0, :, 0]; right = vision[1, :, 0]
    left_dark  = (left < DARK_THRESHOLD).sum()
    right_dark = (right < DARK_THRESHOLD).sum()
    dark_diff  = float(left_dark - right_dark)

    # 目标相对位置
    dx = fly_b_pos[0] - fly_pos[0]
    dy = fly_b_pos[1] - fly_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    # 目标角度（相对朝向）
    angle_to_target = np.arctan2(dy, dx) - fly_heading
    angle_to_target = (angle_to_target + np.pi) % (2*np.pi) - np.pi

    # 目标视角大小
    target_size = np.clip(4.0 / (dist + 1.0), 0, 1)

    # Looming速度
    looming = target_size - prev_target_size

    # 目标运动速度（切向分量）
    if dist > 0.1:
        tangent_speed = (-dy*fly_b_vel[0] + dx*fly_b_vel[1]) / dist
    else:
        tangent_speed = 0.0

    # 目标径向速度
    if dist > 0.1:
        radial_speed = (dx*fly_b_vel[0] + dy*fly_b_vel[1]) / dist
    else:
        radial_speed = 0.0

    # 墙壁信息
    wall_dist = abs(fly_pos[1] - wall_y)
    in_wall_range = float(wall_x_range[0] <= fly_pos[0] <= wall_x_range[1])

    dx_wall = (wall_x_range[0] + wall_x_range[1]) / 2 - fly_pos[0]
    dy_wall = wall_y - fly_pos[1]
    angle_to_wall = np.arctan2(dy_wall, dx_wall) - fly_heading
    angle_to_wall = (angle_to_wall + np.pi) % (2*np.pi) - np.pi

    wall_in_front = float(dy_wall * np.cos(fly_heading) > 0)

    features = np.array([
        # ★ 方向信号：几何角度 × 视角大小（模拟视觉系统的输出）★
        # 原理：复眼→光学叶→LC的完整视觉处理结果就是"目标方向"
        # 视角大小加权：看不到目标时信号为0，看到时按角度给信号
        np.clip(np.sin(angle_to_target) * np.clip(target_size*3, 0, 1), -1, 1),  # [0]
        np.clip(1.0 / (dist + 1.0), 0, 1),                # [1] target_dist_inv
        target_size,                                         # [2] target_size
        float(np.sign(angle_to_target)) * float(target_size > 0.05),  # [3] 方向符号
        np.clip(abs(tangent_speed) / 5.0, 0, 1),           # [4] target_speed
        np.clip(-radial_speed / 5.0, -1, 1),               # [5] target_looming
        np.clip(1.0/(wall_dist+1.0), 0, 1)*in_wall_range,  # [6] wall_dist_inv
        np.clip(np.sin(angle_to_wall), -1, 1),             # [7] wall_direction
        np.clip(wall_in_front * in_wall_range, 0, 1),      # [8] wall_proximity
        np.clip((left_dark+right_dark)/100.0, 0, 1),       # [9] 总暗区（复眼）
        np.clip(dist / 30.0, 0, 1),                        # [10] norm_dist
        np.clip(wall_dist / 15.0, 0, 1),                   # [11] norm_wall_dist
    ], dtype=np.float32)

    return features, dark_diff, dist, target_size


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

# ======== 添加果蝇B（移动体，3倍大）========
# 初始位置
fly_b_init = get_fly_b_position(0)
fly_b_body = arena.root_element.worldbody.add(
    'body', name='fly_b',
    pos=[fly_b_init[0], fly_b_init[1], 1.5])
# 果蝇B的身体（暗色，3倍大小，确保复眼能看到）
fly_b_body.add('geom', name='fly_b_thorax',
    type='ellipsoid', size=[2.4, 1.2, 0.9],
    rgba=[0.05, 0.02, 0.02, 1.0],
    contype='0', conaffinity='0')
fly_b_body.add('geom', name='fly_b_head',
    type='sphere', size=[0.9],
    pos=[2.7, 0, 0.3],
    rgba=[0.08, 0.03, 0.03, 1.0],
    contype='0', conaffinity='0')
# 翅膀（3倍）
fly_b_body.add('geom', name='fly_b_wing_l',
    type='ellipsoid', size=[1.8, 0.45, 0.15],
    pos=[-0.6, 1.5, 0.6],
    rgba=[0.15, 0.15, 0.15, 0.6],
    contype='0', conaffinity='0')
fly_b_body.add('geom', name='fly_b_wing_r',
    type='ellipsoid', size=[1.8, 0.45, 0.15],
    pos=[-0.6, -1.5, 0.6],
    rgba=[0.15, 0.15, 0.15, 0.6],
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
prev_target_size = 0.0
min_dist = 999.0
close_count = 0  # 果蝇A靠近果蝇B的步数

print(f"\nRunning {N_STEPS} steps...")

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

        # 更新果蝇B位置
        fly_b_pos = get_fly_b_position(i)
        fly_b_vel = get_fly_b_velocity(i)
        fly_b_positions.append(fly_b_pos.copy())

        # 移动果蝇B的MuJoCo体
        # 注意：data.xpos是只读的（由前向运动学计算），
        # 必须修改model.body_pos才能移动静态body
        try:
            physics = ctrl.unwrapped.physics
            physics.named.model.body_pos['fly_b'] = [
                fly_b_pos[0], fly_b_pos[1], 1.0]
            # 计算朝向（让果蝇B面朝运动方向）
            speed = np.linalg.norm(fly_b_vel)
            if speed > 1e-6:
                heading_b = np.arctan2(fly_b_vel[1], fly_b_vel[0])
                # 设置四元数旋转（绕z轴）
                quat = [np.cos(heading_b/2), 0, 0, np.sin(heading_b/2)]
                physics.named.model.body_quat['fly_b'] = quat
        except Exception as e:
            # 回退方案：直接操作底层mujoco指针
            try:
                model_ptr = ctrl.unwrapped.physics.model.ptr
                body_id = mujoco.mj_name2id(
                    model_ptr, mujoco.mjtObj.mjOBJ_BODY, 'fly_b')
                model_ptr.body_pos[body_id] = [
                    fly_b_pos[0], fly_b_pos[1], 1.0]
            except Exception as e2:
                if i == 0:
                    print(f"Warning: Cannot move fly B: {e2}")

        # 估算朝向（用最近10步的位移方向，更稳定）
        fly_heading = 0.0
        if len(positions) > 10:
            dp = positions[-1] - positions[-10]
            if np.linalg.norm(dp) > 0.01:
                fly_heading = np.arctan2(dp[1], dp[0])
        elif len(positions) > 1:
            dp = positions[-1] - positions[0]
            if np.linalg.norm(dp) > 0.01:
                fly_heading = np.arctan2(dp[1], dp[0])

        # 提取12维视觉特征
        vis_feat, dark_diff, dist, target_size = \
            extract_visual_features_12d(
                vision, fly_pos, fly_heading,
                fly_b_pos, fly_b_vel,
                WALL_POS_Y, (WALL_X_MIN, WALL_X_MAX),
                prev_target_size)
        prev_target_size = target_size

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
