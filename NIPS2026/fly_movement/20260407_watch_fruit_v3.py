"""
果蝇追苹果 v3 - 真实复眼视觉版本
同时录制：
1. 俯视追踪视频（fruit_chase_v3.mp4）
2. 复眼视角视频（eye_view_v3.mp4）
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

# ============ 模型定义 ============
class LIFNeurons(nn.Module):
    def __init__(self, n, tau=20.0, v_th=0.3,
                 v_rest=0.0, v_reset=-0.05, dt=1.0):
        super().__init__()
        self.tau=tau; self.v_th=v_th
        self.v_rest=v_rest; self.v_reset=v_reset; self.dt=dt
        self.bias=nn.Parameter(torch.zeros(n))
    def forward(self, v, I):
        dv=(self.v_rest-v+I+self.bias)*(self.dt/self.tau)
        v_new=v+dv
        sp=(v_new>=self.v_th).float()
        v_new=v_new*(1-sp)+self.v_reset*sp
        return v_new, sp

class VisualSNN_Direct(nn.Module):
    def __init__(self, W_matrix, lc_idx, dn_idx,
                 motor_idx, C=2):
        super().__init__()
        self.N=W_matrix.shape[0]; self.C=C
        self.lc_idx   =torch.LongTensor(lc_idx)
        self.dn_idx   =torch.LongTensor(dn_idx)
        self.motor_idx=torch.LongTensor(motor_idx)
        self.register_buffer('W', torch.FloatTensor(W_matrix))
        self.turn_pathway=nn.Sequential(
            nn.Linear(1,32), nn.Tanh(),
            nn.Linear(32,16), nn.Tanh(),
            nn.Linear(16,1), nn.Tanh())
        self.eye=nn.Sequential(
            nn.Linear(5,64), nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,len(lc_idx)*C), nn.Sigmoid())
        self.proprio_net=nn.Sequential(
            nn.Linear(12,32), nn.ReLU(),
            nn.Linear(32,len(dn_idx)*C), nn.Tanh())
        self.eta=nn.Parameter(torch.randn(self.N,C)*0.05)
        self.lif=LIFNeurons(n=self.N*C)
        self.update_mlp=nn.Sequential(
            nn.Linear(C+C,C), nn.Tanh())
        self.forward_decoder=nn.Sequential(
            nn.Linear(len(motor_idx)*C,16), nn.ReLU(),
            nn.Linear(16,1), nn.Tanh())
    def forward(self, vis, prop, H=None, V=None):
        batch=vis.shape[0]; dev=vis.device
        if H is None: H=torch.zeros(batch,self.N,self.C,device=dev)
        if V is None: V=torch.zeros(batch,self.N*self.C,device=dev)
        turn=self.turn_pathway(vis[:,0:1])
        lc_in=self.eye(vis).reshape(batch,len(self.lc_idx),self.C)
        H=H.clone(); H[:,self.lc_idx,:]+=lc_in
        dn_in=self.proprio_net(prop).reshape(
            batch,len(self.dn_idx),self.C)
        H[:,self.dn_idx,:]+=dn_in
        M=torch.einsum('ij,bjc->bic',self.W,H)
        et=self.eta.unsqueeze(0).expand(batch,-1,-1)
        Ht=self.update_mlp(torch.cat([M,et],dim=-1))
        V_new,sp=self.lif(V,Ht.reshape(batch,self.N*self.C))
        H_new=sp.reshape(batch,self.N,self.C)
        fwd=self.forward_decoder(
            H_new[:,self.motor_idx,:].reshape(batch,-1))
        cmd=torch.cat([fwd,turn],dim=1)
        return cmd, H_new, V_new, sp
    def reset_state(self,batch=1,device='cpu'):
        return (torch.zeros(batch,self.N,self.C,device=device),
                torch.zeros(batch,self.N*self.C,device=device))

# ============ 参数 ============
FRUIT_POS_MM   = np.array([30.0, 0.0])
N_STEPS        = 22000
RENDER_EVERY   = 50    # 俯视视频帧率
EYE_EVERY      = 100   # 复眼视频帧率
DARK_THRESHOLD = 0.2
MAX_DARK_DIFF  = 50.0

# ============ 加载模型 ============
print("Loading model...")
W    = np.load('20260407_visual_pathway_W.npy')
meta = pd.read_csv('20260407_visual_pathway_meta.csv')
lc_idx    = meta[meta['is_LC']==True].index.values
dn_idx    = meta[meta['is_DN']==True].index.values
motor_idx = meta[meta['is_motor']==True].index.values

snn = VisualSNN_Direct(W, lc_idx, dn_idx, motor_idx, C=2)
snn.load_state_dict(torch.load('20260407_direct_best.pt'))
snn.eval()
print(f"LC:{len(lc_idx)} DN:{len(dn_idx)} Motor:{len(motor_idx)}")

# ============ 视觉特征提取 ============
def extract_visual_features(vision, fly_pos, fruit_pos):
    left  = vision[0,:,0]
    right = vision[1,:,0]
    left_dark  = (left  < DARK_THRESHOLD).sum()
    right_dark = (right < DARK_THRESHOLD).sum()
    dark_diff  = float(left_dark - right_dark)
    dark_diff_norm = np.clip(dark_diff/MAX_DARK_DIFF,-1,1)
    dark_sign  = float(np.sign(dark_diff))
    dx = fruit_pos[0]-fly_pos[0]
    dy = fruit_pos[1]-fly_pos[1]
    dist = np.sqrt(dx**2+dy**2)
    dist_feat = np.clip(1.0/(dist+1.0),0,1)
    total_dark = np.clip(
        float(left_dark+right_dark)/100.0,0,1)
    brightness_diff = np.clip(
        float(left.mean()-right.mean())*10,-1,1)
    return np.array([
        dark_diff_norm, dist_feat,
        total_dark, dark_sign, brightness_diff
    ], dtype=np.float32)

def get_dist(fly_pos, fruit_pos):
    return np.sqrt(((fly_pos-fruit_pos)**2).sum())

def snn_to_legs(cmd_fwd, cmd_turn):
    speed = float(cmd_fwd)*0.3+0.8
    turn  = float(cmd_turn)*0.5
    return (np.clip(speed-turn,0.1,1.5),
            np.clip(speed+turn,0.1,1.5))

# ============ 复眼帧生成 ============
def make_eye_frame(vision, step, dist, dark_diff,
                   left_leg, right_leg, id_map):
    """生成复眼视角的视频帧（RGB图像）"""
    # 重建左右眼图像
    H, W_map = id_map.shape
    left_img  = np.ones((H, W_map)) * 0.5
    right_img = np.ones((H, W_map)) * 0.5

    for row in range(H):
        for col in range(W_map):
            oid = id_map[row, col]
            if 0 <= oid < 721:
                left_img[row,col]  = vision[0,oid,0]
                right_img[row,col] = vision[1,oid,0]

    # 生成图像帧
    fig = plt.figure(figsize=(10,4), facecolor='black')

    ax1 = fig.add_axes([0.02, 0.15, 0.44, 0.75])
    ax2 = fig.add_axes([0.54, 0.15, 0.44, 0.75])

    ax1.imshow(left_img,  cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Left Eye', color='white', fontsize=11)
    ax1.axis('off')

    ax2.imshow(right_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Right Eye', color='white', fontsize=11)
    ax2.axis('off')

    # 状态信息
    direction = 'LEFT' if dark_diff > 0 else ('RIGHT' if dark_diff < 0 else 'FRONT')
    leg_str   = 'TURN LEFT' if left_leg < right_leg else 'TURN RIGHT'
    info = (f"Step:{step:5d} | Dist:{dist:.1f}mm | "
            f"Apple:{direction} | {leg_str} | "
            f"L={left_leg:.2f} R={right_leg:.2f}")
    fig.text(0.5, 0.04, info,
             ha='center', color='yellow',
             fontsize=9, fontfamily='monospace')
    fig.text(0.5, 0.97,
             'Drosophila Compound Eye View (721 ommatidia per eye)',
             ha='center', color='white',
             fontsize=10, fontweight='bold')

    # 转成numpy图像
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    buf = buf[:,:,:3]  # 去掉alpha通道，保留RGB
    plt.close(fig)
    return buf

# ============ 创建场景 ============
print(f"Fruit: {FRUIT_POS_MM} mm")
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
fruit_body = arena.root_element.worldbody.add(
    'body', name='fruit',
    pos=[FRUIT_POS_MM[0], FRUIT_POS_MM[1], 1.5])
fruit_body.add('geom', name='fruit_main',
    type='sphere', size=[1.4],
    rgba=[0.95,0.1,0.05,1.0],
    contype='0', conaffinity='0')
fruit_body.add('geom', name='fruit_stem',
    type='cylinder', size=[0.15,0.8],
    pos=[0,0,1.5], rgba=[0.25,0.12,0.04,1.0],
    contype='0', conaffinity='0')
fruit_body.add('geom', name='fruit_leaf',
    type='ellipsoid', size=[0.6,0.2,0.12],
    pos=[0.5,0,1.9], rgba=[0.1,0.75,0.1,1.0],
    contype='0', conaffinity='0')

cam = flygym.Camera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_zoomout",
    play_speed=0.15,
    output_path="./20260407_fruit_chase_v3.mp4")
ctrl = HybridTurningController(
    fly=fly, arena=arena, cameras=cam, timestep=1e-4)
obs,_ = ctrl.reset()

# 获取retina id_map
id_map = fly.retina.ommatidia_id_map

print(f"Retina: {fly.retina.num_ommatidia_per_eye} ommatidia/eye")
print(f"id_map shape: {id_map.shape}")

# ============ 初始化视频写入器 ============
eye_writer = imageio.get_writer(
    './20260407_eye_view_v3.mp4',
    fps=20, quality=7)
print("Eye view video writer ready!")

# ============ 主循环 ============
H,V = snn.reset_state(batch=1)
positions=[]; dist_hist=[]; turn_hist=[]
left_hist=[]; right_hist=[]; dark_diff_hist=[]
success=False

print(f"\nRunning {N_STEPS} steps...")
print("Recording both top-view AND eye-view videos!")

with mujoco.viewer.launch_passive(
    ctrl.unwrapped.physics.model.ptr,
    ctrl.unwrapped.physics.data.ptr,
) as viewer:
    viewer.cam.azimuth   = 45
    viewer.cam.elevation = -35
    viewer.cam.distance  = 35.0

    for i in range(N_STEPS):
        if not viewer.is_running():
            break

        fly_pos = obs['fly'][0][:2]
        positions.append(fly_pos.copy())
        vision  = obs['vision']
        dist    = get_dist(fly_pos, FRUIT_POS_MM)
        dist_hist.append(dist)

        vis_feat = extract_visual_features(
            vision, fly_pos, FRUIT_POS_MM)
        dark_diff = float(
            (vision[0,:,0]<DARK_THRESHOLD).sum() -
            (vision[1,:,0]<DARK_THRESHOLD).sum())
        dark_diff_hist.append(vis_feat[0])

        joints  = obs.get('joints',np.zeros((3,42)))[0]
        proprio = (joints[:12]/np.pi).astype(np.float32)
        vis_t   = torch.FloatTensor(vis_feat).unsqueeze(0)
        prop_t  = torch.FloatTensor(proprio).unsqueeze(0)

        with torch.no_grad():
            cmd,H,V,sp = snn(vis_t,prop_t,H,V)

        left, right = snn_to_legs(
            cmd[0,0].item(), cmd[0,1].item())
        left_hist.append(left)
        right_hist.append(right)
        turn_hist.append(left-right)

        try:
            obs,_,_,_,_=ctrl.step(
                np.array([left,right],dtype=np.float32))
        except Exception as e:
            print(f"Error: {e}"); break

        # 俯视视频
        if i%RENDER_EVERY==0:
            ctrl.render()

        # 复眼视频
        if i%EYE_EVERY==0:
            frame = make_eye_frame(
                vision, i, dist, dark_diff,
                left, right, id_map)
            eye_writer.append_data(frame)

        viewer.sync()

        if i%3000==0:
            print(f"  Step {i:5d} | "
                  f"dist={dist:.2f}mm | "
                  f"dark_diff={dark_diff:+.0f} | "
                  f"L={left:.2f} R={right:.2f} | "
                  f"{'LEFT' if left<right else 'RIGHT'}")

        if dist<2.0 and not success:
            success=True
            print(f"\n  *** REACHED APPLE at step {i}! ***\n")

eye_writer.close()
ctrl.close()
best_dist=min(dist_hist)
print(f"\nBest: {best_dist:.3f}mm | Success: {success}")
print("Videos saved:")
print("  20260407_fruit_chase_v3.mp4  (top view)")
print("  20260407_eye_view_v3.mp4     (compound eye view)")

# ============ 结果图 ============
positions=np.array(positions)
fig,axes=plt.subplots(2,2,figsize=(14,11))

colors=plt.cm.plasma(np.linspace(0,1,len(positions)))
for k in range(len(positions)-1):
    axes[0,0].plot(positions[k:k+2,0],positions[k:k+2,1],
                   color=colors[k],linewidth=2)
axes[0,0].scatter(*positions[0],s=300,c='lime',
                   marker='*',zorder=6,label='Start')
axes[0,0].scatter(*positions[-1],s=300,c='blue',
                   marker='D',zorder=6,label='End')
axes[0,0].scatter(*FRUIT_POS_MM,s=600,c='red',
                   marker='o',zorder=7,label='Apple')
circle=plt.Circle(FRUIT_POS_MM,2.0,color='red',
                   fill=False,linestyle='--',linewidth=2)
axes[0,0].add_patch(circle)
sm=plt.cm.ScalarMappable(cmap='plasma',
    norm=plt.Normalize(0,len(positions)))
sm.set_array([])
plt.colorbar(sm,ax=axes[0,0],label='Time',shrink=0.8)
axes[0,0].set_title('Fly Trajectory v3\nReal compound eye vision',
                     fontsize=12)
axes[0,0].set_xlabel('X (mm)'); axes[0,0].set_ylabel('Y (mm)')
axes[0,0].legend(fontsize=8); axes[0,0].axis('equal')
axes[0,0].grid(True,alpha=0.3)

axes[0,1].plot(dist_hist,color='purple',linewidth=1.5)
axes[0,1].axhline(y=2.0,color='red',linestyle='--',
                   linewidth=2,label='Success (2mm)')
axes[0,1].axhline(y=best_dist,color='green',
                   linestyle=':',linewidth=1.5,
                   label=f'Best={best_dist:.2f}mm')
axes[0,1].set_title('Distance to Apple',fontsize=12)
axes[0,1].set_xlabel('Step'); axes[0,1].set_ylabel('mm')
axes[0,1].legend(); axes[0,1].grid(True,alpha=0.3)

axes[1,0].plot(dark_diff_hist,color='orange',
               linewidth=1,alpha=0.8)
axes[1,0].axhline(y=0,color='black',linewidth=0.8)
axes[1,0].fill_between(range(len(dark_diff_hist)),
                        dark_diff_hist,
                        alpha=0.2,color='orange')
axes[1,0].set_title('Visual Signal: Dark Diff\n'
                     '(+left darker=apple left, '
                     '-right darker=apple right)',fontsize=11)
axes[1,0].set_xlabel('Step')
axes[1,0].grid(True,alpha=0.3)

axes[1,1].plot(left_hist,color='blue',linewidth=1,
               alpha=0.7,label='Left legs')
axes[1,1].plot(right_hist,color='red',linewidth=1,
               alpha=0.7,label='Right legs')
axes[1,1].set_title('Leg Speed\n(left<right = turn left)',
                     fontsize=12)
axes[1,1].set_xlabel('Step')
axes[1,1].legend(); axes[1,1].grid(True,alpha=0.3)

plt.suptitle(
    f'FlyGM-SNN v3: Real Compound Eye Vision\n'
    f'LC({len(lc_idx)})->DN({len(dn_idx)})->Motor({len(motor_idx)}) | '
    f'721 ommatidia | Best={best_dist:.2f}mm | Success={success}',
    fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig('20260407_fruit_chase_v3_result.png',dpi=150)
print("Saved: 20260407_fruit_chase_v3_result.png")
print()
print("Play videos:")
print("  vlc ./20260407_fruit_chase_v3.mp4")
print("  vlc ./20260407_eye_view_v3.mp4")
