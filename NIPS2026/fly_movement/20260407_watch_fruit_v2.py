"""
果蝇追苹果 v2 - 正确版本
关键修复：
1. action=[left_legs, right_legs]（不是[forward,turn]）
2. 左转：left<right，右转：left>right
3. 用fly_orientation获取真实朝向
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
FRUIT_POS_MM = np.array([20.0, 0.0])
N_STEPS      = 30000
RENDER_EVERY = 50

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

# ============ 工具函数 ============
def get_state(obs):
    fly_pos = obs['fly'][0][:2]
    orient  = obs['fly_orientation']
    heading = np.arctan2(orient[1], orient[0])
    return fly_pos, heading

def get_visual_feat(fly_pos, heading, fruit_pos):
    dx = fruit_pos[0]-fly_pos[0]
    dy = fruit_pos[1]-fly_pos[1]
    dist = np.sqrt(dx**2+dy**2)+1e-8
    fruit_angle = np.arctan2(dy, dx)
    rel = np.arctan2(
        np.sin(fruit_angle-heading),
        np.cos(fruit_angle-heading))
    return np.array([
        np.sin(rel), np.cos(rel),
        np.clip(1.0/(dist+1.0),0,1),
        np.sign(rel),
        np.clip(dist/10.0,0,1)
    ],dtype=np.float32), rel, dist

def snn_to_legs(cmd_fwd, cmd_turn):
    """
    SNN输出 → 腿部控制
    cmd_turn > 0: 苹果在左，需要左转 → left<right
    cmd_turn < 0: 苹果在右，需要右转 → left>right

    FlyGym: action=[left_legs, right_legs]
    左转: left < right
    右转: left > right
    """
    speed = float(cmd_fwd)*0.3 + 0.8   # 基础速度 0.5-1.1
    turn  = float(cmd_turn)*0.5         # 转向幅度

    # 苹果在左(turn>0) → 左转 → 减小left
    left  = np.clip(speed - turn, 0.1, 1.5)
    right = np.clip(speed + turn, 0.1, 1.5)
    return left, right

# ============ 创建场景 ============
print(f"Fruit: {FRUIT_POS_MM} mm")
fly = flygym.Fly(
    enable_adhesion=True, init_pose='tripod',
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
    output_path="./20260407_fruit_chase_v2.mp4")
ctrl = HybridTurningController(
    fly=fly, arena=arena, cameras=cam, timestep=1e-4)
obs,_ = ctrl.reset()

# 验证方向
fly_pos0, heading0 = get_state(obs)
_, rel0, dist0 = get_visual_feat(fly_pos0, heading0, FRUIT_POS_MM)
print(f"Initial: heading={np.degrees(heading0):.1f}deg "
      f"rel={np.degrees(rel0):.1f}deg "
      f"dist={dist0:.1f}mm")
print(f"Fruit is {'LEFT' if rel0>0 else 'RIGHT'} of fly")

# ============ 主循环 ============
H,V = snn.reset_state(batch=1)
positions=[]; dist_hist=[]; turn_hist=[]; left_hist=[]; right_hist=[]
success=False

print(f"\nRunning {N_STEPS} steps...")

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

        fly_pos, heading = get_state(obs)
        positions.append(fly_pos.copy())
        vis_feat, rel_angle, dist = get_visual_feat(
            fly_pos, heading, FRUIT_POS_MM)
        dist_hist.append(dist)

        joints  = obs.get('joints',np.zeros((3,42)))[0]
        proprio = (joints[:12]/np.pi).astype(np.float32)
        vis_t   = torch.FloatTensor(vis_feat).unsqueeze(0)
        prop_t  = torch.FloatTensor(proprio).unsqueeze(0)

        with torch.no_grad():
            cmd,H,V,sp = snn(vis_t,prop_t,H,V)

        # SNN→腿部控制
        left, right = snn_to_legs(
            cmd[0,0].item(), cmd[0,1].item())

        left_hist.append(left)
        right_hist.append(right)
        turn_hist.append(left-right)  # 负=左转，正=右转

        try:
            obs,_,_,_,_=ctrl.step(
                np.array([left,right],dtype=np.float32))
        except Exception as e:
            print(f"Error: {e}"); break

        if i%RENDER_EVERY==0: ctrl.render()
        viewer.sync()

        if i%3000==0:
            print(f"  Step {i:5d} | "
                  f"dist={dist:.2f}mm | "
                  f"heading={np.degrees(heading):+.0f}deg | "
                  f"rel={np.degrees(rel_angle):+.0f}deg | "
                  f"L={left:.2f} R={right:.2f} | "
                  f"{'LEFT' if left<right else 'RIGHT'}")

        if dist<2.0 and not success:
            success=True
            print(f"\n  *** REACHED APPLE at step {i}! ***\n")

ctrl.close()
best_dist=min(dist_hist)
print(f"\nBest: {best_dist:.3f}mm | Success: {success}")

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
axes[0,0].set_title('Fly Trajectory\n[left,right] leg control',
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

axes[1,0].plot(left_hist,color='blue',linewidth=1,
               alpha=0.7,label='Left legs')
axes[1,0].plot(right_hist,color='red',linewidth=1,
               alpha=0.7,label='Right legs')
axes[1,0].axhline(y=0.8,color='gray',linestyle='--',
                   linewidth=1,label='Baseline')
axes[1,0].set_title('Leg Speed Control',fontsize=12)
axes[1,0].set_xlabel('Step')
axes[1,0].set_ylabel('Speed')
axes[1,0].legend(); axes[1,0].grid(True,alpha=0.3)

axes[1,1].plot(turn_hist,color='green',linewidth=1,alpha=0.7)
axes[1,1].axhline(y=0,color='black',linewidth=0.8)
axes[1,1].fill_between(range(len(turn_hist)),
                        turn_hist,alpha=0.2,color='green')
axes[1,1].set_title('Turn Signal (L-R)\nneg=left, pos=right',
                     fontsize=12)
axes[1,1].set_xlabel('Step')
axes[1,1].grid(True,alpha=0.3)

plt.suptitle(
    f'FlyGM-SNN: LC({len(lc_idx)})->DN({len(dn_idx)})->Motor({len(motor_idx)})\n'
    f'Fruit={FRUIT_POS_MM}mm | Best={best_dist:.2f}mm | '
    f'Success={success}',
    fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig('20260407_fruit_chase_v2_result.png',dpi=150)
print("Saved: 20260407_fruit_chase_v2_result.png")
print("Play: vlc ./20260407_fruit_chase_v2.mp4")
