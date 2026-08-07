"""
20260416 - 正确架构观看脚本
复眼视觉 → LC → Central → DN(左/右) → VNC → 腿
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

# ============ 模型定义（与训练完全一致）============
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

class CorrectArchSNN(nn.Module):
    def __init__(self, W, lc_idx, central_idx,
                 dn_left_idx, dn_right_idx, dn_idx_all):
        super().__init__()
        self.lc_idx      =torch.LongTensor(lc_idx)
        self.central_idx =torch.LongTensor(central_idx)
        self.dn_left_idx =torch.LongTensor(dn_left_idx)
        self.dn_right_idx=torch.LongTensor(dn_right_idx)
        self.dn_idx_all  =torch.LongTensor(dn_idx_all)
        W_scaled=W/100.0
        self.register_buffer('W_lc_cent',
            torch.FloatTensor(
                W_scaled[np.ix_(central_idx,lc_idx)]))
        self.register_buffer('W_cent_dn',
            torch.FloatTensor(
                W_scaled[np.ix_(dn_idx_all,central_idx)]))
        self.lif_lc     =SurrogateLIF(len(lc_idx))
        self.lif_central=SurrogateLIF(len(central_idx))
        self.lif_dn     =SurrogateLIF(len(dn_idx_all))
        self.eye_to_lc=nn.Sequential(
            nn.Linear(5,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,len(lc_idx)), nn.Sigmoid())
        self.proprio_to_dn=nn.Sequential(
            nn.Linear(12,64), nn.ReLU(),
            nn.Linear(64,len(dn_idx_all)), nn.Tanh())
        self._bufs={k:[] for k in
                    ['lc','central','dn_left','dn_right']}
        self.win=100

    def forward(self, vis_feat, proprio, state=None):
        batch=vis_feat.shape[0]; dev=vis_feat.device
        if state is None: state=self.init_state(batch,dev)
        v_lc,v_central,v_dn=state
        I_lc=self.eye_to_lc(vis_feat)
        v_lc,sp_lc=self.lif_lc(v_lc,I_lc)
        I_central=torch.einsum(
            'ij,bj->bi',self.W_lc_cent,sp_lc)
        v_central,sp_central=self.lif_central(
            v_central,I_central)
        I_dn=torch.einsum(
            'ij,bj->bi',self.W_cent_dn,sp_central)
        I_prop=self.proprio_to_dn(proprio)*0.1
        v_dn,sp_dn=self.lif_dn(v_dn,I_dn+I_prop)
        n_left=len(self.dn_left_idx)
        n_right=len(self.dn_right_idx)
        sp_dn_left =sp_dn[:,:n_left]
        sp_dn_right=sp_dn[:,n_left:n_left+n_right]
        for key,sp in [('lc',sp_lc),('central',sp_central),
                        ('dn_left',sp_dn_left),
                        ('dn_right',sp_dn_right)]:
            self._bufs[key].append(sp.detach())
            if len(self._bufs[key])>self.win:
                self._bufs[key].pop(0)
        rates={}
        for k in self._bufs:
            if self._bufs[k]:
                rates[k]=torch.stack(
                    self._bufs[k]).mean(0)*1000.0
            else:
                rates[k]=torch.zeros(batch,1,device=dev)
        dn_left_rate =sp_dn_left.mean(dim=1,keepdim=True)
        dn_right_rate=sp_dn_right.mean(dim=1,keepdim=True)
        turn_signal  =dn_left_rate-dn_right_rate
        base_speed=0.8; turn_scale=2.0
        left_leg =torch.clamp(
            base_speed-turn_signal*turn_scale,0.1,1.5)
        right_leg=torch.clamp(
            base_speed+turn_signal*turn_scale,0.1,1.5)
        cmd=torch.cat([left_leg,right_leg],dim=1)
        return cmd,(v_lc,v_central,v_dn),rates,turn_signal

    def init_state(self,batch=1,device='cpu'):
        return (
            torch.zeros(batch,len(self.lc_idx),device=device),
            torch.zeros(batch,len(self.central_idx),device=device),
            torch.zeros(batch,len(self.dn_idx_all),device=device),
        )
    def reset(self,batch=1,device='cpu'):
        self._bufs={k:[] for k in
                    ['lc','central','dn_left','dn_right']}
        return self.init_state(batch,device)

# ============ 参数 ============
FRUIT_POS_MM   = np.array([20.0, 0.0])
N_STEPS        = 18000
RENDER_EVERY   = 50
EYE_EVERY      = 100
DARK_THRESHOLD = 0.2
MAX_DARK_DIFF  = 50.0

# ============ 加载模型 ============
print("Loading correct architecture model...")
W    = np.load('20260416_full_pathway_W_cleft.npy')
meta = pd.read_csv('20260416_full_pathway_meta_cleft.csv')
df_dn_lat = pd.read_csv('20260416_dn_with_laterality.csv')

lc_idx      = meta[meta['is_LC']==True].index.values
central_idx = meta[meta['is_central']==True].index.values

dn_left_ids  = df_dn_lat[df_dn_lat['laterality']=='left']['root_id'].tolist()
dn_right_ids = df_dn_lat[df_dn_lat['laterality']=='right']['root_id'].tolist()
dn_left_meta = meta[meta['root_id'].isin(dn_left_ids)]
dn_right_meta= meta[meta['root_id'].isin(dn_right_ids)]
dn_left_idx  = dn_left_meta.index.values
dn_right_idx = dn_right_meta.index.values
dn_idx_all   = np.concatenate([dn_left_idx,dn_right_idx])

snn=CorrectArchSNN(W,lc_idx,central_idx,
                   dn_left_idx,dn_right_idx,dn_idx_all)
snn.load_state_dict(
    torch.load('20260416_correct_arch_best.pt'))
snn.eval()
print(f"LC:{len(lc_idx)} Central:{len(central_idx)} "
      f"DN_L:{len(dn_left_idx)} DN_R:{len(dn_right_idx)}")

# ============ 视觉特征提取 ============
def extract_visual_features(vision, fly_pos, fruit_pos):
    left =vision[0,:,0]; right=vision[1,:,0]
    left_dark =(left <DARK_THRESHOLD).sum()
    right_dark=(right<DARK_THRESHOLD).sum()
    dark_diff =float(left_dark-right_dark)
    dark_norm =np.clip(dark_diff/MAX_DARK_DIFF,-1,1)
    dx=fruit_pos[0]-fly_pos[0]
    dy=fruit_pos[1]-fly_pos[1]
    dist=np.sqrt(dx**2+dy**2)
    return np.array([
        dark_norm,
        np.clip(1.0/(dist+1.0),0,1),
        np.clip(float(left_dark+right_dark)/100.0,0,1),
        float(np.sign(dark_diff)),
        np.clip(dist/20.0,0,1),
    ],dtype=np.float32), dark_diff, dist

def get_dist(fly_pos,fruit_pos):
    return np.sqrt(((fly_pos-fruit_pos)**2).sum())

# ============ 复眼帧生成 ============
def make_eye_frame(vision, step, dist, dark_diff,
                   left, right, id_map,
                   dn_left_hz, dn_right_hz):
    H_map,W_map=id_map.shape
    left_img =np.ones((H_map,W_map))*0.5
    right_img=np.ones((H_map,W_map))*0.5
    for row in range(H_map):
        for col in range(W_map):
            oid=id_map[row,col]
            if 0<=oid<721:
                left_img[row,col] =vision[0,oid,0]
                right_img[row,col]=vision[1,oid,0]

    fig=plt.figure(figsize=(12,4),facecolor='black')
    ax1=fig.add_axes([0.02,0.15,0.38,0.75])
    ax2=fig.add_axes([0.42,0.15,0.38,0.75])
    ax3=fig.add_axes([0.83,0.15,0.15,0.75])

    ax1.imshow(left_img, cmap='gray',vmin=0,vmax=1)
    ax1.set_title('Left Eye (721 ommatidia)',
                   color='white',fontsize=10)
    ax1.axis('off')

    ax2.imshow(right_img,cmap='gray',vmin=0,vmax=1)
    ax2.set_title('Right Eye (721 ommatidia)',
                   color='white',fontsize=10)
    ax2.axis('off')

    # DN放电率柱状图
    bars=ax3.bar(['DN_L','DN_R'],
                  [dn_left_hz,dn_right_hz],
                  color=['blue','red'],alpha=0.8)
    ax3.set_ylim(0,200)
    ax3.set_title('DN Hz',color='white',fontsize=9)
    ax3.tick_params(colors='white',labelsize=8)
    ax3.set_facecolor('black')
    for spine in ax3.spines.values():
        spine.set_color('white')

    direction=('LEFT'  if dark_diff>5
               else 'RIGHT' if dark_diff<-5
               else 'FRONT')
    leg_str=('TURN LEFT' if left<right
             else 'TURN RIGHT')
    fig.text(0.5,0.03,
        f"Step:{step:5d} | Dist:{dist:.1f}mm | "
        f"Apple:{direction} | {leg_str} | "
        f"L={left:.2f} R={right:.2f}",
        ha='center',color='yellow',
        fontsize=9,fontfamily='monospace')
    fig.text(0.5,0.97,
        'Compound Eye → LC(1420) → Central(2737) → '
        'DN_L(632)+DN_R(621) → VNC → Legs',
        ha='center',color='white',
        fontsize=9,fontweight='bold')

    fig.canvas.draw()
    buf=np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    return buf

# ============ 创建场景 ============
print(f"\nFruit: {FRUIT_POS_MM} mm")
fly=flygym.Fly(
    enable_adhesion=True, init_pose='tripod',
    enable_vision=True, vision_refresh_rate=500,
    contact_sensor_placements=[
        f"{lg}{sg}"
        for lg in ["LF","LM","LH","RF","RM","RH"]
        for sg in ["Tibia","Tarsus1","Tarsus2",
                   "Tarsus3","Tarsus4","Tarsus5"]
    ],
)
arena=flygym.arena.FlatTerrain()
fruit_body=arena.root_element.worldbody.add(
    'body',name='fruit',
    pos=[FRUIT_POS_MM[0],FRUIT_POS_MM[1],1.5])
fruit_body.add('geom',name='fruit_main',
    type='sphere',size=[1.4],
    rgba=[0.95,0.1,0.05,1.0],
    contype='0',conaffinity='0')
fruit_body.add('geom',name='fruit_stem',
    type='cylinder',size=[0.15,0.8],
    pos=[0,0,1.5],rgba=[0.25,0.12,0.04,1.0],
    contype='0',conaffinity='0')
fruit_body.add('geom',name='fruit_leaf',
    type='ellipsoid',size=[0.6,0.2,0.12],
    pos=[0.5,0,1.9],rgba=[0.1,0.75,0.1,1.0],
    contype='0',conaffinity='0')

cam=flygym.Camera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top_zoomout",
    play_speed=0.15,
    output_path="./20260416_correct_arch_chase.mp4")
ctrl=HybridTurningController(
    fly=fly,arena=arena,cameras=cam,timestep=1e-4)
obs,_=ctrl.reset()

id_map=fly.retina.ommatidia_id_map
print(f"Retina: {fly.retina.num_ommatidia_per_eye} ommatidia/eye")

eye_writer=imageio.get_writer(
    './20260416_correct_arch_eye.mp4',fps=20,quality=7)

# ============ 主循环 ============
state=snn.reset(batch=1)
positions=[]; dist_hist=[]
left_hist=[]; right_hist=[]
dn_left_hist=[]; dn_right_hist=[]
dd_hist=[]; success=False

print(f"\nRunning {N_STEPS} steps...")

with mujoco.viewer.launch_passive(
    ctrl.unwrapped.physics.model.ptr,
    ctrl.unwrapped.physics.data.ptr,
) as viewer:
    viewer.cam.azimuth  =45
    viewer.cam.elevation=-35
    viewer.cam.distance =35.0

    for i in range(N_STEPS):
        if not viewer.is_running(): break

        fly_pos=obs['fly'][0][:2]
        positions.append(fly_pos.copy())
        vision=obs['vision']

        vis_feat,dark_diff,dist=extract_visual_features(
            vision,fly_pos,FRUIT_POS_MM)
        dist_hist.append(dist)
        dd_hist.append(vis_feat[0])

        joints =obs.get('joints',np.zeros((3,42)))[0]
        proprio=(joints[:12]/np.pi).astype(np.float32)
        vis_t  =torch.FloatTensor(vis_feat).unsqueeze(0)
        prop_t =torch.FloatTensor(proprio).unsqueeze(0)

        with torch.no_grad():
            cmd,state,rates,turn=snn(vis_t,prop_t,state)

        left =float(cmd[0,0].detach())
        right=float(cmd[0,1].detach())
        left =np.clip(left, 0.1,1.5)
        right=np.clip(right,0.1,1.5)
        left_hist.append(left)
        right_hist.append(right)

        dl=rates['dn_left'].mean().item()
        dr=rates['dn_right'].mean().item()
        dn_left_hist.append(dl)
        dn_right_hist.append(dr)

        try:
            obs,_,_,_,_=ctrl.step(
                np.array([left,right],dtype=np.float32))
        except Exception as e:
            print(f"Error:{e}"); break

        # 翻倒检测：z坐标太低说明果蝇翻了
        fly_z = obs['fly'][0][2]
        if fly_z < 0.5:  # 果蝇高度低于0.5mm
            print(f"  *** Fly flipped at step {i}! Stopping. ***")
            break

        if i%RENDER_EVERY==0: ctrl.render()
        if i%EYE_EVERY==0:
            frame=make_eye_frame(
                vision,i,dist,dark_diff,
                left,right,id_map,dl,dr)
            eye_writer.append_data(frame)
        viewer.sync()

        if i%3000==0:
            print(f"  Step {i:5d} | "
                  f"dist={dist:.2f}mm | "
                  f"dd={dark_diff:+.0f} | "
                  f"DN_L={dl:.0f} DN_R={dr:.0f}Hz | "
                  f"L={left:.2f} R={right:.2f} | "
                  f"{'LEFT' if left<right else 'RIGHT'}")

        if dist<2.0 and not success:
            success=True
            print(f"\n  *** REACHED APPLE at step {i}! ***\n")

eye_writer.close()
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
axes[0,0].set_title(
    'Fly Trajectory\nLC→Central→DN(L/R)→VNC→Legs',
    fontsize=12)
axes[0,0].set_xlabel('X (mm)')
axes[0,0].set_ylabel('Y (mm)')
axes[0,0].legend(fontsize=8)
axes[0,0].axis('equal')
axes[0,0].grid(True,alpha=0.3)

axes[0,1].plot(dist_hist,color='purple',linewidth=1.5)
axes[0,1].axhline(y=2.0,color='red',linestyle='--',
                   linewidth=2,label='Success (2mm)')
axes[0,1].axhline(y=best_dist,color='green',
                   linestyle=':',linewidth=1.5,
                   label=f'Best={best_dist:.2f}mm')
axes[0,1].set_title('Distance to Apple',fontsize=12)
axes[0,1].set_xlabel('Step')
axes[0,1].set_ylabel('mm')
axes[0,1].legend()
axes[0,1].grid(True,alpha=0.3)

axes[1,0].plot(dn_left_hist, color='blue',
               linewidth=1,alpha=0.8,label='DN_left')
axes[1,0].plot(dn_right_hist,color='red',
               linewidth=1,alpha=0.8,label='DN_right')
axes[1,0].set_title(
    'DN Left vs Right Firing Rate\n'
    '(asymmetry drives turning)',fontsize=12)
axes[1,0].set_xlabel('Step')
axes[1,0].set_ylabel('Hz')
axes[1,0].legend()
axes[1,0].grid(True,alpha=0.3)

axes[1,1].plot(left_hist, color='blue',
               linewidth=1,alpha=0.7,label='Left leg')
axes[1,1].plot(right_hist,color='red',
               linewidth=1,alpha=0.7,label='Right leg')
axes[1,1].axhline(y=0.8,color='gray',linestyle='--',
                   linewidth=1,label='Baseline')
axes[1,1].set_title(
    'Leg Speed\n(left<right=turn left)',fontsize=12)
axes[1,1].set_xlabel('Step')
axes[1,1].legend()
axes[1,1].grid(True,alpha=0.3)

plt.suptitle(
    f'Correct Bio Architecture: '
    f'Eye(721) → LC({len(lc_idx)}) → '
    f'Central({len(central_idx)}) → '
    f'DN_L({len(dn_left_idx)})+DN_R({len(dn_right_idx)}) → VNC\n'
    f'FlyWire FAFB | cleft_score | '
    f'Best={best_dist:.2f}mm | Success={success}',
    fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig('20260416_correct_arch_result.png',dpi=150)
print("Saved: 20260416_correct_arch_result.png")
print()
print("Videos:")
print("  vlc ./20260416_correct_arch_chase.mp4")
print("  vlc ./20260416_correct_arch_eye.mp4")
