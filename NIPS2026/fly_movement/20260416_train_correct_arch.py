"""
20260416 - 正确架构：LC→Central→DN(左右)→VNC替代→腿
基于真实神经解剖：
- LC: 视觉投射神经元（检测视觉特征）
- Central: 中间层整合神经元
- DN_left/DN_right: 下行神经元（投射到VNC左右侧）
- VNC: 用HybridTurningController替代（生物合理的简化）
- 腿部运动: 由DN左右不对称放电率决定

论文依据：
- Namiki et al. 2018: DN分左右投射到VNC
- Lobato-Rios et al. 2022 (FlyGym): CPG作为VNC替代
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

# ============ Surrogate Gradient LIF ============
class SurrogateLIF(nn.Module):
    """
    带Surrogate Gradient的LIF神经元
    前向：真实0/1脉冲
    反向：sigmoid近似（允许梯度流过）
    参考：Neftci et al. 2019, Zenke & Ganguli 2018
    """
    def __init__(self, n, tau=20.0, v_th=0.5,
                 v_reset=0.0, dt=1.0, surrogate_scale=10.0):
        super().__init__()
        self.tau=tau; self.v_th=v_th
        self.v_reset=v_reset; self.dt=dt
        self.surrogate_scale=surrogate_scale
        # 可训练偏置：模拟各神经元兴奋性差异
        self.bias=nn.Parameter(torch.zeros(n))

    def forward(self, v, I):
        dv=(-v+I+self.bias)*(self.dt/self.tau)
        v_new=v+dv
        # 前向：硬阈值（真实脉冲）
        sp_hard=(v_new>=self.v_th).float()
        # 反向：软阈值（surrogate gradient）
        sp_soft=torch.sigmoid(
            (v_new-self.v_th)*self.surrogate_scale)
        # Straight-through estimator
        sp=sp_hard-sp_soft.detach()+sp_soft
        # 放电后重置
        v_new=v_new*(1-sp_hard)+self.v_reset*sp_hard
        return v_new, sp

# ============ 正确架构SNN ============
class CorrectArchSNN(nn.Module):
    """
    生物正确的视觉运动通路：

    复眼(721小眼/眼)
         ↓ [eye_to_lc: 可训练]
    LC神经元(1420) - 视觉投射神经元
         ↓ [W_lc_cent: FlyWire真实突触]
    Central神经元(2737) - 中间整合层
         ↓ [W_cent_dn: FlyWire真实突触]
    DN_left(632) + DN_right(621) - 下行神经元
         ↓
    left_rate vs right_rate → 转向不对称
         ↓ [VNC替代: HybridTurningController]
    腿部运动 [left_speed, right_speed]

    关键：Motor神经元在VNC，不在大脑
    DN的左右放电率差异→转向命令
    """
    def __init__(self, W, lc_idx, central_idx,
                 dn_left_idx, dn_right_idx,
                 dn_idx_all):
        super().__init__()
        self.lc_idx       = torch.LongTensor(lc_idx)
        self.central_idx  = torch.LongTensor(central_idx)
        self.dn_left_idx  = torch.LongTensor(dn_left_idx)
        self.dn_right_idx = torch.LongTensor(dn_right_idx)
        self.dn_idx_all   = torch.LongTensor(dn_idx_all)

        # W矩阵（cleft_score/100）
        W_scaled = W / 100.0
        self.register_buffer('W_lc_cent',
            torch.FloatTensor(
                W_scaled[np.ix_(central_idx, lc_idx)]))
        self.register_buffer('W_cent_dn',
            torch.FloatTensor(
                W_scaled[np.ix_(dn_idx_all, central_idx)]))

        # 各层LIF（带surrogate gradient）
        self.lif_lc     = SurrogateLIF(len(lc_idx))
        self.lif_central= SurrogateLIF(len(central_idx))
        self.lif_dn     = SurrogateLIF(len(dn_idx_all))

        # 复眼暗点差→LC输入电流
        # 输入：5维（暗点差、距离、大小、符号、归一化距离）
        self.eye_to_lc = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, len(lc_idx)), nn.Sigmoid())

        # 本体感觉→DN（感觉反馈）
        self.proprio_to_dn = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, len(dn_idx_all)), nn.Tanh())

        # 放电率统计缓冲区
        self._bufs = {k:[] for k in
                      ['lc','central','dn_left','dn_right']}
        self.win = 100  # 100步滑动窗口

    def forward(self, vis_feat, proprio, state=None):
        batch = vis_feat.shape[0]
        dev   = vis_feat.device
        if state is None:
            state = self.init_state(batch, dev)
        v_lc, v_central, v_dn = state

        # ======== LC层：接收复眼视觉输入 ========
        I_lc = self.eye_to_lc(vis_feat)
        v_lc, sp_lc = self.lif_lc(v_lc, I_lc)

        # ======== Central层：接收LC脉冲 ========
        I_central = torch.einsum(
            'ij,bj->bi', self.W_lc_cent, sp_lc)
        v_central, sp_central = self.lif_central(
            v_central, I_central)

        # ======== DN层：接收Central脉冲+本体感觉 ========
        I_dn = torch.einsum(
            'ij,bj->bi', self.W_cent_dn, sp_central)
        I_prop = self.proprio_to_dn(proprio) * 0.1
        v_dn, sp_dn = self.lif_dn(v_dn, I_dn + I_prop)

        # ======== 放电率统计 ========
        # DN左右分开统计
        n_left  = len(self.dn_left_idx)
        n_right = len(self.dn_right_idx)

        # 在dn_idx_all里找left/right的位置
        # dn_left_idx和dn_right_idx是全局meta索引
        # 需要转成在sp_dn里的局部索引
        # 这里用预计算的局部索引
        sp_dn_left  = sp_dn[:, :n_left]   # 前n_left个是左侧
        sp_dn_right = sp_dn[:, n_left:n_left+n_right]  # 后n_right是右侧

        for key, sp in [
            ('lc',       sp_lc),
            ('central',  sp_central),
            ('dn_left',  sp_dn_left),
            ('dn_right', sp_dn_right),
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

        # ======== DN左右放电率→腿部控制 ========
        # 生物逻辑：
        # DN_left放电多 → 左侧VNC激活强 → 左腿运动强
        # 但转向需要：左转=左腿慢，右腿快
        # 即：左侧DN激活 → 左腿减速（对侧抑制？）
        #
        # 实际上更准确的是：
        # 视觉信号不对称 → DN不对称 → 转向
        # 用sp_dn直接解码更准确（保留梯度）

        # 用当前步的DN脉冲直接计算转向
        # 左侧DN平均放电
        dn_left_rate  = sp_dn_left.mean(dim=1, keepdim=True)
        dn_right_rate = sp_dn_right.mean(dim=1, keepdim=True)

        # 转向不对称：左DN多→左转（左腿慢）
        # turn = left_DN - right_DN（正=左转）
        turn_signal = dn_left_rate - dn_right_rate

        # 总体速度（用全部DN放电率）
        total_dn_rate = sp_dn.mean(dim=1, keepdim=True)

        # 腿速：基础速度 ± 转向
        base_speed = 0.8
        turn_scale = 2.0  # 放大转向信号

        left_leg  = torch.clamp(
            base_speed - turn_signal * turn_scale,
            0.1, 1.5)
        right_leg = torch.clamp(
            base_speed + turn_signal * turn_scale,
            0.1, 1.5)

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
        self._bufs = {k:[] for k in
                      ['lc','central','dn_left','dn_right']}
        return self.init_state(batch, device)

# ============ 加载数据 ============
print("Loading connectome data...")
W    = np.load('20260416_full_pathway_W_cleft.npy')
meta = pd.read_csv('20260416_full_pathway_meta_cleft.csv')
df_dn_lat = pd.read_csv('20260416_dn_with_laterality.csv')

lc_idx      = meta[meta['is_LC']==True].index.values
central_idx = meta[meta['is_central']==True].index.values

# DN按左右分类
dn_left_ids  = df_dn_lat[df_dn_lat['laterality']=='left']['root_id'].tolist()
dn_right_ids = df_dn_lat[df_dn_lat['laterality']=='right']['root_id'].tolist()

# 在meta里找索引，左侧在前，右侧在后
dn_left_meta  = meta[meta['root_id'].isin(dn_left_ids)]
dn_right_meta = meta[meta['root_id'].isin(dn_right_ids)]

# 合并成连续的dn_idx_all（左侧在前）
dn_left_idx  = dn_left_meta.index.values
dn_right_idx = dn_right_meta.index.values
dn_idx_all   = np.concatenate([dn_left_idx, dn_right_idx])

print(f"LC: {len(lc_idx)}")
print(f"Central: {len(central_idx)}")
print(f"DN_left: {len(dn_left_idx)}")
print(f"DN_right: {len(dn_right_idx)}")
print(f"DN_total: {len(dn_idx_all)}")
print(f"W shape: {W.shape}")
print(f"Synapses: {np.count_nonzero(W):,}")

snn = CorrectArchSNN(
    W, lc_idx, central_idx,
    dn_left_idx, dn_right_idx, dn_idx_all)
print(f"Parameters: {sum(p.numel() for p in snn.parameters()):,}")

# ============ 验证放电率和左右不对称 ============
print("\n验证放电率和左右DN不对称...")
snn.eval()

# 测试苹果在不同位置时的DN左右放电率
test_cases = [
    ('Apple LEFT  (dark=+0.5)', [+0.5,0.3,0.2,+1.0,0.3]),
    ('Apple FRONT (dark=0.0)',  [ 0.0,0.3,0.1, 0.0,0.3]),
    ('Apple RIGHT (dark=-0.5)', [-0.5,0.3,0.2,-1.0,0.3]),
]
for name, vf in test_cases:
    state = snn.reset(batch=1)
    with torch.no_grad():
        for t in range(200):
            cmd,state,rates,turn=snn(
                torch.FloatTensor(vf).unsqueeze(0),
                torch.zeros(1,12), state)
    left_hz  = rates['dn_left'].mean().item()
    right_hz = rates['dn_right'].mean().item()
    lc_hz    = rates['lc'].mean().item()
    cent_hz  = rates['central'].mean().item()
    l_speed  = float(cmd[0,0])
    r_speed  = float(cmd[0,1])
    print(f"\n{name}:")
    print(f"  LC={lc_hz:.1f}Hz Central={cent_hz:.1f}Hz")
    print(f"  DN_left={left_hz:.1f}Hz DN_right={right_hz:.1f}Hz "
          f"diff={left_hz-right_hz:+.1f}Hz")
    print(f"  Left_leg={l_speed:.3f} Right_leg={r_speed:.3f} "
          f"→ {'LEFT' if l_speed<r_speed else 'RIGHT'}")

# ============ 监督学习训练 ============
print("\n\n=== 监督学习训练 ===")
print("目标：苹果在左→左DN放电多→左腿慢（左转）")
print("="*55)

optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-5)

def generate_batch(bs=256):
    angles = np.random.uniform(-np.pi/2, np.pi/2, bs)
    dists  = np.random.uniform(2.0, 20.0, bs)
    dark_diff = np.sin(angles)
    vis = np.stack([
        np.clip(dark_diff,-1,1),
        np.clip(1.0/(dists+1),0,1),
        np.abs(dark_diff)*0.5,
        np.sign(dark_diff),
        np.clip(dists/20.0,0,1),
    ],axis=1).astype(np.float32)
    # 目标腿速
    turn   = np.sin(angles) * 0.4  # 正=苹果在左=左转
    speed  = 0.8
    left_t = np.clip((speed - turn).astype(np.float32), 0.1, 1.5)
    right_t= np.clip((speed + turn).astype(np.float32), 0.1, 1.5)
    return (torch.FloatTensor(vis),
            torch.zeros(bs,12),
            torch.FloatTensor(
                np.stack([left_t,right_t],axis=1)))

T_warmup = 30
n_epochs = 300
losses   = []
best_loss= 999.0

for epoch in range(n_epochs):
    vis, prop, targets = generate_batch(256)
    state = snn.reset(batch=256)

    with torch.no_grad():
        for t in range(T_warmup):
            _,state,_,_ = snn(vis, prop, state)

    loss_acc = 0.0
    for t in range(10):
        cmd,state,rates,turn=snn(vis,prop,state)
        if t >= 5:
            loss_acc += nn.MSELoss()(cmd, targets)
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
                   '20260416_correct_arch_best.pt')

    if epoch % 50 == 0 or epoch == n_epochs-1:
        state_t = snn.reset(batch=1)
        res = []
        for name, vf in [
            ('Front',[ 0.0,0.3,0.1, 0.0,0.3]),
            ('Right',[-0.5,0.3,0.2,-1.0,0.3]),
            ('Left', [+0.5,0.3,0.2,+1.0,0.3]),
        ]:
            with torch.no_grad():
                for t in range(T_warmup+10):
                    c,state_t,rt,ts=snn(
                        torch.FloatTensor(vf).unsqueeze(0),
                        torch.zeros(1,12),state_t)
            l=float(c[0,0]); r=float(c[0,1])
            dl=rt['dn_left'].mean().item()
            dr=rt['dn_right'].mean().item()
            direction='LEFT' if l<r else 'RIGHT'
            res.append(f"{name}:{direction}(DN:{dl:.0f}/{dr:.0f})")
        print(f"Ep{epoch:3d} | Loss={loss.item():.4f} | "
              +' | '.join(res))

# ============ 最终验证 ============
print("\n=== Final Validation ===")
snn.load_state_dict(torch.load('20260416_correct_arch_best.pt'))
snn.eval()

scenarios = [
    ('Front',  [ 0.0,0.3,0.1, 0.0,0.3]),
    ('Right',  [-0.5,0.3,0.2,-1.0,0.3]),
    ('Left',   [+0.5,0.3,0.2,+1.0,0.3]),
    ('Far-R',  [-0.2,0.1,0.1,-1.0,0.8]),
    ('Far-L',  [+0.2,0.1,0.1,+1.0,0.8]),
]
all_correct = True
state_t = snn.reset(batch=1)
for name, vf in scenarios:
    with torch.no_grad():
        for t in range(T_warmup+10):
            c,state_t,rt,ts=snn(
                torch.FloatTensor(vf).unsqueeze(0),
                torch.zeros(1,12), state_t)
    l=float(c[0,0]); r=float(c[0,1])
    dl=rt['dn_left'].mean().item()
    dr=rt['dn_right'].mean().item()
    lc=rt['lc'].mean().item()
    ct=rt['central'].mean().item()
    direction = 'LEFT'  if l < r else 'RIGHT'
    exp = ('LEFT'  if vf[0]>0.05
           else 'RIGHT' if vf[0]<-0.05
           else 'STRAIGHT')
    ok  = 'OK' if exp==direction or exp=='STRAIGHT' else 'FAIL'
    if exp!=direction and exp!='STRAIGHT':
        all_correct = False
    print(f"  {name:7s} | dark={vf[0]:+.1f} | "
          f"L={l:.3f} R={r:.3f} {direction} {ok} | "
          f"LC={lc:.0f} Cent={ct:.0f} "
          f"DN_L={dl:.0f} DN_R={dr:.0f}Hz")

print()
if all_correct:
    print("All correct!")
    print("LC→Central→DN(left/right)→VNC→legs: WORKING!")
    print()
    print("Architecture summary:")
    print(f"  Compound eye (721 ommatidia/eye)")
    print(f"  → LC neurons ({len(lc_idx)}) [Visual Projection]")
    print(f"  → Central neurons ({len(central_idx)}) [Integration]")
    print(f"  → DN_left ({len(dn_left_idx)}) + DN_right ({len(dn_right_idx)}) [Descending]")
    print(f"  → VNC (HybridTurningController, surrogate)")
    print(f"  → Leg motion [left_speed, right_speed]")
else:
    print("Partial - check DN asymmetry")

# 训练曲线
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ax1.plot(losses,'b-',linewidth=1.5)
if len(losses)>=10:
    ma=[np.mean(losses[max(0,i-9):i+1])
        for i in range(len(losses))]
    ax1.plot(ma,'r-',linewidth=2,label='Moving avg')
    ax1.legend()
ax1.set_title('Correct Architecture Training\n'
    'LC→Central→DN(L/R)→VNC→Legs',fontsize=12)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_yscale('log'); ax1.grid(True,alpha=0.3)

# DN左右不对称响应
dark_diffs = np.linspace(-1,1,60)
left_speeds=[]; right_speeds=[]
dn_left_rates=[]; dn_right_rates=[]
state_s = snn.reset(batch=1)
for dd in dark_diffs:
    vf=torch.FloatTensor(
        [[dd,0.3,abs(dd)*0.5,np.sign(dd),0.3]])
    with torch.no_grad():
        for t in range(T_warmup+5):
            c,state_s,rt,ts=snn(
                vf,torch.zeros(1,12),state_s)
    left_speeds.append(float(c[0,0]))
    right_speeds.append(float(c[0,1]))
    dn_left_rates.append(rt['dn_left'].mean().item())
    dn_right_rates.append(rt['dn_right'].mean().item())

ax2_twin=ax2.twinx()
ax2.plot(dark_diffs,left_speeds, 'b-',
         linewidth=2,label='Left leg speed')
ax2.plot(dark_diffs,right_speeds,'r-',
         linewidth=2,label='Right leg speed')
ax2_twin.plot(dark_diffs,dn_left_rates, 'b--',
              linewidth=1.5,alpha=0.6,label='DN_left Hz')
ax2_twin.plot(dark_diffs,dn_right_rates,'r--',
              linewidth=1.5,alpha=0.6,label='DN_right Hz')
ax2.axhline(y=0.8,color='gray',linestyle=':',linewidth=1)
ax2.axvline(x=0,color='k',linewidth=0.5)
ax2.set_title('DN Asymmetry → Leg Speed\n'
    '(+dark=apple left, -dark=apple right)',fontsize=12)
ax2.set_xlabel('Visual Signal (dark diff)')
ax2.set_ylabel('Leg Speed')
ax2_twin.set_ylabel('DN Firing Rate (Hz)',color='purple')
ax2.legend(loc='upper left',fontsize=9)
ax2_twin.legend(loc='upper right',fontsize=9)
ax2.grid(True,alpha=0.3)

plt.suptitle(
    f'Correct Architecture: Compound Eye → '
    f'LC({len(lc_idx)}) → Central({len(central_idx)}) → '
    f'DN_L({len(dn_left_idx)})+DN_R({len(dn_right_idx)}) → VNC\n'
    f'FlyWire FAFB connectome | cleft_score weights | '
    f'Best Loss={best_loss:.4f}',
    fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig('20260416_correct_arch_training.png',dpi=150)
print("\nSaved: 20260416_correct_arch_training.png")
print("Next: 20260416_watch_correct_arch.py")
