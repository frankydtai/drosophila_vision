"""
测试果蝇复眼视觉
可视化721个小眼看到的画面
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import flygym
import flygym.arena
from flygym.examples.locomotion import HybridTurningController

# 创建场景（带苹果）
fruit_pos = np.array([5.0, 0.0])

fly = flygym.Fly(
    enable_adhesion=True,
    init_pose='tripod',
    enable_vision=True,
    vision_refresh_rate=500,
    contact_sensor_placements=[
        f'{lg}{sg}'
        for lg in ['LF','LM','LH','RF','RM','RH']
        for sg in ['Tibia','Tarsus1','Tarsus2',
                   'Tarsus3','Tarsus4','Tarsus5']
    ],
)
arena = flygym.arena.FlatTerrain()
fruit_body = arena.root_element.worldbody.add(
    'body', name='fruit',
    pos=[fruit_pos[0], fruit_pos[1], 1.5])
fruit_body.add('geom', name='fruit_main',
    type='sphere', size=[1.4],
    rgba=[0.95,0.1,0.05,1.0],
    contype='0', conaffinity='0')

ctrl = HybridTurningController(
    fly=fly, arena=arena, timestep=1e-4)
obs,_ = ctrl.reset()

# 打印视觉信息
vision = obs['vision']
print(f'视觉shape: {vision.shape}')
print(f'左眼[:10]: {vision[0,:10,0]}')
print(f'右眼[:10]: {vision[1,:10,0]}')
print()

# 获取小眼的位置信息
# FlyGym里小眼的坐标
retina = fly.retina
print(f'Retina类型: {type(retina)}')
print(f'Retina属性: {[x for x in dir(retina) if not x.startswith("_")]}')

# 可视化左右眼的亮度
fig, axes = plt.subplots(1,3,figsize=(15,5))

# 左眼
axes[0].bar(range(721), vision[0,:,0], width=1,
            color='blue', alpha=0.6)
axes[0].set_title(f'Left Eye (721 ommatidia)\n'
                   f'mean={vision[0,:,0].mean():.3f}',
                   fontsize=12)
axes[0].set_xlabel('Ommatidium index')
axes[0].set_ylabel('Intensity')

# 右眼
axes[1].bar(range(721), vision[1,:,0], width=1,
            color='red', alpha=0.6)
axes[1].set_title(f'Right Eye (721 ommatidia)\n'
                   f'mean={vision[1,:,0].mean():.3f}',
                   fontsize=12)
axes[1].set_xlabel('Ommatidium index')

# 左右眼对比
axes[2].plot(vision[0,:,0], 'b-', alpha=0.6,
             linewidth=0.5, label='Left')
axes[2].plot(vision[1,:,0], 'r-', alpha=0.6,
             linewidth=0.5, label='Right')
axes[2].set_title('Left vs Right Eye', fontsize=12)
axes[2].set_xlabel('Ommatidium index')
axes[2].legend()

plt.suptitle('Drosophila Compound Eye Vision\n'
             f'Apple at {fruit_pos}mm',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('20260407_vision_test.png', dpi=150)
print("Saved: 20260407_vision_test.png")

# 走几步看视觉变化
print("\n走向苹果时视觉变化：")
for i in range(5):
    for _ in range(500):
        obs,_,_,_,_=ctrl.step(
            np.array([0.8,0.8],dtype=np.float32))
    v = obs['vision']
    fly_pos = obs['fly'][0][:2]
    dist = np.sqrt((fly_pos[0]-fruit_pos[0])**2+
                   (fly_pos[1]-fruit_pos[1])**2)
    print(f'  Step {(i+1)*500}: '
          f'dist={dist:.2f}mm | '
          f'left_mean={v[0,:,0].mean():.3f} | '
          f'right_mean={v[1,:,0].mean():.3f} | '
          f'left_max={v[0,:,0].max():.3f} | '
          f'right_max={v[1,:,0].max():.3f}')

ctrl.close()
