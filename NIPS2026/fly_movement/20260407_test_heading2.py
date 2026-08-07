import numpy as np
import flygym
import flygym.arena
from flygym.examples.locomotion import HybridTurningController

fly = flygym.Fly(enable_adhesion=True, init_pose='tripod',
    contact_sensor_placements=[
        f'{lg}{sg}'
        for lg in ['LF','LM','LH','RF','RM','RH']
        for sg in ['Tibia','Tarsus1','Tarsus2',
                   'Tarsus3','Tarsus4','Tarsus5']
    ])
arena = flygym.arena.FlatTerrain()
ctrl = HybridTurningController(fly=fly,arena=arena,timestep=1e-4)
obs,_ = ctrl.reset()

fruit_pos = np.array([5.0, 0.5])

# 方法：用最近N步的位移方向作为朝向
# 比单步更稳定
pos_history = []
heading = 0.0
N_smooth = 50  # 用最近50步的位移

print('测试平滑朝向推算（N=50步平均）')
print(f'初始朝向: {np.degrees(heading):.1f}度')

for i in range(10):
    for _ in range(100):
        obs,_,_,_,_=ctrl.step(
            np.array([0.8,0.3],dtype=np.float32))
        pos_history.append(obs['fly'][0][:2].copy())
        if len(pos_history) > N_smooth:
            pos_history.pop(0)
        # 用最近N步的总位移方向
        if len(pos_history) >= 2:
            dx = pos_history[-1][0] - pos_history[0][0]
            dy = pos_history[-1][1] - pos_history[0][1]
            dist_moved = np.sqrt(dx**2+dy**2)
            if dist_moved > 1e-5:
                heading = np.arctan2(dy, dx)

    fly_pos = obs['fly'][0][:2]
    dx = fruit_pos[0]-fly_pos[0]
    dy = fruit_pos[1]-fly_pos[1]
    dist = np.sqrt(dx**2+dy**2)
    angle = np.arctan2(dy,dx)
    rel = np.arctan2(np.sin(angle-heading),
                     np.cos(angle-heading))
    print(f'  {(i+1)*100}步: '
          f'pos=({fly_pos[0]:.3f},{fly_pos[1]:.3f}) '
          f'heading={np.degrees(heading):+.1f}deg '
          f'rel={np.degrees(rel):+.1f}deg '
          f'dist={dist:.3f}mm')

ctrl.close()
print()
print('期望：右转时heading应该逐渐增大')
print('期望：rel_angle应该在0附近（对准苹果）')
