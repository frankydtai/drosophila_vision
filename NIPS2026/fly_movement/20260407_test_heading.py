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
prev_pos  = obs['fly'][0][:2].copy()
heading   = 0.0  # 初始朝向

def update_heading(prev_pos, curr_pos, prev_heading,
                   alpha=0.3):
    """从位置变化推算朝向，用低通滤波平滑"""
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist > 1e-6:  # 移动了才更新朝向
        new_heading = np.arctan2(dy, dx)
        # 低通滤波：避免朝向跳变
        dh = np.arctan2(np.sin(new_heading - prev_heading),
                        np.cos(new_heading - prev_heading))
        return prev_heading + alpha * dh
    return prev_heading

print('测试朝向推算：右转500步')
print(f'初始朝向: {np.degrees(heading):.1f}度')

for i in range(5):
    for _ in range(100):
        obs,_,_,_,_=ctrl.step(
            np.array([0.8,0.3],dtype=np.float32))
        curr_pos = obs['fly'][0][:2]
        heading  = update_heading(prev_pos, curr_pos, heading)
        prev_pos = curr_pos.copy()

    fly_pos = obs['fly'][0][:2]
    dx = fruit_pos[0]-fly_pos[0]
    dy = fruit_pos[1]-fly_pos[1]
    dist = np.sqrt(dx**2+dy**2)
    angle = np.arctan2(dy,dx)
    rel = np.arctan2(np.sin(angle-heading),
                     np.cos(angle-heading))
    print(f'  {(i+1)*100}步: '
          f'heading={np.degrees(heading):+.1f}deg '
          f'rel={np.degrees(rel):+.1f}deg '
          f'dist={dist:.3f}mm')

ctrl.close()
print()
print('如果heading在增大（正值）= 果蝇在右转')
print('如果rel_angle在减小 = 果蝇在朝苹果转')
