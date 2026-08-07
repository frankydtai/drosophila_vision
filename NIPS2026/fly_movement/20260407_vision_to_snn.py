"""
把复眼视觉(721个小眼)接入SNN
替换之前的手算几何特征
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

# ============ 视觉特征提取 ============
def extract_visual_features(vision):
    """
    从721个小眼的亮度提取5维特征
    vision shape: (2, 721, 2)
    
    模拟真实LC神经元的功能：
    - 左右眼差异 → 苹果在左还是右
    - 总亮度变化 → 苹果远近
    """
    left  = vision[0,:,0]  # 左眼721个小眼
    right = vision[1,:,0]  # 右眼721个小眼

    # 特征1：左右眼亮度差（负=苹果在左，正=苹果在右）
    # 暗的地方是苹果（红色在灰度里较暗）
    left_mean  = left.mean()
    right_mean = right.mean()
    lr_diff = float(right_mean - left_mean)  # 右眼暗=苹果在右

    # 特征2：左眼最暗区域（苹果位置）
    left_min  = float(left.min())
    right_min = float(right.min())

    # 特征3：整体亮度（距离信号）
    total_brightness = float((left_mean+right_mean)/2)

    # 特征4：左右差的符号
    lr_sign = float(np.sign(lr_diff))

    # 特征5：亮度变化幅度
    contrast = float(left.std() + right.std())

    return np.array([
        lr_diff,            # 左右差（最重要！）
        left_min-right_min, # 最暗点差异
        total_brightness,   # 总亮度
        lr_sign,            # 左右符号
        np.clip(contrast,0,1) # 对比度
    ], dtype=np.float32)

# 测试视觉特征
print("测试视觉特征提取...")
fly = flygym.Fly(
    enable_adhesion=True, init_pose='tripod',
    enable_vision=True, vision_refresh_rate=500,
    contact_sensor_placements=[
        f'{lg}{sg}'
        for lg in ['LF','LM','LH','RF','RM','RH']
        for sg in ['Tibia','Tarsus1','Tarsus2',
                   'Tarsus3','Tarsus4','Tarsus5']
    ],
)
arena = flygym.arena.FlatTerrain()

# 苹果在右边
fruit_body = arena.root_element.worldbody.add(
    'body', name='fruit',
    pos=[5.0, -2.0, 1.5])  # 右边
fruit_body.add('geom', name='fruit_main',
    type='sphere', size=[1.4],
    rgba=[0.95,0.1,0.05,1.0],
    contype='0', conaffinity='0')

ctrl = HybridTurningController(
    fly=fly, arena=arena, timestep=1e-4)
obs,_ = ctrl.reset()

vision = obs['vision']
feat = extract_visual_features(vision)
print(f"苹果在右边(y=-2):")
print(f"  特征: {feat}")
print(f"  lr_diff={feat[0]:.4f} "
      f"({'苹果在右' if feat[0]<0 else '苹果在左'})")
ctrl.close()

# 苹果在左边
arena2 = flygym.arena.FlatTerrain()
fruit_body2 = arena2.root_element.worldbody.add(
    'body', name='fruit',
    pos=[5.0, 2.0, 1.5])  # 左边
fruit_body2.add('geom', name='fruit_main',
    type='sphere', size=[1.4],
    rgba=[0.95,0.1,0.05,1.0],
    contype='0', conaffinity='0')

fly2 = flygym.Fly(
    enable_adhesion=True, init_pose='tripod',
    enable_vision=True, vision_refresh_rate=500,
    contact_sensor_placements=[
        f'{lg}{sg}'
        for lg in ['LF','LM','LH','RF','RM','RH']
        for sg in ['Tibia','Tarsus1','Tarsus2',
                   'Tarsus3','Tarsus4','Tarsus5']
    ],
)
ctrl2 = HybridTurningController(
    fly=fly2, arena=arena2, timestep=1e-4)
obs2,_ = ctrl2.reset()

vision2 = obs2['vision']
feat2 = extract_visual_features(vision2)
print(f"\n苹果在左边(y=+2):")
print(f"  特征: {feat2}")
print(f"  lr_diff={feat2[0]:.4f} "
      f"({'苹果在右' if feat2[0]<0 else '苹果在左'})")
ctrl2.close()

print()
if feat[0] * feat2[0] < 0:
    print("✅ 视觉特征可以区分左右！")
    print("可以用复眼视觉替换几何公式！")
else:
    print("❌ 视觉特征无法区分左右")
    print("需要调整特征提取方法")
