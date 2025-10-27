import h5py
import numpy as np
import cv2

# 打开文件
with h5py.File('/mnt/inspurfs/evla1_t/lijiarui/datasets/play_games_collector0_20251022/episode_0.hdf5', 'r') as f:
    # ===== 顶层属性 =====
    print("Attributes:")
    print(f"  sim: {f.attrs['sim']}")              # False (真实机器人)
    print(f"  task: {f.attrs['task']}")            # 'dual_arm_manipulation'
    
    # ===== 数据集结构 =====
    print("\nDatasets:")
    print(f"  action: {f['action'].shape}")       # (T, 14) 双臂动作
    print(f"  reward: {f['action_eef'].shape}")       # (T, 14) 末端动作
    
    print("\nObservations:")
    print(f"  qpos: {f['observations/qpos'].shape}")     # (T, 14) 关节位置
    print(f"  qvel: {f['observations/qvel'].shape}")     # (T, 14) 关节速度
    print(f"  effort: {f['observations/effort'].shape}") # (T, 14) 关节力矩
    print(f"  eef: {f['observations/eef'].shape}")       # (T, 14) 末端位姿
    
    print("\nImages:")
    print(f"  head: {f['observations/images/head'].shape}")           # (T, padded_size)
    print(f"  left_wrist: {f['observations/images/left_wrist'].shape}") # (T, padded_size)
    print(f"  right_wrist: {f['observations/images/right_wrist'].shape}") # (T, padded_size)
    
    first_qpos = f['observations/qpos'][0]
    print(f"\nFirst frame qpos: {first_qpos}")
    
    # ===== 计算平均值 =====
    # 计算所有帧的 action 平均值
    action_mean = np.mean(f['action'][:], axis=0)
    print(f"\nAction mean across all frames: {action_mean}")
    
    # 计算所有帧的 qpos 平均值
    qpos_mean = np.mean(f['observations/qpos'][:], axis=0)
    print(f"\nQpos mean across all frames: {qpos_mean}")
    
    
    # # 读取第一帧的图像（需要解压缩）
    # first_image_compressed = f['observations/images/head'][0]
    # first_image = cv2.imdecode(first_image_compressed, 1)
    # print(f"\nFirst frame image shape: {first_image}")