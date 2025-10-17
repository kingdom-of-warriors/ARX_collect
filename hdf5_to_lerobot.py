import h5py
import numpy as np
import cv2

# 打开文件
with h5py.File('/home/arx/ROS2_LIFT_Play/act/datasets/episode_0.hdf5', 'r') as f:
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
    
    # 读取第一帧的关节位置
    first_qpos = f['observations/qpos'][0]
    print(f"\nFirst frame qpos: {first_qpos}")
    
    # 读取第一帧的图像（需要解压缩）
    first_image_compressed = f['observations/images/head'][0]
    first_image = cv2.imdecode(first_image_compressed, 1)
    print(f"\nFirst frame image shape: {first_image}")
    
    print("\n提取头部相机视频...")
    output_video = '/home/arx/ROS2_LIFT_Play/act/datasets/episode_0_head.mp4'
    
    # 获取视频参数
    head_images = f['observations/images/head']
    total_frames = len(head_images)
    height, width = first_image.shape[:2]
    fps = 30
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 写入所有帧
    for i in range(total_frames):
        img = cv2.imdecode(head_images[i], 1)
        if img is not None:
            out.write(img)
        print(f"\r处理进度: {i+1}/{total_frames}", end='')
    
    out.release()
    print(f"\n视频已保存: {output_video}")