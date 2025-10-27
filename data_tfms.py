import h5py
import numpy as np
import os
from pathlib import Path
import shutil
from tqdm import tqdm

def fix_gripper_data(file_path, backup=True):
    """
    修复单个 HDF5 文件中的夹爪数据
    
    Args:
        file_path: HDF5 文件路径
        backup: 是否创建备份
    """
    if backup:
        backup_path = str(file_path) + '.backup'
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    with h5py.File(file_path, 'r+') as f:
        # 读取数据
        action = f['action'][:]
        action_eef = f['action_eef'][:]
        qpos = f['observations/qpos'][:]
        
        # 检查数据形状
        print(f"Processing {os.path.basename(file_path)}")
        print(f"  action shape: {action.shape}")
        print(f"  action_eef shape: {action_eef.shape}")
        print(f"  qpos shape: {qpos.shape}")
        
        # 显示修改前的值（第一帧作为示例）
        print(f"\nBefore fix (frame 0):")
        print(f"  action[0, 6]: {action[0, 6]}, action[0, 13]: {action[0, 13]}")
        print(f"  action_eef[0, 6]: {action_eef[0, 6]}, action_eef[0, 13]: {action_eef[0, 13]}")
        print(f"  qpos[0, 6]: {qpos[0, 6]}, qpos[0, 13]: {qpos[0, 13]}")
        
        # 修复数据：将 qpos 的夹爪值复制到 action 和 action_eef
        action[:, 6] = qpos[:, 6]
        action[:, 13] = qpos[:, 13]
        action_eef[:, 6] = qpos[:, 6]
        action_eef[:, 13] = qpos[:, 13]
        
        # 显示修改后的值
        print(f"\nAfter fix (frame 0):")
        print(f"  action[0, 6]: {action[0, 6]}, action[0, 13]: {action[0, 13]}")
        print(f"  action_eef[0, 6]: {action_eef[0, 6]}, action_eef[0, 13]: {action_eef[0, 13]}")
        
        # 写回文件
        del f['action']
        del f['action_eef']
        f.create_dataset('action', data=action)
        f.create_dataset('action_eef', data=action_eef)
        
        print(f"✓ Successfully fixed {os.path.basename(file_path)}\n")

def fix_all_episodes(dataset_dir, backup=True):
    """
    修复指定目录下所有 episode 文件
    
    Args:
        dataset_dir: 数据集目录路径
        backup: 是否创建备份
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_dir} does not exist!")
        return
    
    # 查找所有 episode 文件
    episode_files = sorted(dataset_path.glob('episode_*.hdf5'))
    
    if not episode_files:
        print(f"No episode files found in {dataset_dir}")
        return
    
    print(f"Found {len(episode_files)} episode files")
    print(f"Backup enabled: {backup}\n")
    
    # 处理每个文件
    for episode_file in tqdm(episode_files, desc="Processing episodes"):
        try:
            fix_gripper_data(episode_file, backup=backup)
        except Exception as e:
            print(f"Error processing {episode_file}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"All episodes processed!")
    print(f"Dataset directory: {dataset_dir}")
    if backup:
        print(f"Backups saved with .backup extension")

if __name__ == "__main__":
    # 设置数据集路径
    dataset_directory = "/home/arx/ARX_collect/datasets/play_games_collector0_20251022"
    
    # 执行修复（默认会创建备份）
    fix_all_episodes(dataset_directory, backup=True)
    
    # 如果不需要备份，可以使用：
    # fix_all_episodes(dataset_directory, backup=False)