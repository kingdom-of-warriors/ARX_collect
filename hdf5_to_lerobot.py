import h5py
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def decompress_image(compressed_data):
    try:
        img_array = np.frombuffer(compressed_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"解压缩图像失败: {e}")
        return None


def convert_episode_to_lerobot(hdf5_path, output_dir, episode_idx, chunk_idx, task_name):
    """
    将单个HDF5 episode转换为LeRobot格式，并放入指定的chunk中。
    """
    output_dir = Path(output_dir)
    chunk_name = f"chunk-{chunk_idx:03d}"
    data_dir = output_dir / "data" / chunk_name
    video_chunk_dir = output_dir / "videos" / chunk_name
    meta_dir = output_dir / "meta"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    video_chunk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        total_frames = len(f['action'])
        
        cameras = list(f['observations/images'].keys())
        camera_shapes = {}
        
        for camera in cameras:
            camera_video_dir_name = f"observation.image.{camera}"
            camera_video_dir = video_chunk_dir / camera_video_dir_name
            camera_video_dir.mkdir(parents=True, exist_ok=True)
            camera_data = f[f'observations/images/{camera}']
        
            first_img = decompress_image(camera_data[0])            
            if first_img is None:
                print(f"错误: 无法解压 {camera} 的第一帧图像，跳过此 episode。")
                return False, None

            height, width = first_img.shape[:2]
            camera_shapes[camera] = (height, width)
            
            video_filename = f"{episode_idx:06d}.mp4"
            video_path = camera_video_dir / video_filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for i in range(total_frames):
                img = decompress_image(camera_data[i])
                if img is not None:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black_frame)
            out.release()
        
        actions = f['action'][:]
        qpos = f['observations/qpos'][:]
        qvel = f['observations/qvel'][:]
        effort = f['observations/effort'][:]
        eef = f['observations/eef'][:] if 'observations/eef' in f else np.zeros((total_frames, 14))
        robot_base = f['observations/robot_base'][:] if 'observations/robot_base' in f else np.zeros((total_frames, 6))
        base_velocity = f['observations/base_velocity'][:] if 'observations/base_velocity' in f else np.zeros((total_frames, 4))
        action_base = f['action_base'][:] if 'action_base' in f else np.zeros((total_frames, 6))
        action_velocity = f['action_velocity'][:] if 'action_velocity' in f else np.zeros((total_frames, 4))
        
        episode_data = {
            'episode_index': np.full(total_frames, episode_idx, dtype=np.int64),
            'frame_index': np.arange(total_frames, dtype=np.int64),
            'timestamp': np.arange(total_frames, dtype=np.float32) / 30.0,
            'action': list(actions.astype(np.float32)),
            'observation.state': list(qpos.astype(np.float32)),
            'observation.velocity': list(qvel.astype(np.float32)),
            'observation.effort': list(effort.astype(np.float32)),
            'observation.eef': list(eef.astype(np.float32)),
            'observation.base_pose': list(robot_base.astype(np.float32)),
            'observation.base_velocity': list(base_velocity.astype(np.float32)),
            'action.base_pose': list(action_base.astype(np.float32)),
            'action.base_velocity': list(action_velocity.astype(np.float32)),
        }
        
        for camera in cameras:
            if camera in camera_shapes:
                video_rel_path = f"videos/{chunk_name}/observation.image.{camera}/{episode_idx:06d}.mp4"
                episode_data[f'observation.image.{camera}'] = np.array([video_rel_path] * total_frames)
                episode_data[f'observation.image.{camera}_frame_index'] = np.arange(total_frames, dtype=np.int64)
        
        df = pd.DataFrame(episode_data)
        parquet_filename = f"episode_{episode_idx:06d}.parquet"
        parquet_path = data_dir / parquet_filename
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        episode_meta_info = {"episode_index": episode_idx, "task_description": task_name}
        
        return True, episode_meta_info


def process_all_episodes(all_hdf5_files, output_dir, chunk_size):
    output_dir = Path(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧的元数据文件
    for meta_file in ["episodes.jsonl", "tasks.jsonl", "info.json", "dataset.json"]:
        if (meta_dir / meta_file).exists():
            (meta_dir / meta_file).unlink()

    print(f"\n{'='*80}")
    print(f"批量转换所有数据集到统一的LeRobot格式")
    print(f"{'='*80}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(all_hdf5_files)} 个episodes")
    print(f"每个Chunk大小: {chunk_size} episodes")
    
    success_count = 0
    total_frames = 0
    unique_tasks = set()
    all_episode_meta = []

    for global_episode_idx, hdf5_file in tqdm(enumerate(all_hdf5_files), total=len(all_hdf5_files), desc="Processing Episodes"):
        # 从父目录名解析task, e.g., "lift_object_collector0_20251017" -> "lift_object"
        task_prefix = hdf5_file.parent.name.split('_collector')[0]
        task_name = task_prefix.replace('_', ' ')
        unique_tasks.add(task_name)
            
        chunk_idx = global_episode_idx // chunk_size
        success, episode_meta = convert_episode_to_lerobot(hdf5_file, output_dir, global_episode_idx, chunk_idx, task_name)
        
        if success:
            success_count += 1
            all_episode_meta.append(episode_meta)
    
    print(f"\n{'='*80}")
    print("生成最终元数据...")
    
    # 写入 episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, 'w') as f:
        for meta in sorted(all_episode_meta, key=lambda x: x['episode_index']):
            json.dump(meta, f)
            f.write('\n')
    print(f"✓ episodes.jsonl 已保存，包含 {len(all_episode_meta)} 个episodes")

    # 写入 tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    sorted_tasks = sorted(list(unique_tasks))
    with open(tasks_path, 'w') as f:
        for idx, task in enumerate(sorted_tasks):
            json.dump({"task_index": idx, "task": task}, f)
            f.write('\n')
    print(f"✓ tasks.jsonl 已保存，包含 {len(sorted_tasks)} 个独立任务")

    # 统计总帧数
    data_root = output_dir / "data"
    if data_root.exists():
        for episode_file in data_root.glob("**/episode_*.parquet"):
            table = pq.read_table(episode_file)
            total_frames += len(table)

    # 写入 dataset.json
    dataset_meta = {
        'name': output_dir.name,
        'num_episodes': success_count,
        'total_frames': total_frames,
        'fps': 30,
        'robot': 'arx_dual_arm',
        'tasks': sorted_tasks
    }
    with open(meta_dir / "dataset.json", 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    print(f"✓ dataset.json 已保存")
    
    # 写入 info.json
    if sorted_tasks:
        info_meta = {'task': sorted_tasks[0], 'fps': 30}
        with open(meta_dir / "info.json", 'w') as f:
            json.dump(info_meta, f, indent=2)
        print(f"✓ info.json 已保存")

    print(f"\n{'='*80}")
    print(f"转换完成!")
    print(f"成功: {success_count}/{len(all_hdf5_files)}")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='将多个HDF5数据集子目录合并转换为一个LeRobot格式数据集')
    
    parser.add_argument('--input', type=str, default='./datasets', help='包含多个数据集子目录的根目录')
    parser.add_argument('--output', type=str, default='./datasets_lerobot', help='统一的LeRobot格式输出目录')
    parser.add_argument('--chunk-size', type=int, default=1000, help='每个chunk包含的episode数量')
    
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.is_dir():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    all_hdf5_files = sorted(list(input_dir.glob("**/episode_*.hdf5")))
    if not all_hdf5_files:
        print(f"错误: 在 {input_dir} 中没有找到任何 episode_*.hdf5 文件。")
        return
        
    process_all_episodes(all_hdf5_files, output_dir, args.chunk_size)


if __name__ == '__main__':
    main()