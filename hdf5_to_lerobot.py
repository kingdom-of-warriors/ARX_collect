import h5py
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse


def decompress_image(compressed_data):
    """
    解压缩图像数据
    """
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


def convert_episode_to_lerobot(hdf5_path, output_dir, episode_idx, chunk_idx):
    """
    将单个HDF5 episode转换为LeRobot格式，并放入指定的chunk中。
    """
    
    output_dir = Path(output_dir)
    
    # 根据chunk_idx动态生成目录
    chunk_name = f"chunk-{chunk_idx:03d}"
    data_dir = output_dir / "data" / chunk_name
    video_chunk_dir = output_dir / "videos" / chunk_name
    meta_dir = output_dir / "meta"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    video_chunk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"转换 Episode {episode_idx} -> Chunk {chunk_idx}")
    print(f"{'='*60}")
    print(f"输入: {hdf5_path}")
    print(f"输出: {output_dir}")
    
    with h5py.File(hdf5_path, 'r') as f:
        total_frames = len(f['action'])
        task_name = f.attrs.get('task', 'unknown_task')
        
        print(f"\n任务: {task_name}")
        print(f"总帧数: {total_frames}")
        
        # 提取视频
        print("\n[1/3] 提取视频...")
        cameras = list(f['observations/images'].keys())
        print(f"  发现相机: {cameras}")
        camera_shapes = {}
        
        for camera in cameras:
            print(f"  处理 {camera} 相机...")
            
            # 为每个相机创建单独的视频目录
            camera_video_dir_name = f"observation.image.{camera}"
            camera_video_dir = video_chunk_dir / camera_video_dir_name
            camera_video_dir.mkdir(parents=True, exist_ok=True)

            camera_data = f[f'observations/images/{camera}']
            
            # 解压第一帧获取尺寸
            first_img = decompress_image(camera_data[0])
            if first_img is None:
                print(f"    ⚠ 无法读取 {camera} 的第一帧，跳过")
                continue
            
            height, width = first_img.shape[:2]
            camera_shapes[camera] = (height, width)
            
            # 创建视频文件，文件名直接是episode编号
            video_filename = f"{episode_idx:06d}.mp4"
            video_path = camera_video_dir / video_filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"    ✗ 无法创建视频: {video_path}")
                continue
            
            # 写入所有帧
            failed_count = 0
            for i in range(total_frames):
                img = decompress_image(camera_data[i])
                if img is not None:
                    # 转回BGR for VideoWriter
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                else:
                    failed_count += 1
                    # 写入黑帧
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black_frame)
            
            out.release()
            
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ {camera}: {file_size_mb:.2f} MB (失败帧: {failed_count}) -> {video_path.relative_to(output_dir)}")
        
        # 提取数据
        print("\n[2/3] 提取数据...")
        
        # 读取所有数据
        actions = f['action'][:]
        qpos = f['observations/qpos'][:]
        qvel = f['observations/qvel'][:]
        effort = f['observations/effort'][:]
        
        if 'observations/eef' in f:
            eef = f['observations/eef'][:]
        else:
            eef = np.zeros((total_frames, 14))
        
        if 'observations/robot_base' in f:
            robot_base = f['observations/robot_base'][:]
            base_velocity = f['observations/base_velocity'][:]
            action_base = f['action_base'][:]
            action_velocity = f['action_velocity'][:]
        else:
            robot_base = np.zeros((total_frames, 6))
            base_velocity = np.zeros((total_frames, 4))
            action_base = np.zeros((total_frames, 6))
            action_velocity = np.zeros((total_frames, 4))
        
        # 构建LeRobot格式的数据字典
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
        
        # 添加视频帧索引
        for camera in cameras:
            if camera in camera_shapes:
                video_rel_path = f"videos/{chunk_name}/observation.image.{camera}/{episode_idx:06d}.mp4"
                episode_data[f'observation.image.{camera}'] = np.array([video_rel_path] * total_frames)
                episode_data[f'observation.image.{camera}_frame_index'] = np.arange(total_frames, dtype=np.int64)
        
        # 保存为Parquet格式
        print("\n[3/3] 保存数据...")
        
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            df = pd.DataFrame(episode_data)
            parquet_filename = f"episode_{episode_idx:06d}.parquet"
            parquet_path = data_dir / parquet_filename
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path, compression='snappy')
            
            file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ 数据已保存: {file_size_mb:.2f} MB")
            
        except ImportError:
            print("  ⚠ 缺少 pandas/pyarrow 库，无法保存为Parquet。")
        
        # 创建元数据
        meta_info = {
            'task': task_name,
            'fps': 30,
            'cameras': {}
        }
        for camera, (height, width) in camera_shapes.items():
            meta_info['cameras'][camera] = {'width': width, 'height': height, 'fps': 30}
        
        meta_path = meta_dir / "info.json"
        with open(meta_path, 'w') as f_json:
            json.dump(meta_info, f_json, indent=2)
        
        print(f"\n  ✓ 元数据已更新")
        print(f"Episode {episode_idx} 转换完成!")
        
        return True


def convert_dataset_to_lerobot(input_dir, output_dir, chunk_size):
    """
    批量转换整个数据集
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    episode_files = sorted(input_dir.glob("episode_*.hdf5"))
    
    print(f"\n{'='*80}")
    print(f"批量转换数据集到LeRobot格式")
    print(f"{'='*80}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(episode_files)} 个episodes")
    print(f"每个Chunk大小: {chunk_size} episodes")
    
    success_count = 0
    for i, hdf5_file in tqdm(enumerate(episode_files)):
        try:
            chunk_idx = i // chunk_size
            success = convert_episode_to_lerobot(hdf5_file, output_dir, i, chunk_idx)
            if success:
                success_count += 1
        except Exception as e:
            print(f"✗ 转换失败 {hdf5_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建最终的数据集级别元数据
    print(f"\n{'='*80}")
    print("生成最终元数据...")
    
    total_frames = 0
    data_root = output_dir / "data"
    if data_root.exists():
        for chunk_dir in data_root.iterdir():
            if chunk_dir.is_dir() and chunk_dir.name.startswith('chunk-'):
                for episode_file in chunk_dir.glob("episode_*.parquet"):
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(episode_file)
                        total_frames += len(table)
                    except Exception:
                        pass

    dataset_meta = {
        'name': input_dir.name,
        'num_episodes': len(episode_files),
        'num_successful': success_count,
        'total_frames': total_frames,
        'fps': 30,
        'robot': 'arx_dual_arm',
    }
    
    meta_path = output_dir / "meta" / "dataset.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(dataset_meta, f, indent=2)
    
    print(f"✓ dataset.json 已保存")
    print(f"{'='*80}")
    print(f"转换完成!")
    print(f"成功: {success_count}/{len(episode_files)}")
    print(f"总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='转换HDF5数据集到LeRobot格式')
    
    parser.add_argument(
        '--input',
        type=str,
        default='/home/arx/ROS2_LIFT_Play/act/datasets',
        help='输入HDF5数据集目录 (包含 episode_*.hdf5 文件)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/arx/ROS2_LIFT_Play/act/datasets_lerobot',
        help='输出LeRobot格式目录'
    )
    parser.add_argument(
        '--episode',
        type=int,
        default=None,
        help='只转换指定的episode编号 (默认: 转换全部)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='每个chunk包含的episode数量'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if args.episode is not None:
        hdf5_file = input_dir / f"episode_{args.episode}.hdf5"
        if not hdf5_file.exists():
            print(f"错误: 文件不存在: {hdf5_file}")
            return
        chunk_idx = args.episode // args.chunk_size
        convert_episode_to_lerobot(hdf5_file, output_dir, args.episode, chunk_idx)
    else:
        convert_dataset_to_lerobot(input_dir, output_dir, args.chunk_size)


if __name__ == '__main__':
    main()