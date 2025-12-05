import os
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import yaml
import h5py
import argparse
import time
import threading
import rclpy
import numpy as np

from utils.ros_operator import RosOperator
from utils.setup_loader import setup_loader

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_hdf5(dataset_path):
    dataset_path = Path(dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as root:
        qposes = root['/observations/qpos'][()]
    return qposes


def init_robot(ros_operator):
    closed = 4
    left_qpos = [0, 0, 0, 0, 0, 0, closed]
    right_qpos = [0, 0, 0, 0, 0, 0, closed]
    ros_operator.follow_arm_publish_continuous(left_qpos, right_qpos)
    print("机械臂已初始化到初始位置")


def is_position_reached(current_qpos, target_qpos, threshold=0.05):
    diff = np.abs(current_qpos - target_qpos)
    return np.all(diff < threshold)


def measure_action_time(ros_operator, target_qpos, threshold=0.05, timeout=10.0):
    gripper_idx = [6, 13]
    
    target_qpos_adjusted = target_qpos.copy()
    left_action = target_qpos_adjusted[:gripper_idx[0] + 1]
    right_action = target_qpos_adjusted[gripper_idx[0] + 1:gripper_idx[1] + 1]
    
    start_time = time.time()
    ros_operator.follow_arm_publish(left_action, right_action)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"警告: 超时 ({timeout}s)，未到达目标位置")
            return None
        
        obs = ros_operator.get_observation()
        if not obs:
            time.sleep(0.001)
            continue
        
        current_qpos = obs['qpos']
        if is_position_reached(current_qpos, target_qpos, threshold):
            action_time = time.time() - start_time
            return action_time
        
        time.sleep(0.001)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--episode_path', type=str,
                        default='/home/haoming/ARX/ARX_collect/datasets/episode_1.hdf5',
                        help='HDF5 file path')
    parser.add_argument('--data', type=str,
                        default='/home/haoming/ARX/ARX_collect/config.yaml',
                        help='config file')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='number of positions to sample')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='position difference threshold')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='timeout for each action (seconds)')
    parser.add_argument('--use_base', action='store_true', default=False)
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance')
    parser.add_argument('--use_depth_image', action='store_true')
    parser.add_argument('--frame_rate', type=int, default=30)
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'])
    
    return parser.parse_args()


def main(args):
    setup_loader(ROOT)
    rclpy.init()
    
    config = load_yaml(args.data)
    ros_operator = RosOperator(args, config, in_collect=False)
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    
    init_robot(ros_operator)
    time.sleep(5)
    
    print("正在加载 HDF5 数据...")
    qposes = load_hdf5(args.episode_path)
    print(f"加载完成，共 {len(qposes)} 个位姿")
    
    num_samples = min(args.num_samples, len(qposes))
    indices = np.linspace(0, len(qposes) - 1, num_samples, dtype=int)
    
    print(f"\n开始测量，将测试 {num_samples} 个位姿...")
    print(f"位置阈值: {args.threshold}")
    print(f"超时时间: {args.timeout}s\n")
    
    action_times = []
    
    for i, idx in enumerate(indices):
        target_qpos = qposes[idx]
        print(f"[{i+1}/{num_samples}] 测试位姿 {idx}...")
        
        action_time = measure_action_time(
            ros_operator, 
            target_qpos, 
            threshold=args.threshold,
            timeout=args.timeout
        )
        
        if action_time is not None:
            action_times.append(action_time)
            print(f"  耗时: {action_time*1000:.2f} ms")
        else:
            print(f"  跳过（超时）")
        
        time.sleep(0.5)
    
    print("\n" + "="*50)
    print("测量完成！")
    print(f"成功测量: {len(action_times)}/{num_samples}")
    
    if len(action_times) > 0:
        action_times_array = np.array(action_times)
        print(f"\n平均耗时: {action_times_array.mean()*1000:.2f} ms")
        print(f"中位数耗时: {np.median(action_times_array)*1000:.2f} ms")
        print(f"最小耗时: {action_times_array.min()*1000:.2f} ms")
        print(f"最大耗时: {action_times_array.max()*1000:.2f} ms")
        print(f"标准差: {action_times_array.std()*1000:.2f} ms")
    
    ros_operator.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == '__main__':
    args = parse_args()
    main(args)