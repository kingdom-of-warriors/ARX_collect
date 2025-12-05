# -- coding: UTF-8
import os
import sys
from functools import partial
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
import signal

import rclpy

import threading

import numpy as np

np.set_printoptions(linewidth=200)

from functools import partial

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None


def load_hdf5(dataset_path):
    dataset_path = Path.joinpath(ROOT, dataset_path)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at: {dataset_path}")

    try:
        with h5py.File(dataset_path, 'r') as root:
            qposes = root.get('/observations/qpos')

            # 确保所有所需的数据集都存在
            if any(item is None for item in [qposes]):
                missing_datasets = [name for name, item in zip(
                    ['/observations/qpos'],
                    [qposes]
                ) if item is None]

                raise ValueError(f"Missing datasets in HDF5 file: {', '.join(missing_datasets)}")

            return qposes[()]
    except Exception as e:
        raise RuntimeError(f"Error occurred while loading the HDF5 file: {e}")


def resample_sequence(data, src_hz, target_hz):
    """
    对时间序列做上下采样，使得在 target_hz 下播放时，总时长与 src_hz 下相同。
    data: np.ndarray [T, ...]
    src_hz: 原始采集频率（Hz）
    target_hz: 目标播放频率（Hz）
    """
    if np.isclose(src_hz, target_hz):
        return data  # 频率相同，直接返回

    T_src = data.shape[0]
    duration = T_src / src_hz
    T_tgt = int(round(duration * target_hz))
    if T_tgt <= 1 or T_src <= 1:
        return data

    # 原始时间轴与目标时间轴（单位：秒）
    t_src = np.linspace(0, duration, T_src)
    t_tgt = np.linspace(0, duration, T_tgt)

    # 逐点插值
    flat = data.reshape(T_src, -1)           # [T_src, D]
    flat_tgt = np.empty((T_tgt, flat.shape[1]), dtype=flat.dtype)

    for d in range(flat.shape[1]):
        flat_tgt[:, d] = np.interp(t_tgt, t_src, flat[:, d])

    return flat_tgt.reshape((T_tgt,) + data.shape[1:])


def robot_action(ros_operator, args, action):
    gripper_idx = [6, 13]
    for idx in gripper_idx:
        action[idx] += 3.4
        if action[idx] > 1.5:
            action[idx] = 6.0
        else:
            action[idx] = 0.0
    left_action = action[:gripper_idx[0] + 1]  # 取8维度
    right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]  # action[7:14]

    print(f'{left_action=}')
    ros_operator.follow_arm_publish(left_action, right_action)  # follow_arm_publish_continuous_thread


def init_robot(ros_operator):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    ros_operator.follow_arm_publish_continuous(init0, init0)
    ros_operator.robot_base_shutdown()


def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')
    # 底盘给零
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)


def main(args):
    setup_loader(ROOT)
    rclpy.init()
    config = load_yaml(args.data)
    ros_operator = RosOperator(args, config, in_collect=False)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    init_robot(ros_operator)
    signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))
    init_robot(ros_operator)
    qpoes = load_hdf5(args.episode_path)

    # 假设数据是以 src_hz 采集的
    src_hz = args.src_hz
    target_hz = args.target_hz
    replay_actions = qpoes

    # 是否需要采样来回放动作
    if args.sample:
        replay_actions = resample_sequence(replay_actions, src_hz, target_hz)

    import time
    time.sleep(3)
    rate = Rate(target_hz)
    for idx in range(len(replay_actions)):
        print(f'{replay_actions[idx]=}')
        robot_action(ros_operator, args, replay_actions[idx])
        rate.sleep()

    ros_operator.base_enable = False
    ros_operator.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--episode_path', type=str, help='episode_path',
                        default='/home/haoming/ARX/ARX_collect/datasets/infer_20251127_110021.hdf5')
    parser.add_argument('--data', type=str, default='/home/haoming/ARX/ARX_collect/config.yaml', help='config file')

    parser.add_argument('--use_base', action='store_true', default=False, help='use base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')

    parser.add_argument('--states_replay', default=False, action='store_true', help='use qpos replay')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')
    parser.add_argument('--src_hz', type=float, default=30.0,
                        help='原始采集频率（Hz），例如数据集是 30Hz 录的')
    parser.add_argument('--target_hz', type=float, default=30,
                        help='目标回放频率（Hz），例如 60.0 表示上采样为 60Hz 播放')
    parser.add_argument('--sample', action='store_true', help='是否需要上/下采样来回放动作')
    parser.add_argument('--frame_rate', default=30)
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
