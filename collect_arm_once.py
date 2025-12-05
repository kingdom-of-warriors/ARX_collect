# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import time
import h5py
import argparse
import rclpy
import cv2
import yaml
import threading
import pyttsx3
import numpy as np
from copy import deepcopy
from collections import deque

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from arx5_arm_msg.msg import RobotStatus

np.set_printoptions(linewidth=200)

voice_engine = pyttsx3.init()
voice_engine.setProperty('voice', 'en')
voice_engine.setProperty('rate', 120)
voice_lock = threading.Lock()


def voice_process(voice_engine, line):
    with voice_lock:
        voice_engine.say(line)
        voice_engine.runAndWait()
        print(line)


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


class ArmDataCollector(Node):
    def __init__(self, args, config):
        super().__init__('arm_data_collector')
        self.args = args
        self.config = config
        
        max_queue_size = args.frame_rate * 2 
        self.master_left_deque = deque(maxlen=max_queue_size)
        self.master_right_deque = deque(maxlen=max_queue_size)
        self.img_head_deque = deque(maxlen=max_queue_size)
        self.img_left_deque = deque(maxlen=max_queue_size)
        self.img_right_deque = deque(maxlen=max_queue_size)
                             
        self.last_update_time = self.get_clock().now()
        
        self._create_subscriptions()
        self.get_logger().info("ArmDataCollector initialized")
    
    def _create_subscriptions(self):
        self.create_subscription(RobotStatus, '/arm_master_l_status', self.master_left_callback, 10)
        self.create_subscription(RobotStatus, '/arm_master_r_status', self.master_right_callback, 10)
        self.create_subscription(CompressedImage, '/camera/camera_h/color/image_rect_raw/compressed', self.img_head_callback, 10)
        self.create_subscription(CompressedImage, '/camera/camera_l/color/image_rect_raw/compressed', self.img_left_callback, 10)
        self.create_subscription(CompressedImage, '/camera/camera_r/color/image_rect_raw/compressed', self.img_right_callback, 10)
        self.get_logger().info("All subscriptions created")
    
    def master_left_callback(self, msg):
        self.master_left_deque.append(msg)
    
    def master_right_callback(self, msg):
        self.master_right_deque.append(msg)
    
    def img_head_callback(self, msg):
        img = self._decompress_image(msg)
        if img is not None: self.img_head_deque.append(img)
    
    def img_left_callback(self, msg):
        img = self._decompress_image(msg)
        if img is not None: self.img_left_deque.append(img)
    
    def img_right_callback(self, msg):
        img = self._decompress_image(msg)
        if img is not None: self.img_right_deque.append(img)
    
    def _decompress_image(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Failed to decompress image: {e}")
            return None
    
    def extract_arm_data(self, msg):
        qpos = np.array(msg.joint_pos[:7])
        qvel = np.array(msg.joint_vel[:7])
        effort = np.array(msg.joint_cur[:7])
        eef = np.concatenate([np.array(msg.end_pos[:6]), [msg.joint_pos[6]]])
        return qpos, qvel, effort, eef
    
    def get_observation(self):
        if not all([self.master_left_deque, self.master_right_deque, self.img_head_deque, self.img_left_deque, self.img_right_deque]):
            return None
        
        left_msg = self.master_left_deque[-1]
        right_msg = self.master_right_deque[-1]
        left_qpos, left_qvel, left_effort, left_eef = self.extract_arm_data(left_msg)
        right_qpos, right_qvel, right_effort, right_eef = self.extract_arm_data(right_msg)
        
        return {
            'qpos': np.concatenate([left_qpos, right_qpos]),
            'qvel': np.concatenate([left_qvel, right_qvel]),
            'effort': np.concatenate([left_effort, right_effort]),
            'eef': np.concatenate([left_eef, right_eef]),
            'images': {
                'head': self.img_head_deque[-1].copy(),
                'left_wrist': self.img_left_deque[-1].copy(),
                'right_wrist': self.img_right_deque[-1].copy(),
            }
        }


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.last_time = time.time()
    
    def sleep(self):
        elapsed = time.time() - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()


def detect(collector):
    '''右夹爪闭合返回true'''
    time.sleep(2)  # 等待夹爪动作稳定
    obs = collector.get_observation()
    right_gripper = obs['qpos'][13]
    
    return right_gripper > -1.0 # 右夹爪闭合返回true


def collect_detect(voice_engine):
    for i in range(2, 0, -1):
        voice_process(voice_engine, str(i))
        time.sleep(1)
    
    voice_process(voice_engine, "Go!")


def collect_information_new(args, ros_operator, voice_engine):
    timesteps = []
    count = 0
    rate = Rate(args.frame_rate)

    STATE_RECORDING = 0
    STATE_STOP_PENDING = 1  # 运动已停止，正在确认
    current_state = STATE_RECORDING

    # 速度阈值
    VELOCITY_STOP_THRESHOLD = 0.5  # 运动停止的阈值
    VELOCITY_RESTART_THRESHOLD = args.v_threshold # 恢复运动的阈值
    
    # 停止确认时间：必须保持静止 1.5 秒才算真正结束
    STOP_CONFIRM_DURATION_SEC = 1.5
    stop_pending_start_time = 0
    
    print(f"录制将在机械臂 [静止 {STOP_CONFIRM_DURATION_SEC} 秒] 后 [自动停止]。")
    WINDOW_NAME = "Dual Arm Collection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    gripper_idx = [6, 13]
    gripper_close = -2.1

    while rclpy.ok():
        obs_dict = ros_operator.get_observation()

        # 同步帧检测
        if obs_dict is None:
            print("Synchronization frame", end='\r')
            rate.sleep()
            continue
        
        # 计算所有关节速度的绝对值之和，作为总运动量
        current_qvel = obs_dict['qvel']
        total_velocity = np.sum(np.abs(current_qvel))
        if current_state == STATE_RECORDING:
            if total_velocity < VELOCITY_STOP_THRESHOLD:
                print(f"\n[INFO] 运动停止 (V={total_velocity:.2f})。等待 {STOP_CONFIRM_DURATION_SEC} 秒确认...")
                current_state = STATE_STOP_PENDING
                stop_pending_start_time = time.time() # 启动计时器
            
        elif current_state == STATE_STOP_PENDING:
            if total_velocity > VELOCITY_RESTART_THRESHOLD:
                print("[INFO] 只是一个暂停，恢复录制。")
                current_state = STATE_RECORDING
                
            elif (time.time() - stop_pending_start_time) > STOP_CONFIRM_DURATION_SEC:
                print(f"\n✓ 自动停止：已静止 {STOP_CONFIRM_DURATION_SEC} 秒。")
                voice_process(voice_engine, "Stopped, Stopped!") # 语音反馈
                break


        # 夹爪动作处理
        # for idx in gripper_idx:
        #     action[idx] = 0 if action[idx] > gripper_close else action[idx]
        # action_eef[6] = 0 if action_eef[6] > gripper_close else action_eef[6]
        # action_eef[13] = 0 if action_eef[13] > gripper_close else action_eef[13]

        # 总是收集数据，之后再清理
        timesteps.append(obs_dict)

        # 显示采集时的图片
        head_img = obs_dict['images']['head']
        left_img = obs_dict['images']['left_wrist']
        right_img = obs_dict['images']['right_wrist']
        h, w, _ = head_img.shape
        left_img_resized = cv2.resize(left_img, (w, h))
        right_img_resized = cv2.resize(right_img, (w, h))
        combined_img = np.hstack([left_img_resized, head_img, right_img_resized])
        text = f"Frames: {count} (Press 'e' to end)"
        cv2.putText(combined_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, combined_img)
        cv2.waitKey(1)

        count += 1
        if not rclpy.ok():
            exit(-1)

        rate.sleep()

    print(f"\nlen(timesteps): {len(timesteps)}")
    
    # 清理“待确认”期间的污染数据
    if current_state == STATE_STOP_PENDING and len(timesteps) > 0:
        frames_to_remove = int(STOP_CONFIRM_DURATION_SEC * args.frame_rate)
        print(f"清理：正在移除最后 {frames_to_remove} 帧静止数据...")
        timesteps = timesteps[:-frames_to_remove]

        print(f"✓ 清理完毕。最终帧数: {len(timesteps)}")

    return timesteps


def compress_and_pad_images(data_dict, camera_names, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    all_encoded_lengths = []
    
    for cam in camera_names:
        key = f'/observations/images/{cam}'
        encoded_list = []
        for img in data_dict[key]:
            success, enc = cv2.imencode('.jpg', img, encode_param)
            if not success: raise ValueError(f"Failed to encode image for camera {cam}")
            encoded_list.append(enc)
            all_encoded_lengths.append(len(enc))
        data_dict[key] = encoded_list
    
    if not all_encoded_lengths: return 0
        
    padded_size = max(all_encoded_lengths)
    
    for cam in camera_names:
        key = f'/observations/images/{cam}'
        padded = [np.pad(enc, (0, padded_size - len(enc)), constant_values=0) for enc in data_dict[key]]
        data_dict[key] = padded
    
    return padded_size


def create_and_write_hdf5(args, data_dict, dataset_path, data_size, padded_size):
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['task'] = str(args.task)
        root.attrs['frame_rate'] = args.frame_rate
        root.attrs['compressed'] = True
        
        obs = root.create_group('observations')
        images = obs.create_group('images')
        
        for cam_name in args.camera_names:
            images.create_dataset(cam_name, (data_size, padded_size), dtype='uint8', chunks=(1, padded_size), compression='gzip', compression_opts=4)
        
        obs_specs = {'qpos': 14, 'qvel': 14, 'effort': 14, 'eef': 14}
        for name, dim in obs_specs.items():
            obs.create_dataset(name, (data_size, dim), dtype='float64')

        obs.create_dataset('robot_base', (data_size, 6), dtype='float32')
        obs.create_dataset('base_velocity', (data_size, 4), dtype='float32')
        
        root.create_dataset('action', (data_size, 14), dtype='float64')
        root.create_dataset('action_eef', (data_size, 14), dtype='float64')
        root.create_dataset('action_base', (data_size, 6), dtype='float32')
        root.create_dataset('action_velocity', (data_size, 4), dtype='float32')

        for name, arr in data_dict.items():
            root[name][...] = arr


def save_data(args, timesteps, dataset_path):
    data_size = len(timesteps)
    print(f"\n{'='*60}\nSaving {data_size} frames to HDF5...\n{'='*60}")
    
    data_dict = {
        '/observations/qpos': [], '/observations/qvel': [], '/observations/effort': [],
        '/observations/eef': [], '/observations/robot_base': [], '/observations/base_velocity': [],
        '/action': [], '/action_eef': []
    }
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    
    for ts in timesteps:
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/observations/eef'].append(ts['eef'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])
    
    # 动作数据与观测数据相同
    data_dict['/action'] = deepcopy(data_dict['/observations/qpos'])
    data_dict['/action_eef'] = deepcopy(data_dict['/observations/eef'])

    # 底盘数据先记为 0
    data_dict['/action_base'] = np.zeros((data_size, 6), dtype=np.float32) 
    data_dict['/action_velocity'] = np.zeros((data_size, 4), dtype=np.float32)
    data_dict['/observations/robot_base'] = np.zeros((data_size, 6), dtype=np.float32) 
    data_dict['/observations/base_velocity'] = np.zeros((data_size, 4), dtype=np.float32)
    
    padded_size = compress_and_pad_images(data_dict, args.camera_names)
    create_and_write_hdf5(args, data_dict, dataset_path, data_size, padded_size)
    
    voice_process(voice_engine, "Saved")
    print(f"  Location: {dataset_path}.hdf5\n{'='*60}\n")


def main(args):
    print(f"\n{'='*60}\nARM DATA COLLECTOR - Single Trajectory Mode\n{'='*60}")
    rclpy.init()
    config = load_yaml(args.config) if args.config else {}
    
    collector = ArmDataCollector(args, config)
    spin_thread = threading.Thread(target=rclpy.spin, args=(collector,), daemon=True)
    spin_thread.start()
    print("Waiting for topics to be ready...")
    time.sleep(1)
    
    date_str = datetime.now().strftime("%Y%m%d")
    dataset_dir = Path(args.dataset_dir) / f"{args.task.replace(' ', '_')}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {dataset_dir}\n")
    
    max_episode = -1
    if dataset_dir.exists():
        for filename in os.listdir(dataset_dir):
            if filename.startswith('episode_') and filename.endswith('.hdf5'):
                try:
                    episode_num = int(filename.split('_')[1].split('.')[0])
                    max_episode = max(max_episode, episode_num)
                except (ValueError, IndexError):
                    continue
    
    current_episode = max_episode + 1
    print(f"\nPreparing to record episode {current_episode}")
    print("Close the right gripper to continue:\n")
    voice_process(voice_engine, "Continue or not?")
    while not detect(collector):
        print("检测未通过，等待右夹爪闭合...", end='\r') # end='\r' 让打印在同一行刷新
        time.sleep(0.2)
    collect_detect(voice_engine)
    
    timesteps = collect_information_new(args, collector, voice_engine)

    print("  Right gripper CLOSE = Save")
    print("  Right gripper OPEN  = Delete")
    voice_process(voice_engine, "Delete or save?")
    flag_save_or_not = detect(collector)
        
    if flag_save_or_not:
        dataset_path = dataset_dir / f"episode_{current_episode}"
        save_data(args, timesteps, str(dataset_path))
    
    else:
        voice_process(voice_engine, "Deleted!")
        print("\033[31m[INFO] Episode discarded.\033[0m")
    
    time.sleep(0.1)
    
    print("\nShutting down...")
    cv2.destroyAllWindows()
    collector.destroy_node()
    rclpy.shutdown()
    if spin_thread.is_alive():
        spin_thread.join()
    print("✓ Shutdown complete")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect demonstration data from dual-arm robot')
    
    parser.add_argument('--dataset_dir', type=str, default=str(Path(__file__).parent / 'datasets'), help='Dataset directory')
    parser.add_argument('--user_id', type=int, default=0, help='Collector user ID')
    parser.add_argument('--frame_rate', type=int, default=30, help='Data collection frame rate (Hz)')
    parser.add_argument('--config', type=str, default=None, help='Config file path (optional)')
    parser.add_argument('--camera_names', nargs='+', type=str, choices=['head', 'left_wrist', 'right_wrist'], default=['head', 'left_wrist', 'right_wrist'], help='Camera names to use')
    parser.add_argument('--key_collect', action='store_true', help='Use keyboard trigger instead of automatic detection')
    parser.add_argument('--task', type=str, default='dual_arm_manipulation', help='Task name')
    parser.add_argument('--v_threshold', type=float, default=1.0, help='velocity threshold for collect')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)