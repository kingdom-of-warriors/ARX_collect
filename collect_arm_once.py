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


def collect_detect(start_episode, voice_engine):
    print(f"\nPreparing to record episode {start_episode}")
    input("Press Enter to start recording...")
    
    for i in range(3, 0, -1):
        voice_process(voice_engine, str(i))
        time.sleep(1)
    
    voice_process(voice_engine, "Go!")


def collect_information(args, collector):
    timesteps, actions, actions_eef = [], [], []
    count = 0
    rate = Rate(args.frame_rate)
    gripper_idx = [6, 13]
    gripper_close_threshold = -1.0
    
    print("\nStarting data collection... Press 'e' to END this trajectory recording")
    
    WINDOW_NAME = "Dual Arm Collection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while rclpy.ok():
        obs_dict = collector.get_observation()
        if obs_dict is None:
            print("Synchronization frame, waiting...", end='\r')
            rate.sleep()
            continue
        
        action = deepcopy(obs_dict['qpos'])
        action_eef = deepcopy(obs_dict['eef'])

        # 对于采集的数据来说，夹爪打开时为0，闭合时为-3出头
        # for idx in gripper_idx:
        #     action[idx] = 0 if action[idx] > gripper_close_threshold else action[idx]
        #     action_eef[idx] = 0 if action_eef[idx] > gripper_close_threshold else action_eef[idx]
        
        # action_eef[6] = 0 if action_eef[6] > gripper_close_threshold else action_eef[6]
        # action_eef[13] = 0 if action_eef[13] > gripper_close_threshold else action_eef[13]
        
        timesteps.append(obs_dict)
        actions.append(action)
        actions_eef.append(action_eef)

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
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print(f"\n✓ Trajectory recording finished.")
            break
        count += 1
        if not rclpy.ok(): exit(-1)
        
        rate.sleep()
    
    print(f"\n✓ Collection stopped. Total frames: {len(timesteps)}")
    return timesteps, actions, actions_eef


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


def save_data(args, timesteps, actions, actions_eef, dataset_path):
    data_size = len(actions)
    if data_size == 0:
        print("\nNo data to save.")
        return

    print(f"\n{'='*60}\nSaving {data_size} frames to HDF5...\n{'='*60}")
    
    data_dict = {
        '/observations/qpos': [], '/observations/qvel': [], '/observations/effort': [],
        '/observations/eef': [], '/action': [], '/action_eef': []
    }
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    
    for ts, action, action_eef in zip(timesteps, actions, actions_eef):
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/observations/eef'].append(ts['eef'])
        data_dict['/action'].append(action)
        data_dict['/action_eef'].append(action_eef)
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])
    
    data_dict['/action_base'] = np.zeros((data_size, 6), dtype=np.float32)
    data_dict['/action_velocity'] = np.zeros((data_size, 4), dtype=np.float32)
    data_dict['/observations/robot_base'] = deepcopy(data_dict['/action_base'])
    data_dict['/observations/base_velocity'] = deepcopy(data_dict['/action_velocity'])
    
    t0 = time.time()
    padded_size = compress_and_pad_images(data_dict, args.camera_names)
    print(f"✓ Image compression complete in {time.time() - t0:.2f}s")
    
    t0 = time.time()
    create_and_write_hdf5(args, data_dict, dataset_path, data_size, padded_size)
    
    voice_process(voice_engine, "Saved")
    print(f"\n{'='*60}\n✓ Data saved successfully in {time.time() - t0:.1f}s")
    print(f"  Location: {dataset_path}.hdf5\n{'='*60}\n")


def main(args):
    print(f"\n{'='*60}\nARM DATA COLLECTOR - Single Trajectory Mode\n{'='*60}")
    rclpy.init()
    config = load_yaml(args.config) if args.config else {}
    
    collector = ArmDataCollector(args, config)
    spin_thread = threading.Thread(target=rclpy.spin, args=(collector,), daemon=True)
    spin_thread.start()
    print("Waiting for topics to be ready...")
    time.sleep(2)
    
    date_str = datetime.now().strftime("%Y%m%d")
    dataset_dir = Path(args.dataset_dir) / f"{args.task.replace(' ', '_')}_collector{args.user_id}_{date_str}"
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
    print(f"Starting episode {current_episode}\n")
    
    collect_detect(current_episode, voice_engine)
    
    timesteps, actions, actions_eef = collect_information(args, collector)
    
    if timesteps:
        print("\n--- Action Required ---")
        print("  [s] Save trajectory")
        print("  [d] Discard trajectory and exit")
        print(">>> Click the image window and press a key <<<")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                print(f"\033[33m[INFO] Saving episode {current_episode}...\033[0m")
                dataset_path = dataset_dir / f"episode_{current_episode}"
                try:
                    save_data(args, timesteps, actions, actions_eef, str(dataset_path))
                    print(f"\033[32m[INFO] Episode {current_episode} saved successfully.\033[0m")
                except Exception as e:
                    print(f"\033[31m[ERROR] Failed to save episode: {e}\033[0m")
                break
            
            elif key == ord('d'):
                print("\033[31m[INFO] Episode discarded. Not saved.\033[0m")
                break
    else:
        print("\nNo data collected for this episode.")
    
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
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)