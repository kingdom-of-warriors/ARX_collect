# -- coding: UTF-8
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
    
import cv2
from datetime import datetime
import argparse
import collections
import yaml
import rclpy
import torch
import threading
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import requests
import json

from rclpy.executors import MultiThreadedExecutor
from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader


obs_dict = collections.OrderedDict()

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def make_shm_name_dict(args, shapes):
    shm_name_dict = {}
    for cam in args.camera_names:
        shm_name_dict[cam] = f"shm_img_{cam}"
    for state_key in shapes["states"]:
        shm_name_dict[state_key] = f"shm_state_{state_key}"
    shm_name_dict["action"] = "shm_action"
    return shm_name_dict


def create_shm_dict(shm_name_dict, shapes, dtypes):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        size = np.prod(shape) * np.dtype(dtypes[cam]).itemsize
        shm = SharedMemory(name=shm_name_dict[cam], create=True, size=size)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        size = np.prod(shape) * np.dtype(np.float32).itemsize
        shm = SharedMemory(name=shm_name_dict[state_key], create=True, size=size)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = 14
    size = np.prod(action_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["action"], create=True, size=size)
    shm_dict["action"] = (shm, action_shape, np.float32)
    return shm_dict


def connect_shm_dict(shm_name_dict, shapes, dtypes):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        shm = SharedMemory(name=shm_name_dict[cam], create=False)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        shm = SharedMemory(name=shm_name_dict[state_key], create=False)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = (14,)
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, action_shape, np.float32)
    return shm_dict


def robot_action(action, shm_dict):
    shm, shape, dtype = shm_dict["action"]
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = action

def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5
    return min_gripper if action_value < gate else max_gripper


def init_robot(ros_operator, connected_event, start_event):
    init0 = [0, 0, 0, 0, 0, 0, 4] # 夹爪关闭
    init1 = [0, 0, 0, 0, 0, 0, 0] # 夹爪张开
    ros_operator.follow_arm_publish_continuous(init0, init0)
    connected_event.set()
    start_event.wait()
    ros_operator.follow_arm_publish_continuous(init0, init0)


def cleanup_shm(names):
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def ros_process(args, meta_queue, connected_event, start_event, shm_ready_event):
    def _ros_spin(executor):
        executor.spin()

    setup_loader(ROOT)
    rclpy.init()
    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_operator)
    spin_thread = threading.Thread(target=_ros_spin, args=(executor,), daemon=True)
    spin_thread.start()
    init_robot(ros_operator, connected_event, start_event)
    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if obs:
            shapes = {"images": {}, "states": {}, "dtypes": {}}
            for cam in args.camera_names:
                img = obs["images"][cam]
                shapes["images"][cam] = img.shape
                shapes["dtypes"][cam] = img.dtype
            shapes["states"]["eef"] = obs["eef"].shape 
            shapes["states"]["qpos"] = obs["qpos"].shape
            shapes["states"]["qvel"] = obs["qvel"].shape
            shapes["states"]["effort"] = obs["effort"].shape
            shapes["states"]["robot_base"] = obs["robot_base"].shape
            shapes["states"]["base_velocity"] = obs["base_velocity"].shape
            
            meta_queue.put(shapes)
            break
        rate.sleep()

    shm_name_dict = meta_queue.get()
    cleanup_shm(shm_name_dict.values())
    shm_dict = create_shm_dict(shm_name_dict, shapes, shapes["dtypes"])
    shm_ready_event.set()
    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if not obs:
            rate.sleep()
            continue
        
        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]
        
        shm, shape, dtype = shm_dict["action"]
        action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        if np.any(action): 
            gripper_gate = args.gripper_gate
            gripper_idx = [6, 13]
            left_action = action[:gripper_idx[0] + 1]
            if gripper_gate != -1:
                left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)
            right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
            if gripper_gate != -1:
                right_action[gripper_idx[0]] = apply_gripper_gate(right_action[gripper_idx[0]], gripper_gate)
            ros_operator.follow_arm_publish(left_action, right_action)
        rate.sleep()

    executor.shutdown()
    rclpy.shutdown()
    for shm, _, _ in shm_dict.values():
        shm.close()
        shm.unlink()


def inference_process(args, shm_dict, shapes, ros_proc):
    rate = Rate(args.frame_rate)
    server_url = "http://10.140.60.180:7891/process_frame" 
    task_prompt = "Use the right arm to pick up the blue block and place it in the transparent box"
    
    print(f"--- 客户端推理已启动 ---")
    print(f"目标服务器: {server_url}")

    # 视频录制设置 (与你之前的代码相同)
    try:
        h, w, _ = shapes["images"]["right_wrist"]
        fps = args.frame_rate
        print(f"视频录制参数已设置: (W: {w}, H: {h}, FPS: {fps})")
    except Exception as e:
        print(f"!!! 警告: 无法从 'shapes' 获取视频参数: {e} !!!")
        w, h, fps = (None, None, None)

    video_writer = None
    video_filename = ""
    session = requests.Session() # 使用 Session 保持连接

    try:
        while ros_proc.is_alive():
            timestep = 0
            
            # (每个回合开始时)
            if w is not None and video_writer is None: 
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"inference_right_wrist_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, float(fps), (w, h))
                    print(f"--- 正在录制 'right_wrist' 视频到 {video_filename} ---")
                except Exception as e:
                    print(f"!!! 警告: 无法创建 VideoWriter: {e} !!!")
                    video_writer = None

            while timestep < args.max_publish_step and ros_proc.is_alive():
                obs_dict = {"images": {}, "eef": None, "qpos": None, "qvel": None,
                            "effort": None, "robot_base": None, "base_velocity": None}

                for cam in args.camera_names:
                    shm, shape, dtype = shm_dict[cam]
                    obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
                for state_key in shapes["states"]:
                    shm, shape, dtype = shm_dict[state_key]
                    obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

                if video_writer is not None:
                    video_writer.write(obs_dict["images"]["right_wrist"])

                # 准备数据
                qpos_list = obs_dict['qpos'].tolist()
                states_json = json.dumps(qpos_list)
                payload = {
                    "text": task_prompt,
                    "states": states_json
                }

                _, img_head_buffer = cv2.imencode('.jpg', obs_dict["images"]["head"])
                _, img_left_buffer = cv2.imencode('.jpg', obs_dict["images"]["left_wrist"])
                _, img_right_buffer = cv2.imencode('.jpg', obs_dict["images"]["right_wrist"])

                files_list = [
                    ('image', ('head.jpg', img_head_buffer.tobytes(), 'image/jpeg')),
                    ('image', ('left.jpg', img_left_buffer.tobytes(), 'image/jpeg')),
                    ('image', ('right.jpg', img_right_buffer.tobytes(), 'image/jpeg')),
                ]

                response = session.post(server_url, data=payload, files=files_list, timeout=5)
                response.raise_for_status()
                action = np.array(response.json()['response'][0]) # 返回[50, 14]，取第一个动作
                

                action[6] = action[6] + 3.4; action[13] = action[13] + 3.4
                print("final_action:", action)
                robot_action(action, shm_dict)

                timestep += 1

            if video_writer is not None:
                video_writer.release()
                print(f"--- 视频 {video_filename} 已保存 (回合结束) ---")
                video_writer = None 

    finally:
        print("\n--- 'inference_process' 正在进入 finally 清理块 ---")
        if video_writer is not None and video_writer.isOpened():
            print(f"检测到中断！正在强制释放 video_writer...")
            video_writer.release()
            print(f"--- 视频 {video_filename} 已被强制保存 ---")


def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')
    parser.add_argument('--ckpt_path', type=str, default='/home/haoming/ARX/pi0_sft/single_arm_picking_data_new',
                        help='ckpt path')
    parser.add_argument('--data', type=str,
                        default='/home/haoming/ARX/ARX_collect/config.yaml',
                        help='config file')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='position lookahead step')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone model architecture')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size for input data')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist', ],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='camera names to use')
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--gripper_gate', type=float, default=1.8, help='gripper gate threshold')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    meta_queue = mp.Queue()
    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()

    ros_proc = mp.Process(target=ros_process, args=(args, meta_queue, connected_event, 
                                                   start_event, shm_ready_event))
    ros_proc.start()
    connected_event.wait()
    input("Enter any key to continue :")
    start_event.set()

    shapes = meta_queue.get()
    shm_name_dict = make_shm_name_dict(args, shapes)
    meta_queue.put(shm_name_dict)
    shm_ready_event.wait()
    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"])

    try:
        inference_process(args, shm_dict, shapes, ros_proc)
    except KeyboardInterrupt:
        pass
    finally:
        for shm, _, _ in shm_dict.values():
            shm.close()
            shm.unlink()
        ros_proc.terminate()
        ros_proc.join()


if __name__ == '__main__':
    args = parse_args()
    main(args)