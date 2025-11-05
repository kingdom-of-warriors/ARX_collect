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


import argparse
import collections
import cv2
import yaml
from einops import rearrange
import rclpy
import torch
import threading

from rclpy.executors import MultiThreadedExecutor

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader

import sys

obs_dict = collections.OrderedDict()

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np

# PI0 加载模型
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

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



def auto_model_from_pretrained(path, **kwargs):
    map_location = "cuda:0"
    return PI0Policy.from_pretrained(path, **kwargs).to(map_location)


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        # print(f'{cam_name=}')
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5

    return min_gripper if action_value < gate else max_gripper


def get_obervations(args, timestep, ros_operator):
    global obs_dict

    rate = Rate(args.frame_rate)
    while True and rclpy.ok():
        obs_dict = ros_operator.get_observation(ts=timestep)
        if not obs_dict:
            print("syn fail")
            rate.sleep()

            continue

        return obs_dict


def init_robot(ros_operator, connected_event, start_event):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    # 发布初始位置（关节空间姿态）
    ros_operator.follow_arm_publish_continuous(init0, init0)
    # ros_operator.robot_base_shutdown()

    connected_event.set()
    start_event.wait()

    ros_operator.follow_arm_publish_continuous(init1, init1)


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
            shapes["states"]["eef"] = obs["eef"].shape # add eef
            shapes["states"]["qpos"] = obs["qpos"].shape
            shapes["states"]["qvel"] = obs["qvel"].shape
            shapes["states"]["effort"] = obs["effort"].shape
            shapes["states"]["robot_base"] = obs["robot_base"].shape
            shapes["states"]["base_velocity"] = obs["base_velocity"].shape
            

            meta_queue.put(shapes)

            break

        rate.sleep()

    # 创建共享内存
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

        # 写入共享内存
        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]

        # 读取动作并执行
        shm, shape, dtype = shm_dict["action"]
        action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        if np.any(action):  # 确保动作不全是 0
            gripper_gate = args.gripper_gate

            gripper_idx = [6, 13]

            left_action = action[:gripper_idx[0] + 1]  # 取8维度
            if gripper_gate != -1:
                left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

            right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
            if gripper_gate != -1:
                right_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

            ros_operator.follow_arm_publish(left_action, right_action)

        rate.sleep()

    executor.shutdown()
    rclpy.shutdown()
    for shm, _, _ in shm_dict.values():
        shm.close()
        shm.unlink()


def inference_process(args, shm_dict, shapes, ros_proc):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    H, W = 224, 224
    model = PI0Policy.from_pretrained(args.ckpt_path).to(device)
    model.eval()  # 切换到评估模式
    print("模型加载成功。")

    task_prompt = ["pick up the red block"]
    repo_id = ["dual_arm_picking"]  # 必须与模型训练时的 repo_id 匹配

    while ros_proc.is_alive():
        timestep = 0

        while timestep < args.max_publish_step and ros_proc.is_alive():
            obs_dict = {"images": {}, "eef": None, "qpos": None, "qvel": None,
                        "effort": None, "robot_base": None, "base_velocity": None}

            # 读取图像数据
            for cam in args.camera_names:
                shm, shape, dtype = shm_dict[cam]
                obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            
            # 读取状态数据
            for state_key in shapes["states"]: # 假设 states 包含 'qpos' 和 'eef'
                shm, shape, dtype = shm_dict[state_key]
                obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

            with torch.inference_mode():
                input_data = {}

                # 处理图像 (HWC -> CHW, Resize, ToTensor, ToDevice, Batch)
                img_head = obs_dict["images"]["head"]
                img_head = rearrange(img_head, 'h w c -> c h w') # (C, H, W)
                input_data["observation.images.head"] = torch.from_numpy(img_head / 255.0).float().to(device).unsqueeze(0)

                img_left = obs_dict["images"]["left_wrist"]
                img_left = rearrange(img_left, 'h w c -> c h w')
                input_data["observation.images.left_wrist"] = torch.from_numpy(img_left / 255.0).float().to(device).unsqueeze(0)
                
                img_right = obs_dict["images"]["right_wrist"]
                img_right = rearrange(img_right, 'h w c -> c h w')
                input_data["observation.images.right_wrist"] = torch.from_numpy(img_right / 255.0).float().to(device).unsqueeze(0)
                
                # 处理状态 (ToFloat, ToTensor, ToDevice, Batch)
                input_data["observation.state"] = torch.from_numpy(obs_dict['qpos']).float().to(device).unsqueeze(0)
                input_data["observation.eef"] = torch.from_numpy(obs_dict['eef']).float().to(device).unsqueeze(0)
                
                # 添加文本输入
                input_data["task"] = task_prompt
                input_data["repo_id"] = repo_id

                # 执行模型推理
                action_tensor = model.select_action(input_data) # 预期 shape [1, 14]

            # 将动作写回共享内存 ---
            action = action_tensor.squeeze(0).cpu().numpy() # 变为 (14,)
            action[6] = action[6] + 3.36; action[13] = action[13] + 3.36  # gripper offset
            robot_action(action, shm_dict)

            timestep += 1

        robot_action(action, shm_dict)


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')

    # 检查点设置
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='ckpt path')

    # 配置文件
    parser.add_argument('--data', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 推理设置
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='position lookahead step')

    # 模型结构设置
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone model architecture')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size for input data')

    # 摄像头设置
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist', ],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='camera names to use')

    # 机器人设置
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    # 状态和动作设置
    parser.add_argument('--gripper_gate', type=float, default=-1, help='gripper gate threshold')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    meta_queue = mp.Queue()

    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()

    # 启动ROS进程
    ros_proc = mp.Process(target=ros_process, args=(args, meta_queue, connected_event, 
                                                    start_event, shm_ready_event))
    ros_proc.start()

    connected_event.wait()
    input("Enter any key to continue :")
    start_event.set()

    # 等待meta信息
    shapes = meta_queue.get()
    shm_name_dict = make_shm_name_dict(args, shapes)
    meta_queue.put(shm_name_dict)
    shm_ready_event.wait()
    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"])

    # 推理
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
