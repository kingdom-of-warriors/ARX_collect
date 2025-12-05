import os
import sys
import h5py
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
import time
import argparse
import yaml
from einops import rearrange
import rclpy
import torch
import threading

from rclpy.executors import MultiThreadedExecutor

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader

import json
import numpy as np

from lerobot.policies.pi0.modeling_pi0 import PI0Policy
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def auto_model_from_pretrained(path, **kwargs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = PI0Policy.from_pretrained(path, **kwargs).to(device)
    cfg_path = os.path.join(path, "train_config.json")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    return model, cfg


def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5
    return min_gripper if action_value < gate else max_gripper


def init_robot(ros_operator):
    """初始化机械臂到初始位置"""
    closed = 4
    left_qpos  = [0, 0, 0, 0, 0, 0, closed]
    right_qpos = [0, 0, 0, 0, 0, 0, closed]
    ros_operator.follow_arm_publish_continuous(left_qpos, right_qpos)
    print("机械臂已初始化到初始位置")


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')

    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/haoming/ARX/pi0_sft/single_arm_picking_pi0_sft_new_room_v1',
                        help='ckpt path')
    parser.add_argument('--is_delta', type=bool, default=False)

    parser.add_argument('-b', '--gripper_binary', action='store_true', help='gripper binary action')
    parser.add_argument('--data', type=str,
                        default='/home/haoming/ARX/ARX_collect/config.yaml',
                        help='config file')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='camera names to use')
    
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    
    parser.add_argument('--gripper_gate', type=float, default=1.5, help='gripper gate threshold')

    parser.add_argument('--wait_after_settle', type=float, default=0.0,
                        help='等待机械臂稳定后再获取 obs 的时间(秒)')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    setup_loader(ROOT)
    rclpy.init()

    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_operator)

    def _ros_spin(executor):
        executor.spin()
    
    spin_thread = threading.Thread(target=_ros_spin, args=(executor,), daemon=True)
    spin_thread.start()

    init_robot(ros_operator)
    input("按任意键开始推理...")
    device = "cuda:0"
    model, cfg = auto_model_from_pretrained(args.ckpt_path)
    model.eval()
    print("模型加载成功。")

    task_prompt = ["Use the right arm to pick up the blue block and place it in the transparent box"]
    repo_id = cfg["dataset"]["repo_id"]
    print("repo_id:", repo_id)

    obs = ros_operator.get_observation()
    while not obs:
        time.sleep(0.01)
        obs = ros_operator.get_observation()

    qpos_record = []
    action_record = []
    gripper_idx = [6, 13]

    get_obs_times = []
    inference_times = []
    publish_times = []
    total_step_times = []
    rate = Rate(args.frame_rate)
    timestep = 0

    try:
        while timestep < args.max_publish_step and rclpy.ok():
            step_start = time.time()
            obs_start = time.time()
            
            if args.wait_after_settle > 0 and len(model._action_queue) == 0:
                time.sleep(args.wait_after_settle)

            obs = ros_operator.get_observation()
            if not obs:
                rate.sleep()
                continue

            obs_end = time.time()
            get_obs_times.append(obs_end - obs_start)

            obs['qpos'][6] -= 3.74
            obs['qpos'][13] -= 3.74

            infer_start = time.time()
            with torch.inference_mode():
                input_data = {}
                
                img_head = rearrange(obs["images"]["head"], 'h w c -> c h w')
                input_data["observation.images.head"] = torch.from_numpy(
                    img_head / 255.0).float().to(device).unsqueeze(0)
                img_left = rearrange(obs["images"]["left_wrist"], 'h w c -> c h w')
                input_data["observation.images.left_wrist"] = torch.from_numpy(
                    img_left / 255.0).float().to(device).unsqueeze(0)
                img_right = rearrange(obs["images"]["right_wrist"], 'h w c -> c h w')
                input_data["observation.images.right_wrist"] = torch.from_numpy(
                    img_right / 255.0).float().to(device).unsqueeze(0)
                
                input_data["observation.state"] = torch.from_numpy(
                    obs['qpos']).float().to(device).unsqueeze(0)
                input_data["task"] = task_prompt
                input_data["repo_id"] = repo_id

                action_tensor = model.select_action(input_data)
                qpos_record.append(obs['qpos'].copy())

            infer_end = time.time()
            inference_times.append(infer_end - infer_start)
            action = action_tensor.squeeze(0).cpu().numpy()
            action_record.append(action.copy())

            if args.is_delta:
                action_delta = action
                mask = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
                action = (action + obs['qpos']) * (1 - mask) + mask * action_delta

            if args.gripper_binary:
                action[6] = np.where(action[6] < 0.5, 5.0, 0.0)
                action[13] = np.where(action[13] < 0.5, 5.0, 0.0)
            else:
                action[6] += 3.74
                action[13] += 3.74

            publish_start = time.time()
            gripper_gate = args.gripper_gate
            left_action = action[:gripper_idx[0] + 1]
            if gripper_gate != -1:
                left_action[gripper_idx[0]] = apply_gripper_gate(
                    left_action[gripper_idx[0]], gripper_gate)

            right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
            if gripper_gate != -1:
                right_action[gripper_idx[0]] = apply_gripper_gate(
                    right_action[gripper_idx[0]], gripper_gate)

            ros_operator.follow_arm_publish(left_action, right_action)
            publish_end = time.time()
            publish_times.append(publish_end - publish_start)

            step_end = time.time()
            total_step_times.append(step_end - step_start)

            timestep += 1
            rate.sleep()

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在退出...")
    finally:
        if len(action_record) > 0:
            try:
                print("\n正在保存 action_record 到 HDF5...")
                action_data = np.array(action_record)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                datasets_dir = Path(ROOT) / "datasets"
                datasets_dir.mkdir(parents=True, exist_ok=True)
                
                hdf5_filename = datasets_dir / f"infer_{timestamp}.hdf5"
                
                with h5py.File(hdf5_filename, 'w') as f:
                    obs_group = f.create_group('observations')
                    obs_group.create_dataset('qpos', data=action_data, compression='gzip')
                
                print(f"--- action_record 已保存至: {hdf5_filename} ---")
                print(f"    数据形状: {action_data.shape}")
            except Exception as e:
                print(f"!!! 警告: HDF5 保存失败: {e} !!!")

        print("\n=== 延迟统计 ===")
        if len(get_obs_times) > 0:
            print(f"get_observation 平均耗时: {np.mean(get_obs_times)*1000:.2f} ms")
        if len(inference_times) > 0:
            print(f"模型推理平均耗时:      {np.mean(inference_times)*1000:.2f} ms")
        if len(publish_times) > 0:
            print(f"发布动作平均耗时:      {np.mean(publish_times)*1000:.2f} ms")
        if len(total_step_times) > 0:
            print(f"单步总耗时:            {np.mean(total_step_times)*1000:.2f} ms")
            print(f"实际控制频率:          {1.0/np.mean(total_step_times):.2f} Hz")

        if hasattr(model, 'start_queue') and hasattr(model, 'end_queue'):
            min_len = min(len(model.start_queue), len(model.end_queue))
            if min_len >= 2:
                start_queue = np.array(model.start_queue[:min_len])
                end_queue = np.array(model.end_queue[:min_len])
                infer_mean_time = (end_queue - start_queue).mean()
                action_time_queue = start_queue[1:] - end_queue[:-1]
                print(f"\n模型内部推理平均耗时: {infer_mean_time*1000:.2f} ms")
                print(f"两次推理间隔平均:     {action_time_queue.mean()*1000:.2f} ms")

        if len(qpos_record) > 0:
            try:
                print("\n正在生成 qpos 轨迹图...")
                data = np.array(qpos_record)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig, axes = plt.subplots(7, 2, figsize=(15, 25))
                axes = axes.flatten()
                x_ticks = np.arange(0, data.shape[0] + 1, 50)

                for i in range(min(14, data.shape[1])):
                    axes[i].plot(data[:, i])
                    axes[i].set_title(f"Joint {i}")
                    axes[i].set_xlabel("Timestep")
                    axes[i].set_ylabel("Position")
                    axes[i].set_xticks(x_ticks)
                    axes[i].grid(True)
                
                plt.tight_layout()
                plot_filename = f"images/qpos_plot_{timestamp}.png"
                plt.savefig(plot_filename)
                print(f"--- qpos 曲线已保存至: {plot_filename} ---")
                plt.close()
            except Exception as e:
                print(f"!!! 警告: qpos 绘图失败: {e} !!!")

        executor.shutdown()
        rclpy.shutdown()
        spin_thread.join(timeout=2)
        print("程序已退出")


if __name__ == '__main__':
    args = parse_args()
    main(args)