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
import argparse
import yaml
import rclpy
import threading
import numpy as np
import requests
import json

from rclpy.executors import MultiThreadedExecutor
from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5
    return min_gripper if action_value < gate else max_gripper

def init_robot(ros_operator):
    init0 = [0, 0, 0, 0, 0, 0, 4]  # 夹爪关闭
    ros_operator.follow_arm_publish_continuous(init0, init0)
    print("[Init] Robot moved to initial pose (gripper closed).")


def run_inference_loop(args, ros_operator):
    rate = Rate(args.frame_rate)
    server_url = "http://10.140.60.180:7891/process_frame"
    task_prompt = "Use the right arm to pick up the blue block and place it in the transparent box"
    print(f"目标服务器: {server_url}")

    session = requests.Session()
    current_action = None  

    while rclpy.ok():
        obs = ros_operator.get_observation()
        if not obs:
            rate.sleep()
            continue

        # 构造输入信息
        gripper_idx = [6, 13]
        qpos_list = obs['qpos'].tolist()
        for i in gripper_idx:
            qpos_list[i] -= 3.74  # 处理输入前的夹爪值
        states_json = json.dumps(qpos_list)
        payload = {"text": task_prompt, "states": states_json}
        _, img_head_buffer = cv2.imencode('.jpg', obs["images"]["head"])
        _, img_left_buffer = cv2.imencode('.jpg', obs["images"]["left_wrist"])
        _, img_right_buffer = cv2.imencode('.jpg', obs["images"]["right_wrist"])

        files_list = [
            ('image', ('head.jpg',  img_head_buffer.tobytes(),  'image/jpeg')),
            ('image', ('left.jpg',  img_left_buffer.tobytes(),  'image/jpeg')),
            ('image', ('right.jpg', img_right_buffer.tobytes(), 'image/jpeg')),
        ]

        response = session.post(server_url, data=payload, files=files_list, timeout=5)
        response.raise_for_status()
        action_seq = np.array(response.json()['response'])
        print("[Client] action_seq.shape:", action_seq.shape)

        # 取第 0 个 action作为下一个动作
        current_action = action_seq[0].astype(np.float32)   # (14,)
        action = current_action.copy()  # (14,)

        # 夹爪值处理
        action[6] = action[6] + 3.74
        action[13] = action[13] + 3.74
        gripper_gate = args.gripper_gate
        left_action = action[:7].copy()
        right_action = action[7:14].copy()
        if gripper_gate != -1:
            left_action[6] = apply_gripper_gate(left_action[6], gripper_gate)
            right_action[6] = apply_gripper_gate(right_action[6], gripper_gate)

        print("[ROS] left_action[6], right_action[6]:", left_action[6], right_action[6])
        ros_operator.follow_arm_publish(left_action, right_action)
        rate.sleep()



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
                        choices=['head', 'left_wrist', 'right_wrist'],
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
    setup_loader(ROOT)
    rclpy.init()

    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_operator)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    init_robot(ros_operator)
    input("Enter any key to start inference : ")

    try:
        run_inference_loop(args, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        ros_operator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    args = parse_args()
    main(args)