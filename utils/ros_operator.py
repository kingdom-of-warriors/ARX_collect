import os
import threading
import collections
import cv2

import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from scipy.spatial.transform import Rotation as R  # eef:ZXY

import rclpy

from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, Imu
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int32MultiArray

from utils.controller import PIDController

import time


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.last_time = time.time()

    def sleep(self):
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_time = time.time()


class RosOperator(Node):
    def __init__(self, args, config, in_collect=False):
        super().__init__('robot_operator')

        from arm_control.msg._pos_cmd import PosCmd
        from arm_control.msg._joint_control import JointControl
        from arx5_arm_msg.msg._robot_cmd import RobotCmd
        from arx5_arm_msg.msg._robot_status import RobotStatus

        self.args = args
        self.config = config

        self.in_collect = in_collect

        self.base_enable = False
        self.robot_base_pose_init = [0, 0, 0]  # rlative, the head_pitch and height and head yaw is the adsolutly
        self.robot_base_target = np.zeros((6,))
        self.base_velocity_target = np.zeros((4,))
        self.base_control_thread = None

        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()

        self.bridge = CvBridge()

        self.img_head_deque = deque()
        self.img_left_deque = deque()
        self.img_right_deque = deque()

        self.img_head_depth_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()

        self.master_left_arm_deque = deque()
        self.master_right_arm_deque = deque()
        self.follow_left_arm_deque = deque()
        self.follow_right_arm_deque = deque()

        self.controller_left_deque = deque()
        self.controller_right_deque = deque()

        self.follow_left_arm_joint_deque = deque()
        self.follow_right_arm_joint_deque = deque()
        self.follow_left_arm_eef_deque = deque()
        self.follow_right_arm_eef_deque = deque()

        self.base_pose_deque = deque()
        self.robot_base_origin = deque()
        self.robot_base_deque = deque()
        self.base_velocity_deque = deque()

        self.follow_arm_publish_lock = threading.Lock()
        self.follow_arm_publish_lock.acquire()

        self.pos_cmd = PosCmd
        self.joint_control = JointControl
        self.robot_cmd = RobotCmd
        self.robot_status = RobotStatus

        # 摄像头订阅
        img_topics = {
            'img_head': 'img_head_topic',
            'img_left': 'img_left_topic',
            'img_right': 'img_right_topic',
        }
        for key, topic in img_topics.items():
            try:
                self.create_subscription(CompressedImage,
                                         self.config['camera_config'][topic],
                                         getattr(self, f"{key}_callback"),
                                         2)
            except KeyError as e:
                self.get_logger().error(f"Topic config missing: {e}")
            except AttributeError as e:
                self.get_logger().error(f"Callback not found for key: {key} -> {e}")

        if self.args.use_depth_image:
            depth_img_topics = {
                'img_head_depth': 'img_head_depth_topic',
                'img_left_depth': 'img_left_depth_topic',
                'img_right_depth': 'img_right_depth_topic',
            }
            for key, topic in depth_img_topics.items():
                try:
                    self.create_subscription(CompressedImage,
                                             self.config['camera_config'][topic],
                                             getattr(self, f"{key}_callback"),
                                             2)
                except KeyError as e:
                    self.get_logger().error(f"Topic config missing: {e}")
                except AttributeError as e:
                    self.get_logger().error(f"Callback not found for key: {key} -> {e}")

        # 机械臂订阅
        arm_topics = {
            'controller_left': ('controller_left_topic', self.pos_cmd),
            'controller_right': ('controller_right_topic', self.pos_cmd),
            'arm_feedback_left': ('arm_feedback_left_topic', self.robot_status),
            'arm_feedback_right': ('arm_feedback_right_topic', self.robot_status),

            'follow_arm_left_feedback': ('follow_arm_left_feedback_topic', self.robot_status),
            'follow_arm_right_feedback': ('follow_arm_right_feedback_topic', self.robot_status),
        }
        for key, (topic_key, msg_type) in arm_topics.items():
            try:
                self.create_subscription(msg_type,
                                         self.config['arm_config'][topic_key],
                                         getattr(self, f"{key}_callback"),
                                         2)
            except KeyError as e:
                self.get_logger().error(f"Topic config missing: {e}")
            except AttributeError as e:
                self.get_logger().error(f"Callback not found for key: {key} -> {e}")

        # 底盘订阅
        if self.args.use_base:
            self.create_subscription(self.pos_cmd,
                                     self.config['robot_base_config']['robot_base_topic'],
                                     self.robot_base_callback,
                                     2)

            if self.args.record == 'Distance':
                self.create_subscription(TFMessage,
                                         '/tf',
                                         self.base_pose_callback,
                                         2)
            if self.args.record == 'Speed':
                self.create_subscription(self.pos_cmd,
                                         self.config['robot_base_config']['robot_base_topic'],
                                         self.base_velocity_callback,
                                         2)
        # 推理模式相关发布
        if not self.in_collect:
            self.follow_arm_left_publisher = self.create_publisher(
                self.joint_control,
                self.config['arm_config']['follow_arm_left_cmd_topic'],
                10
            )
            self.follow_arm_right_publisher = self.create_publisher(
                self.joint_control,
                self.config['arm_config']['follow_arm_right_cmd_topic'],
                10
            )
            self.base_actuator_publisher = self.create_publisher(
                self.pos_cmd,
                self.config['robot_base_config']['robot_base_cmd_topic'],
                10
            )

    # 推理
    def follow_arm_publish(self, left, right):
        if len(left) == 7:
            joint_state_msg = self.joint_control()
        else:
            print("\033[31mERROR action\033[0m")

            return

        joint_state_msg.joint_pos[:7] = left
        self.follow_arm_left_publisher.publish(joint_state_msg)  # /joint_control
        if len(right) != 0:
            joint_state_msg.joint_pos[:7] = right
            self.follow_arm_right_publisher.publish(joint_state_msg)  # /joint_control2

    def init_robot_base_pose(self):
        if len(self.robot_base_origin) == 0:
            print(r'there is no base_pose_deque')

            return None
        base_pose = self.robot_base_origin.pop()
        tf_info = base_pose.transforms[0].transform
        base_quaternion = [tf_info.rotation.x, tf_info.rotation.y,
                           tf_info.rotation.z, tf_info.rotation.w]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler('xyz', degrees=False)
        base_pose = [tf_info.translation.x, -tf_info.translation.y, base_pose_yaw]
        self.robot_base_pose_init = base_pose

        self.robot_base_target = np.zeros((6,))

        return True

    def set_robot_base_target(self, target_base):
        self.robot_base_target[0] = target_base[0]  # x
        self.robot_base_target[1] = target_base[1]  # y
        self.robot_base_target[2] = target_base[2]  # Wz
        self.robot_base_target[3] = target_base[3]  # height
        self.robot_base_target[4] = target_base[4]  # head_pitch
        self.robot_base_target[5] = target_base[5]  # head_yaw

        self.base_velocity_target[0] = target_base[6]  # motor1
        self.base_velocity_target[1] = target_base[7]  # motor2
        self.base_velocity_target[2] = target_base[8]  # motor3
        self.base_velocity_target[3] = target_base[9]  # motor4

    def start_base_control_thread(self):
        if self.args.use_base:
            self.init_robot_base_pose()
            self.base_enable = True
            self.base_control_thread = threading.Thread(target=self.robot_base_control_thread,
                                                        args=())  # 执行指令单独的线程,，可以边说话边执行，多线程操作
            self.base_control_thread.start()

            return

    def visualize_pid_base(self, states, target, plot_path=None):
        STATE_NAMES = ["DX", "DY", "Yaw"]
        label1, label2 = 'states', 'target'
        states = np.array(states)
        target = np.array(target)

        num_ts, num_dim = states.shape
        fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

        all_names = [f"{name}_left" for name in STATE_NAMES] + [f"{name}_right" for name in STATE_NAMES]

        for dim_idx, ax in enumerate(axs):
            ax.plot(states[:, dim_idx], label=label1, color='orangered')
            ax.plot(target[:, dim_idx], label=label2)
            ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path)
            print(f'Saved pid control plot to: {plot_path}')
        else:
            plt.show()

        plt.close()

    def robot_base_shutdown(self):
        rate = self.create_rate(self.args.frame_rate)

        shutdown_control = self.pos_cmd()
        shutdown_control.height = self.robot_base_target[3]

        for mode in [1, 2]:
            shutdown_control.mode1 = mode
            self.base_actuator_publisher.publish(shutdown_control)

            rate.sleep()

        self.base_enable = False

        return

    def robot_base_control_thread(self):  # inference init robot arm in qpos
        rate = self.create_rate(self.args.frame_rate)
        control = self.pos_cmd()
        max_velocity = 1.0

        if self.args.record == 'Distance':
            pid_controllers = {
                'x': PIDController(kp=10.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity),
                'y': PIDController(kp=10.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity),
                'z': PIDController(kp=1.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity)
            }

            recorded_base_poses = []
            recorded_target_poses = []
            recorded_control_outputs = []
            timeout = 0

            while rclpy.ok() and self.base_enable:
                if len(self.base_pose_deque) == 0:
                    print('\033[33mThere is no base_pose_deque\033[0m')

                    timeout += 1
                    if timeout > 100:
                        self.base_enable = False
                        break
                    rate.sleep()

                    continue

                base_pose = self.base_pose_deque.pop()
                current_x, current_y, current_Wz = base_pose
                target_x, target_y, target_Wz, target_height, target_pitch, target_yaw = self.robot_base_target

                # 更新控制命令
                control.chx = pid_controllers['x'].update(current_x, target_x, dt=0.017)
                control.chy = pid_controllers['y'].update(current_y, target_y, dt=0.017)
                control.chz = pid_controllers['z'].update(current_Wz, target_Wz, dt=0.017)
                control.height = target_height
                control.head_pit = target_pitch
                control.head_yaw = target_yaw
                control.mode1 = 1

                # 记录数据
                # target_pose = [target_x, target_y, current_Wz]
                output_control = [control.chx, control.chy, control.chz]

                # recorded_base_poses.append(base_pose)
                # recorded_target_poses.append(target_pose)
                recorded_control_outputs.append(output_control)

                self.base_actuator_publisher.publish(control)
                rate.sleep()

            if not self.base_enable:
                self.robot_base_shutdown()

                plot_path = (
                    os.path.join(self.args.ckpt_dir, f"{self.args.ckpt_name}_PID.png")
                    if self.args.episode_path == "./datasets"
                    else os.path.join(f"{self.args.episode_path}_PID.png")
                )
                self.visualize_pid_base(recorded_base_poses, recorded_target_poses, plot_path=plot_path)
        if self.args.record == 'Speed':
            while rclpy.ok() and self.base_enable:
                _, _, _, target_height, target_pitch, target_yaw = self.robot_base_target
                target_motor1, target_motor2, target_motor3, target_motor4 = self.base_velocity_target

                control.height = target_height
                control.head_pit = target_pitch
                control.head_yaw = target_yaw
                control.temp_float_data[1:5] = [target_motor1, target_motor2, target_motor3, target_motor4]
                control.mode1 = 3

                self.base_actuator_publisher.publish(control)
                rate.sleep()

        return

    def follow_arm_publish_continuous(self, left_target, right_target):
        arm_steps_length = [0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.2]
        left_arm = None
        right_arm = None

        rate = self.create_rate(self.args.frame_rate)
        while rclpy.ok():
            if len(self.follow_left_arm_deque) != 0:
                left_arm = list(self.follow_left_arm_deque[-1].joint_pos)

            if len(self.follow_right_arm_deque) != 0:
                right_arm = list(self.follow_right_arm_deque[-1].joint_pos)

            if left_arm is not None and right_arm is not None:
                break

        # 计算方向标志位
        left_symbol = [1 if left_target[i] - left_arm[i] > 0 else -1 for i in range(len(left_target))]
        right_symbol = [1 if right_target[i] - right_arm[i] > 0 else -1 for i in range(len(right_target))]

        step = 0
        while rclpy.ok():
            left_done = 0
            right_done = 0

            if self.follow_arm_publish_lock.acquire(False):
                return

            left_done = self._update_arm_position(left_target, left_arm, left_symbol, arm_steps_length)
            right_done = self._update_arm_position(right_target, right_arm, right_symbol, arm_steps_length)

            if left_done > len(left_target) - 1 and right_done > len(right_target) - 1:
                print('left_done and right_done')

                break

            # JointControl topic
            if len(left_arm) == 7:
                joint_state_msg = self.joint_control()
            else:
                print("\033[31mInvalid joint length\033[0m")

                return

            joint_state_msg.joint_pos[:7] = left_target
            self.follow_arm_left_publisher.publish(joint_state_msg)
            rate.sleep()

            joint_state_msg.joint_pos[:7] = right_target
            self.follow_arm_right_publisher.publish(joint_state_msg)

            step += 1
            print("follow_arm_publish_continuous:", step)
            rate.sleep()

    def _extract_eef_data(self, eef):
        return [eef.x, eef.y, eef.z, eef.roll, eef.pitch, eef.yaw]

    def get_observation(self, ts=-1):  # get the robot observation
        img_data = {
            'head': None,
            'left_wrist': None,
            'right_wrist': None,
        }
        img_depth_data = {
            'head': None,
            'left_wrist': None,
            'right_wrist': None,
        }
        arm_data = {
            'follow_left_arm': self.robot_status(),
            'follow_right_arm': self.robot_status(),
        }

        # 获取图像信息
        for cam_name in self.args.camera_names:
            if cam_name in img_data:
                deque_map = {
                    'head': self.img_head_deque,
                    'left_wrist': self.img_left_deque,
                    'right_wrist': self.img_right_deque,
                }

                if len(deque_map[cam_name]) == 0:
                    print(f'there is no {cam_name}_deque')

                    return None

                # 是否压缩处理图像
                img_data[cam_name] = self.bridge.compressed_imgmsg_to_cv2(deque_map[cam_name].pop(),
                                                                          'passthrough')

            if self.args.use_depth_image:
                if cam_name in img_depth_data:
                    deque_map = {
                        'head_depth': self.img_head_depth_deque,
                        'left_wrist_depth': self.img_left_depth_deque,
                        'right_wrist_depth': self.img_right_depth_deque,
                    }

                    key = cam_name + '_depth'

                    if len(deque_map[key]) == 0:
                        print(f'there is no {key}_deque')

                        return None

                    img_depth_data[key] = self.bridge.imgmsg_to_cv2(deque_map[key].pop(),
                                                                    'passthrough')

        # 获取机械臂状态
        for arm_name in ['follow_left_arm', 'follow_right_arm']:
            deque_map = {
                'follow_left_arm': self.follow_left_arm_deque,
                'follow_right_arm': self.follow_right_arm_deque,
            }

            if len(deque_map[arm_name]) == 0:
                print(f'there is no {arm_name}_deque')

                return None

            arm_data[arm_name] = deque_map[arm_name].pop()

        obs_dict = collections.OrderedDict()  # 有序的字典

        # 保存图像
        obs_dict['images'] = {cam: img for cam, img in img_data.items() if cam in self.args.camera_names}

        if self.args.use_depth_image:
            obs_dict['images_depth'] = {cam: img_depth_data[cam] for cam in img_depth_data if
                                        cam in self.args.camera_names}

        # 保存机械臂状态
        left_eef = np.concatenate([
            arm_data['follow_left_arm'].end_pos,
            [arm_data['follow_left_arm'].joint_pos[-1]],
        ])

        right_eef = np.concatenate([
            arm_data['follow_right_arm'].end_pos,
            [arm_data['follow_right_arm'].joint_pos[-1]]
        ])

        obs_dict['eef'] = np.concatenate((left_eef, right_eef), axis=0)
        obs_dict['qpos'] = np.concatenate((np.array(arm_data['follow_left_arm'].joint_pos),
                                           np.array(arm_data['follow_right_arm'].joint_pos)), axis=0)
        obs_dict['qvel'] = np.concatenate((np.array(arm_data['follow_left_arm'].joint_vel),
                                           np.array(arm_data['follow_right_arm'].joint_vel)), axis=0)
        obs_dict['effort'] = np.concatenate((np.array(arm_data['follow_left_arm'].joint_cur),
                                             np.array(arm_data['follow_right_arm'].joint_cur)), axis=0)

        # 保存底盘状态
        if self.args.use_base and ts != 0:
            if len(self.robot_base_deque) == 0:
                print(r'there is no robot_base_deque, maby there is no VR message')

                return None

            if self.args.record == 'Distance':
                if len(self.base_pose_deque) == 0:
                    print(r'there is no base_pose_deque')

                    return None
            if self.args.record == 'Speed':
                if len(self.base_velocity_deque) == 0:
                    print(r'there is no base_velocity_deque')

                    return None

            robot_base = self.robot_base_deque.pop()

            if self.args.record == 'Distance':
                base_pose = self.base_pose_deque.pop()
                obs_dict['robot_base'] = [base_pose[0], base_pose[1], base_pose[2], robot_base.height,
                                          robot_base.head_pit, robot_base.head_yaw]

                obs_dict['base_velocity'] = np.zeros((4,))
            if self.args.record == 'Speed':
                obs_dict['robot_base'] = [0, 0, 0,
                                          robot_base.height, robot_base.head_pit, robot_base.head_yaw]

                base_velocity = self.base_velocity_deque.pop()
                obs_dict['base_velocity'] = [base_velocity[0], base_velocity[1], base_velocity[2], base_velocity[3]]
        else:
            obs_dict['robot_base'] = np.zeros((6,))
            obs_dict['base_velocity'] = np.zeros((4,))

        return obs_dict

    def get_action(self):
        joints_dim = 7

        action_dict = collections.OrderedDict()

        deque_map = {
            'control_left_arm_deque': self.controller_left_deque,
            'control_right_arm_deque': self.controller_right_deque,
        }

        for name, deque in deque_map.items():
            if len(deque) == 0:
                print(f'there is no {name}')

                return None

        # 获取主臂状态
        left_frame = deque_map['control_left_arm_deque'].pop()
        right_frame = deque_map['control_right_arm_deque'].pop()

        control_left_arm = self._extract_eef_data(left_frame)
        control_right_arm = self._extract_eef_data(right_frame)
        control_left_arm_gripper = left_frame.gripper
        control_right_arm_gripper = right_frame.gripper

        # 主臂保存状态
        control_left_arm_eef = np.concatenate([control_left_arm, [control_left_arm_gripper]])
        control_right_arm_eef = np.concatenate([control_right_arm, [control_right_arm_gripper]])

        # 构建动作字典
        action_dict['action'] = np.zeros((joints_dim * 2,))
        action_dict['action_qvel'] = np.zeros((joints_dim * 2,))
        action_dict['action_eef'] = np.concatenate((control_left_arm_eef,
                                                    control_right_arm_eef), axis=0)
        action_dict['action_base'] = np.zeros((13,))  # waiting for the obersevation

        return action_dict

    def img_head_callback(self, msg):
        if len(self.img_head_deque) >= 2000:
            self.img_head_deque.popleft()
        self.img_head_deque.append(msg)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_head_depth_callback(self, msg):
        if len(self.img_head_depth_deque) >= 2000:
            self.img_head_depth_deque.popleft()
        self.img_head_depth_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def master_left_arm_callback(self, msg):
        if len(self.master_left_arm_deque) >= 2000:
            self.master_left_arm_deque.popleft()
        self.master_left_arm_deque.append(msg)

    def master_right_arm_callback(self, msg):
        if len(self.master_right_arm_deque) >= 2000:
            self.master_right_arm_deque.popleft()
        self.master_right_arm_deque.append(msg)

    def follow_left_arm_callback(self, msg):
        if len(self.follow_left_arm_deque) >= 2000:
            self.follow_left_arm_deque.popleft()
        self.follow_left_arm_deque.append(msg)

    def follow_right_arm_callback(self, msg):
        if len(self.follow_right_arm_deque) >= 2000:
            self.follow_right_arm_deque.popleft()
        self.follow_right_arm_deque.append(msg)

    def controller_left_callback(self, msg):
        if len(self.controller_left_deque) >= 2000:
            self.controller_left_deque.popleft()
        self.controller_left_deque.append(msg)

    def controller_right_callback(self, msg):
        if len(self.controller_right_deque) >= 2000:
            self.controller_right_deque.popleft()
        self.controller_right_deque.append(msg)

    def arm_feedback_left_callback(self, msg):
        if len(self.follow_left_arm_deque) >= 2000:
            self.follow_left_arm_deque.popleft()
        self.follow_left_arm_deque.append(msg)

    def arm_feedback_right_callback(self, msg):
        if len(self.follow_right_arm_deque) >= 2000:
            self.follow_right_arm_deque.popleft()
        self.follow_right_arm_deque.append(msg)

    def follow_arm_left_feedback_callback(self, msg):
        if len(self.follow_left_arm_deque) >= 2000:
            self.follow_left_arm_deque.popleft()
        self.follow_left_arm_deque.append(msg)

    def follow_arm_right_feedback_callback(self, msg):
        if len(self.follow_right_arm_deque) >= 2000:
            self.follow_right_arm_deque.popleft()
        self.follow_right_arm_deque.append(msg)

    # robot robot_base
    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def base_pose_callback(self, msg):
        if len(self.base_pose_deque) >= 2:
            self.base_pose_deque.popleft()

        if len(self.robot_base_origin) >= 2:
            self.robot_base_origin.popleft()
        self.robot_base_origin.append(msg)

        tf_info = msg.transforms[0].transform
        base_quaternion = [tf_info.rotation.x, tf_info.rotation.y,
                           tf_info.rotation.z, tf_info.rotation.w]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler('xyz', degrees=False)
        base_pose = [tf_info.translation.x, -tf_info.translation.y, base_pose_yaw]

        base_pose[0] = base_pose[0] - self.robot_base_pose_init[0]  # 如果这个值是负的
        base_pose[1] = base_pose[1] - self.robot_base_pose_init[1]
        base_pose[2] = base_pose[2] - self.robot_base_pose_init[2]

        self.base_pose_deque.append(base_pose)

    def base_velocity_callback(self, msg):
        if len(self.base_velocity_deque) >= 2:
            self.base_velocity_deque.popleft()

        velocity = msg.temp_float_data[1:5]

        self.base_velocity_deque.append(velocity)

    def _update_arm_position(self, target, arm, symbol, steps_length):
        diff = [abs(target[i] - arm[i]) for i in range(len(target))]
        done = 0
        for i in range(len(target)):
            if diff[i] < steps_length[i]:
                arm[i] = target[i]
                done += 1
            else:
                arm[i] += symbol[i] * steps_length[i]

        return done
