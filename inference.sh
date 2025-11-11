#!/bin/bash

workspace=$(pwd)

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# CAN 配置
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 3 # 等待CAN启动

# 启动升降节点
gnome-terminal --title="body" -x bash -c "cd /home/arx/LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; exec bash;"
sleep 3

# 启动相机节点
gnome-terminal --title="realsense" -x bash -c "cd /home/arx/ROS2_LIFT_Play/realsense/; ./realsense.sh; exec bash;"
sleep 3

# 设置升降高度 (0-20)
gnome-terminal -t "head_pose" -x bash -c "cd /home/arx/LIFT/body/ROS2; source install/setup.bash && ros2 topic pub -1 /ARX_VR_L arm_control/msg/PosCmd '{height: 14.0}'; exec bash;"
sleep 2

# 机械臂复位
gnome-terminal --title="lift" -x bash -c "cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; exec bash;"

# Inference
gnome-terminal --title="inference" -x bash -c "cd /home/arx/ROS2_LIFT_Play/act; source ~/miniconda3/etc/profile.d/conda.sh; conda activate act; python inference.py; $shell_exec"   