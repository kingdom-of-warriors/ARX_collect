#!/bin/bash
workspace=$(pwd)
source ~/.bashrc

# CAN 配置
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
gnome-terminal -t "can" -x sudo bash -c "cd /home/arx/LIFT/ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 3 # 不能删除，需要等待启动完成

# 启动升降节点
gnome-terminal --title="body" -x bash -c "cd /home/arx/LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 3
gnome-terminal --title="lift" -x bash -c "cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; $shell_exec"
LIFT_PID=$!

# 第 20 行设置升降高度，0-20 之间
sleep 5
gnome-terminal -t "head_pose" -x  bash -c "cd /home/arx/LIFT/body/ROS2; \
source install/setup.bash && \
ros2 topic pub -l /lift_height_cmd arm_control/msg/PosCmd '{height: 10.0}' ; \
exec bash;"

# 杀死 lift 进程，来解锁机械臂关节
sleep 5
kill $LIFT_PID 2>/dev/null
pkill -f "open_double_arm.launch.py" 2>/dev/null
sleep 5

# 启动重力补偿机械臂节点
gnome-terminal -t "L" -x  bash -c "cd /home/arx/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_collect.launch.py; exec bash;"
sleep 3

# 启动相机节点
gnome-terminal --title="realsense" -x bash -c "cd /home/arx/ROS2_LIFT_Play/realsense/; ./realsense.sh; $shell_exec"
sleep 3

# 启动数据采集代码
gnome-terminal -t "collect_arx" -x  bash -c "cd /home/arx/collect; \
source ~/miniconda3/etc/profile.d/conda.sh; conda activate act; \
python collect_arm.py --frame_rate 30 --task "lift_object"; exec bash;"
