#!/bin/bash

workspace=$(pwd)

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# CAN
gnome-terminal -t "can1" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can3" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
sleep 0.3
gnome-terminal -t "can5" -x bash -c "cd ${workspace}; cd ../../LIFT/ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 0.3

# Body
gnome-terminal --title="body" -x $shell_type -i -c "cd ../../LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 1

# Lift
gnome-terminal --title="lift" -x $shell_type -i -c "cd ../../LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; $shell_exec"
sleep 1

# Realsense
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${workspace}; cd ../realsense; ./realsense.sh; $shell_exec"
sleep 3

# Inference
gnome-terminal --title="inference" -x $shell_type -i -c "cd ${workspace}; cd ../act; conda activate act; python inference.py; $shell_exec"   