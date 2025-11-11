#!/bin/bash
workspace=$(pwd)
source ~/.bashrc

echo "================================================="
echo "            启动初始化节点 (只运行一次)          "
echo "================================================="

# CAN 配置
gnome-terminal -t "can" -x bash -c "cd /home/haoming/ARX/LIFT/ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
gnome-terminal -t "can" -x bash -c "cd /home/haoming/ARX/LIFT/ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
gnome-terminal -t "can" -x bash -c "cd /home/haoming/ARX/LIFT/ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 20 # 等待CAN启动

# 启动升降节点
gnome-terminal --title="body" -x bash -c "cd /home/haoming/ARX/LIFT/body/ROS2; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; exec bash;"
sleep 3

# 启动相机节点
gnome-terminal --title="realsense" -x bash -c "cd /home/haoming/ARX/ROS2_LIFT_Play/realsense/; ./realsense.sh; exec bash;"
sleep 3

# 设置升降高度 (0-20)
gnome-terminal -t "head_pose" -x bash -c "cd /home/haoming/ARX/LIFT/body/ROS2; source install/setup.bash && ros2 topic pub -1 /ARX_VR_L arm_control/msg/PosCmd '{height: 14.0}'; exec bash;"
sleep 2

# 主循环
while true
do
    echo "================================================="
    echo "           准备开始新一轮数据采集...             "
    echo "================================================="

    # 确保之前的复位进程已结束
    pkill -f "open_double_arm.launch.py" 2>/dev/null
    sleep 2

    # 启动重力补偿机械臂节点
    echo "-> 正在启动重力补偿节点 (L)..."
    gnome-terminal -t "L" -x bash -c "cd /home/haoming/ARX/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_collect.launch.py; exec bash;"
    sleep 5

    # 启动一条数据采集代码 (在当前终端运行，以阻塞脚本)
    echo "-> 启动数据采集程序，请根据提示操作..."
    (
        source ~/miniconda3/etc/profile.d/conda.sh
        source /home/haoming/ARX/LIFT/ARX_X5/ROS2/X5_ws/install/setup.bash
        conda activate lerobot
        python collect_arm_once.py --frame_rate 30 --task "pick_blockings"
    )
    echo "-> 数据采集程序已结束。"

    # 杀死重力补偿节点 L
    echo "-> 正在停止重力补偿节点 (L)..."
    pkill -f "v2_collect.launch.py" 2>/dev/null
    sleep 2

    # 启动复位程序 lift
    echo "-> 正在启动机械臂复位程序 (lift)..."
    gnome-terminal --title="lift" -x bash -c "cd /home/haoming/ARX/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash; ros2 launch arx_x5_controller open_double_arm.launch.py; exec bash;"
    echo "✓ 机械臂已复位。"
    
    # 询问是否继续
    echo ""
    read -p "是否开始下一条数据采集? (y/n): " choice
    case "$choice" in 
      y|Y ) echo "好的，准备下一轮...";;
      n|N ) echo "结束采集。"; break;;
      * ) echo "无效输入，默认结束采集。"; break;;
    esac
done

echo "================================================="
echo "所有采集任务已完成，请手动关闭其他终端窗口。"
echo "================================================="