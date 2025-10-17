#!/bin/bash
echo "======================================"
echo "停止所有机器人系统"
echo "======================================"

# 杀死所有机械臂
pkill -9 -f X5Controller
pkill -9 -f arx_x5_controller

# 杀死底盘
pkill -9 -f arx_lift_controller
pkill -9 -f lift.launch

# 杀死相机
pkill -9 -f realsense

# 杀死所有ROS2
pkill -9 -f "ros2 launch"
pkill -9 -f "ros2 topic"
pkill -9 -f ros2

# 停止CAN
sudo pkill -9 -f arx_can
sudo ip link set can1 down 2>/dev/null
sudo ip link set can3 down 2>/dev/null
sudo ip link set can5 down 2>/dev/null

sleep 2

# 验证
echo ""
echo "验证结果:"
remaining=$(ros2 node list 2>/dev/null | wc -l)
if [ $remaining -eq 0 ]; then
    echo "✓ 所有节点已停止"
else
    echo "⚠ 还有 $remaining 个节点在运行"
    ros2 node list 2>/dev/null
fi

echo ""
echo "完成！"