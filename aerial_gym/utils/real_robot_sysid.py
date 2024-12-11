#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import PositionTarget


class PositionTargetCommandNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("position_target_command_node")

        # 发布者
        self.pos_target_pub = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=10
        )

    def send_position_target_command(
        self, x_command, y_command, z_command, yaw_rate_command, mode="velocity"
    ):
        """
        发送位置目标命令
        
        参数:
        x_command: x方向的速度或加速度指令
        y_command: y方向的速度或加速度指令
        z_command: z方向的速度或加速度指令
        yaw_rate_command: 偏航率指令
        mode: 指定模式，"velocity"表示使用速度，"acceleration"表示使用加速度
        """
        
        msg = PositionTarget()  # 创建PositionTarget消息对象
        msg.header.stamp = rospy.Time.now()  # 设置时间戳
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED  # 设置坐标系为机体NED（北东下）

        # 忽略位置并使用速度
        msg.type_mask = (
            PositionTarget.IGNORE_PX
            + PositionTarget.IGNORE_PY
            + PositionTarget.IGNORE_PZ
            + PositionTarget.IGNORE_YAW
        )
        
        # 根据模式设置type_mask
        if mode == "velocity":
            msg.type_mask += (
                PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
            )  # 忽略加速度
        elif mode == "acceleration":
            msg.type_mask += (
                PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
            )  # 忽略速度

        # 根据模式设置速度或加速度
        if mode == "velocity":
            # 设置速度
            msg.velocity.x = x_command
            msg.velocity.y = y_command
            msg.velocity.z = z_command
        elif mode == "acceleration":
            # 设置加速度
            msg.acceleration_or_force.x = x_command
            msg.acceleration_or_force.y = y_command
            msg.acceleration_or_force.z = z_command

        # 设置偏航率
        msg.yaw_rate = yaw_rate_command
        self.pos_target_pub.publish(msg)  # 发布消息

    def run(self):
        rate = rospy.Rate(10)  # 设定循环频率为10Hz

        while not rospy.is_shutdown():
            # 示例：发送组合的速度和加速度命令
            # 以1 m/s的速度向前移动，并具有0.5 m/s^2的前进加速度
            self.send_position_target_command(0.0, 0.0, 0.0, 0.0, mode="velocity")
            rate.sleep()  # 按照设定频率休眠

if __name__ == "__main__":
    try:
        node = PositionTargetCommandNode()  # 实例化PositionTargetCommandNode类
        node.run()  # 运行节点
    except rospy.ROSInterruptException:
        pass  # 捕获ROS中断异常
