#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from mavros_msgs.msg import PositionTarget

from rl_games_inference import MLP
import torch
import time

# WEIGHTS_PATH = "gen_ppo.pth"
# WEIGHTS_PATH = "vel_control_lmf2.pth"

COMMAND_MODE = "acceleration"  # 控制模式，"velocity" 或 "acceleration"

if COMMAND_MODE == "velocity":
    WEIGHTS_PATH = "networks/vel_control_lmf2_direct.pth"  # 速度控制模型权重路径
    CLIP_VALUE = 1.0  # 动作裁剪值
    VELOCITY_ACTION_MAGNITUDE = 1.0  # 速度动作幅度
    YAW_RATE_ACTION_MAGNITUDE = 1.0  # 偏航率动作幅度

elif COMMAND_MODE == "acceleration":
    WEIGHTS_PATH = "networks/acc_command_2_multiplier_disturbance.pth"  # 加速度控制模型权重路径
    # WEIGHTS_PATH = "acc_control_lmf2_direct.pth"
    CLIP_VALUE = 1.0  # 动作裁剪值
    VELOCITY_ACTION_MAGNITUDE = 1.5  # 速度动作幅度
    YAW_RATE_ACTION_MAGNITUDE = 0.8  # 偏航率动作幅度


ROBOT_BASE_LINK_ID = "base_link"  # 机器人基座链接ID


ODOMETRY_TOPIC = "/mavros/local_position/odom"  # 里程计话题
GOAL_TOPIC = "/target_position"  # 目标位置话题
COMMAND_TOPIC = "/mavros/setpoint_raw/local"  # 命令话题
COMMAND_TOPIC_VIZ = "/mavros/setpoint_raw/local_viz"  # 可视化命令话题

STATE_TENSOR_BUFFER_DEVICE = "cpu"  # 状态张量缓冲设备，一般为CPU
NN_INFERENCE_DEVICE = "cpu"  # 神经网络推理设备


class RobotPositionControlNode:
    def __init__(self):
        rospy.init_node("robot_position_control_node")  # 初始化ROS节点

        # 参数设置
        self.update_rate = rospy.get_param("~update_rate", 50)  # 更新频率 (Hz)
        self.max_velocity = rospy.get_param("~max_velocity", 1.0)  # 最大速度 (m/s)
        self.max_acceleration = rospy.get_param("~max_acceleration", 0.5)  # 最大加速度 (m/s^2)

        # 状态变量初始化
        self.current_position = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )  # 当前位置信息
        self.current_orientation = torch.zeros(
            4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )  # 当前方向信息（四元数）
        self.current_orientation[3] = 1.0  # 四元数的w分量初始化为1
        self.current_body_velocity = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )  # 当前身体线速度
        self.current_body_angular_velocity = torch.zeros(
            3, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )  # 当前身体角速度
        self.current_state = torch.zeros(13, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False)  # 当前状态向量
        self.target_position = None  # 目标位置
        self.weights_path = WEIGHTS_PATH  # 模型权重路径
        self.actions = torch.zeros(4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False)  # 动作输出
        self.previous_actions = torch.zeros(
            4, device=STATE_TENSOR_BUFFER_DEVICE, requires_grad=False
        )  # 上一轮动作输出

        self.obs_tensor = torch.zeros(1, 17, device=NN_INFERENCE_DEVICE, requires_grad=False)  # 观察张量

        # 创建神经网络控制器
        self.controller = (
            MLP(input_dim=17, output_dim=4, path=self.weights_path).to(NN_INFERENCE_DEVICE).eval()
        )

        # 发布者
        self.cmd_pub = rospy.Publisher(COMMAND_TOPIC, PositionTarget, queue_size=1)  # 发布命令
        self.cmd_pub_viz = rospy.Publisher(COMMAND_TOPIC_VIZ, TwistStamped, queue_size=1)  # 发布可视化命令

        # 订阅者
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback, queue_size=1)  # 订阅里程计数据
        rospy.Subscriber(GOAL_TOPIC, PoseStamped, self.goal_callback, queue_size=1)  # 订阅目标位置

    def odom_callback(self, msg):
        """处理里程计回调函数"""
        msgpose = msg.pose.pose  # 获取位姿信息
        msgtwist = msg.twist.twist  # 获取速度信息
        # 更新当前位置信息和速度
        self.current_position[0] = msgpose.position.x
        self.current_position[1] = msgpose.position.y
        self.current_position[2] = msgpose.position.z

        self.current_body_velocity[0] = msgtwist.linear.x
        self.current_body_velocity[1] = msgtwist.linear.y
        self.current_body_velocity[2] = msgtwist.linear.z

        quat_sign = 1.0 if msgpose.orientation.w >= 0 else -1.0  # 确定四元数符号
        self.current_orientation[0] = msgpose.orientation.x
        self.current_orientation[1] = msgpose.orientation.y
        self.current_orientation[2] = msgpose.orientation.z
        self.current_orientation[3] = msgpose.orientation.w

        self.current_orientation[:] = quat_sign * self.current_orientation  # 调整四元数符号

        self.current_body_angular_velocity[0] = msgtwist.angular.x
        self.current_body_angular_velocity[1] = msgtwist.angular.y
        self.current_body_angular_velocity[2] = msgtwist.angular.z

        # 更新当前状态
        self.current_state[:] = torch.concatenate(
            [
                self.current_position,
                self.current_orientation,
                self.current_body_velocity,
                self.current_body_angular_velocity,
            ]
        )

    def goal_callback(self, msg):
        """处理目标位置回调函数"""
        # 更新目标位置
        self.target_position = torch.tensor(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            device=STATE_TENSOR_BUFFER_DEVICE,
            requires_grad=False,
        )

    def send_position_target_command(
        self, x_command, y_command, z_command, yaw_rate_command, mode="velocity"
    ):
        """发送位置目标命令"""
        msg = PositionTarget()  # 创建位置目标消息
        msg.header.stamp = rospy.Time.now()  # 设置时间戳
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED  # 坐标系设定为机体坐标系

        # 忽略位置并使用速度
        msg.type_mask = (
            PositionTarget.IGNORE_PX
            + PositionTarget.IGNORE_PY
            + PositionTarget.IGNORE_PZ
            + PositionTarget.IGNORE_YAW
        )
        if mode == "velocity":  # 如果是速度模式
            msg.type_mask += (
                PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
            )
        elif mode == "acceleration":  # 如果是加速度模式
            msg.type_mask += (
                PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
            )

        if mode == "velocity":  # 设置速度
            msg.velocity.x = x_command
            msg.velocity.y = y_command
            msg.velocity.z = z_command
        elif mode == "acceleration":  # 设置加速度
            msg.acceleration_or_force.x = x_command
            msg.acceleration_or_force.y = y_command
            msg.acceleration_or_force.z = z_command

        # 设置偏航率
        msg.yaw_rate = yaw_rate_command
        self.cmd_pub.publish(msg)  # 发布命令

        viz_msg = TwistStamped()  # 创建可视化消息
        viz_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        viz_msg.header.frame_id = ROBOT_BASE_LINK_ID  # 设置帧ID
        viz_msg.twist.linear.x = x_command
        viz_msg.twist.linear.y = y_command
        viz_msg.twist.linear.z = z_command
        viz_msg.twist.angular.z = yaw_rate_command
        self.cmd_pub_viz.publish(viz_msg)  # 发布可视化命令

    def get_observations_tensor(self, current_state, previous_actions, target_position):
        """获取观察张量"""
        self.obs_tensor[0, 0:3] = (target_position - current_state[:3]).to(NN_INFERENCE_DEVICE)  # 计算目标与当前位置的差
        self.obs_tensor[0, 3:7] = current_state[3:7].to(NN_INFERENCE_DEVICE)  # 当前方向
        self.obs_tensor[0, 7:10] = current_state[7:10].to(NN_INFERENCE_DEVICE)  # 当前线速度
        self.obs_tensor[0, 10:13] = current_state[10:13].to(NN_INFERENCE_DEVICE)  # 当前角速度
        self.obs_tensor[0, 13:17] = previous_actions.to(NN_INFERENCE_DEVICE)  # 上一轮动作
        return self.obs_tensor  # 返回观察张量

    def compute_command(self):
        """计算控制命令"""
        if self.target_position is None:  # 如果没有目标位置，则返回None
            return None
        obs_tensor = self.get_observations_tensor(
            self.current_state, self.previous_actions, self.target_position
        )  # 获取观察张量
        actions = self.controller(obs_tensor)  # 使用控制器计算动作
        return actions  # 返回计算出的动作

    def filter_actions(self, actions):
        """过滤动作以确保在允许范围内"""
        clipped_actions = torch.clip(actions, -CLIP_VALUE, CLIP_VALUE)  # 裁剪动作到[-CLIP_VALUE, CLIP_VALUE]
        clipped_actions[0] *= VELOCITY_ACTION_MAGNITUDE  # 对x轴速度进行缩放
        clipped_actions[1] *= VELOCITY_ACTION_MAGNITUDE  # 对y轴速度进行缩放
        clipped_actions[2] *= VELOCITY_ACTION_MAGNITUDE  # 对z轴速度进行缩放
        clipped_actions[3] *= YAW_RATE_ACTION_MAGNITUDE  # 对偏航率进行缩放
        return clipped_actions  # 返回过滤后的动作

    def run(self):
        """主循环运行函数"""
        rate = rospy.Rate(self.update_rate)  # 设置更新速率

        while not rospy.is_shutdown():  # 当ROS未关闭时持续运行
            try:
                start_time = time.time()  # 记录开始时间
                command = self.compute_command()  # 计算控制命令
                if command is None:  # 如果没有命令则发送零命令
                    self.send_position_target_command(0, 0, 0, 0, mode="velocity")
                else:
                    self.actions[:] = command  # 将计算得到的命令赋值给actions
                    end_time = time.time()  # 记录结束时间
                    print(f"Control loop time: {end_time - start_time}")  # 打印控制循环时间
                    self.previous_actions[:] = self.actions  # 保存上一次的动作
                    filtered_actions = self.filter_actions(self.actions).to(
                        STATE_TENSOR_BUFFER_DEVICE
                    )  # 过滤动作并转移到指定设备
                    self.send_position_target_command(
                        filtered_actions[0],
                        filtered_actions[1],
                        filtered_actions[2],
                        filtered_actions[3],
                        mode=COMMAND_MODE,
                    )  # 发送目标命令
            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")  # 捕获异常并打印错误信息
            rate.sleep()  # 按照设定频率休眠


if __name__ == "__main__":
    try:
        node = RobotPositionControlNode()  # 实例化控制节点
        node.run()  # 运行控制节点
    except rospy.ROSInterruptException:
        pass  # 捕获ROS中断异常
