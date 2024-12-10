import torch
from aerial_gym.utils.math import *


from aerial_gym.control.controllers.base_lee_controller import *
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("velocity_controller")


class LeeVelocityController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)  # 调用父类的初始化方法

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)  # 初始化张量，调用父类方法

    def update(self, command_actions):
        """
        Lee姿态控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含在飞行器坐标系下的期望推力、滚转、俯仰和偏航率命令
        :return: 归一化的推力(m*g)和归一化的惯性力矩
        """
        self.reset_commands()  # 重置命令

        # 计算所需的加速度
        self.accel[:] = self.compute_acceleration(
            setpoint_position=self.robot_position,  # 设定位置
            setpoint_velocity=command_actions[:, 0:3],  # 设定速度
        )

        # 计算所需的力
        forces = (self.accel[:] - self.gravity) * self.mass  # 计算力 = (加速度 - 重力) * 质量
        
        # 通过机体朝向的z分量转换推力命令
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )

        # 在计算力之后，计算期望的欧拉角
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, self.robot_euler_angles[:, 2], self.buffer_tensor  # 计算期望的四元数
        )

        self.euler_angle_rates[:, :2] = 0.0  # 将前两个欧拉角速度设为0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]  # 设置偏航角速度

        # 将期望的欧拉角速度转换为机体角速度
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )

        # 计算所需的力矩
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel  # 计算机体力矩
        )

        return self.wrench_command  # 返回计算得到的力矩命令
