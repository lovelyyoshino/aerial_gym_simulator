import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import *


class LeeVelocitySteeringAngleController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        # 初始化一个与机器人的角速度相同形状的零张量，用于存储欧拉角速率
        self.euler_angle_rates = torch.zeros_like(self.robot_body_angvel)

    def update(self, command_actions):
        """
        Lee姿态控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含期望的推力、滚转、俯仰和偏航率命令（在车辆坐标系中）
        :return: m*g归一化的推力和惯性归一化的扭矩
        """
        self.reset_commands()  # 重置命令
        # 计算期望的加速度
        self.accel[:] = self.compute_acceleration(
            setpoint_position=self.robot_position,
            setpoint_velocity=command_actions[:, 0:3],
        )
        # 计算力，考虑重力的影响
        forces = (self.accel[:] - self.gravity) * self.mass
        # 推力命令通过机体朝向的z分量进行转换
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )

        # 在计算力之后，计算期望的欧拉角
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[:, 3], self.buffer_tensor
        )
        self.euler_angle_rates[:] = 0.0  # 初始化欧拉角速率为0
        # 将期望的欧拉角速率转换为机体坐标系下的角速度
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )
        # 计算所需的机体扭矩
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command  # 返回推力和扭矩命令
