import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *

from aerial_gym.control.controllers.base_lee_controller import *


class LeeAccelerationController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee姿态控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含期望的推力、滚转、俯仰和偏航速率命令（在飞行器坐标系中）
        :return: m*g归一化的推力和惯性归一化的扭矩
        """
        self.reset_commands()  # 重置命令

        # 从command_actions中提取加速度信息
        self.accel[:] = command_actions[:, 0:3]  
        
        # 计算作用于机器人的力
        forces = self.mass * (self.accel - self.gravity)
        
        # 推力命令通过机体朝向的z分量进行变换
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )

        # 在计算完力后，基于力和当前偏航角计算期望的欧拉角
        self.desired_quat[:] = calculate_desired_orientation_from_forces_and_yaw(
            forces, self.robot_euler_angles[:, 2]
        )

        # 初始化欧拉角速率，前两个维度设为0，第三个维度为命令中的偏航速率
        self.euler_angle_rates[:, :2] = 0.0  
        self.euler_angle_rates[:, 2] = command_actions[:, 3]  

        # 将期望的欧拉角速率转换为机体速率
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )
        
        # 计算所需的身体扭矩
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command  # 返回施加的扭矩和推力命令
