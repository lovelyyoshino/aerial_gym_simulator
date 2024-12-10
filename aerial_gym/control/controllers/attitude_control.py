import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *

from aerial_gym.control.controllers.base_lee_controller import *


class LeeAttitudeController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        # 初始化张量，调用父类的初始化方法
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee姿态控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含期望的推力、滚转角、俯仰角和偏航速率命令（在飞行器坐标系中）
        :return: m*g归一化的推力和惯性归一化的扭矩
        """
        self.reset_commands()  # 重置命令

        # 计算并设置z轴方向的推力
        self.wrench_command[:, 2] = (
            (command_actions[:, 0] + 1.0) * self.mass.squeeze(1) * torch.norm(self.gravity, dim=1)
        )

        # 设置欧拉角速度，前两个分量为0，第三个分量为命令的偏航速率
        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]

        # 将期望的身体角速度转换为机器人坐标系下的角速度
        self.desired_body_angvel[:] = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor
        )

        # 计算期望的四元数
        # 期望的欧拉角等于命令的滚转角、俯仰角和当前的偏航角
        quat_desired = quat_from_euler_xyz(
            command_actions[:, 1], command_actions[:, 2], self.robot_euler_angles[:, 2]
        )
        
        # 计算身体扭矩，并将其存储到wrench_command中
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            quat_desired, self.desired_body_angvel
        )

        return self.wrench_command  # 返回计算得到的推力和扭矩
