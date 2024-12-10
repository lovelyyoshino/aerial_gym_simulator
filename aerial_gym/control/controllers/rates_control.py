import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *

from aerial_gym.control.controllers.base_lee_controller import *

class LeeRatesController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee姿态控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含在载具坐标系下的期望推力、滚转、俯仰和偏航率命令
        :return: m*g归一化推力和惯性归一化的扭矩
        """
        self.reset_commands()  # 重置命令
        # 计算期望的四元数
        self.wrench_command[:, 2] = (command_actions[:, 0] - self.gravity) * self.mass  # 计算归一化的推力
        # 计算身体扭矩
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.robot_orientation, command_actions[:, 1:4]
        )

        return self.wrench_command  # 返回计算得到的力和扭矩命令
