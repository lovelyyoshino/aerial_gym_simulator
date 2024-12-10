import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import *


class FullyActuatedController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        完全驱动控制器。输入为期望的位置和方向。
        command_actions = [p_x, p_y, p_z, qx, qy, qz, qw]
        位置设定点在世界坐标系中
        方向参考相对于世界坐标系
        """
        self.reset_commands()  # 重置命令

        # 归一化四元数，确保其为单位四元数
        command_actions[:, 3:7] = normalize(command_actions[:, 3:7])

        # 计算加速度，目标位置为command_actions的前3个元素，目标速度为零
        self.accel[:] = self.compute_acceleration(
            command_actions[:, 0:3], torch.zeros_like(command_actions[:, 0:3])
        )

        # 计算施加的力，使用质量和重力
        forces = self.mass * (self.accel - self.gravity)

        # 将施加的力从机器人坐标系转换到世界坐标系
        self.wrench_command[:, 0:3] = quat_rotate_inverse(self.robot_orientation, forces)

        # 设置期望的方向四元数
        self.desired_quat[:] = command_actions[:, 3:]

        # 计算施加的扭矩，目标方向为desired_quat，目标角速度为零
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, torch.zeros_like(command_actions[:, 0:3])
        )

        return self.wrench_command  # 返回施加的力和扭矩命令
