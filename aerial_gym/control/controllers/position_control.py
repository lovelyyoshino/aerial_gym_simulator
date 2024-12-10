import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("lee_position_controller")

from aerial_gym.control.controllers.base_lee_controller import *

class LeePositionController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        # 初始化函数，调用父类的初始化方法
        super().__init__(config, num_envs, device)

    def init_tensors(self, global_tensor_dict=None):
        # 初始化张量，调用父类的初始化方法
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        Lee位置控制器
        :param command_actions: 形状为(num_envs, 4)的张量，包含期望的推力、滚转、俯仰和偏航率指令，车辆坐标系下
        :return: m*g归一化的推力和惯性归一化的扭矩
        """
        self.reset_commands()  # 重置命令

        # 计算期望加速度，设置目标位置为command_actions的前3个分量，目标速度为零
        self.accel[:] = self.compute_acceleration(
            setpoint_position=command_actions[:, 0:3],
            setpoint_velocity=torch.zeros_like(self.robot_vehicle_linvel),
        )
        logger.debug(f"accel: {self.accel}, command_actions: {command_actions}")  # 记录加速度和命令

        # 计算所需的力，考虑重力
        forces = (self.accel - self.gravity) * self.mass
        # 根据机器人当前朝向的z轴分量变换推力指令
        self.wrench_command[:, 2] = torch.sum(
            forces * quat_to_rotation_matrix(self.robot_orientation)[:, :, 2], dim=1
        )

        # 计算所需的欧拉角
        self.desired_quat[:] = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[:, 3], self.buffer_tensor
        )

        self.euler_angle_rates[:] = 0.0  # 初始化欧拉角速率为零
        self.desired_body_angvel[:] = 0.0  # 初始化期望的机体角速度为零

        # 计算所需的身体扭矩
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command  # 返回推力和扭矩指令
