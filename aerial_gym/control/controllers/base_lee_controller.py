import torch

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_lee_controller")

logger.setLevel("DEBUG")


import pytorch3d.transforms as p3d_transforms

from aerial_gym.control.controllers.base_controller import *


class BaseLeeController(BaseController):
    """
    该类作为所有控制器的基类，将被特定控制器类继承。
    """

    def __init__(self, control_config, num_envs, device, mode="robot"):
        super().__init__(control_config, num_envs, device, mode)
        self.cfg = control_config  # 控制器配置参数
        self.num_envs = num_envs  # 环境数量
        self.device = device  # 设备类型（CPU或GPU）

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)

        # 从配置中读取并设置控制器参数的值
        self.K_pos_tensor_max = torch.tensor(
            self.cfg.K_pos_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最大位置增益张量
        self.K_pos_tensor_min = torch.tensor(
            self.cfg.K_pos_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最小位置增益张量
        self.K_linvel_tensor_max = torch.tensor(
            self.cfg.K_vel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最大线速度增益张量
        self.K_linvel_tensor_min = torch.tensor(
            self.cfg.K_vel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最小线速度增益张量
        self.K_rot_tensor_max = torch.tensor(
            self.cfg.K_rot_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最大旋转增益张量
        self.K_rot_tensor_min = torch.tensor(
            self.cfg.K_rot_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最小旋转增益张量
        self.K_angvel_tensor_max = torch.tensor(
            self.cfg.K_angvel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最大角速度增益张量
        self.K_angvel_tensor_min = torch.tensor(
            self.cfg.K_angvel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)  # 最小角速度增益张量

        # 设置当前控制器参数的值
        self.K_pos_tensor_current = (self.K_pos_tensor_max + self.K_pos_tensor_min) / 2.0
        self.K_linvel_tensor_current = (self.K_linvel_tensor_max + self.K_linvel_tensor_min) / 2.0
        self.K_rot_tensor_current = (self.K_rot_tensor_max + self.K_rot_tensor_min) / 2.0
        self.K_angvel_tensor_current = (self.K_angvel_tensor_max + self.K_angvel_tensor_min) / 2.0

        # 定义后续控制器需要的张量
        self.accel = torch.zeros((self.num_envs, 3), device=self.device)  # 加速度张量
        self.wrench_command = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # [fx, fy, fz, tx, ty, tz] 力矩命令张量

        # 定义后续控制器需要的张量
        self.desired_quat = torch.zeros_like(self.robot_orientation)  # 期望四元数
        self.desired_body_angvel = torch.zeros_like(self.robot_body_angvel)  # 期望身体角速度
        self.euler_angle_rates = torch.zeros_like(self.robot_body_angvel)  # 欧拉角速率

        # 用于torch.jit函数的缓冲张量
        self.buffer_tensor = torch.zeros((self.num_envs, 3, 3), device=self.device)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)  # 调用更新方法

    def reset_commands(self):
        self.wrench_command[:] = 0.0  # 重置力矩命令为零

    def reset(self):
        self.reset_idx(env_ids=None)  # 重置环境索引

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.K_rot_tensor.shape[0])  # 获取所有环境的索引
        self.randomize_params(env_ids)  # 随机化参数

    def randomize_params(self, env_ids):
        if self.cfg.randomize_params == False:
            logger.debug(
                "根据配置设置，控制器参数的随机化已禁用。"
            )
            return
        # 随机化控制器参数
        self.K_pos_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_pos_tensor_min[env_ids], self.K_pos_tensor_max[env_ids]
        )
        self.K_linvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_linvel_tensor_min[env_ids], self.K_linvel_tensor_max[env_ids]
        )
        self.K_rot_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_rot_tensor_min[env_ids], self.K_rot_tensor_max[env_ids]
        )
        self.K_angvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_angvel_tensor_min[env_ids], self.K_angvel_tensor_max[env_ids]
        )

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        position_error_world_frame = setpoint_position - self.robot_position  # 计算世界坐标系下的位置误差
        logger.debug(
            f"position_error_world_frame: {position_error_world_frame}, setpoint_position: {setpoint_position}, robot_position: {self.robot_position}"
        )
        setpoint_velocity_world_frame = quat_rotate(
            self.robot_vehicle_orientation, setpoint_velocity
        )  # 将期望速度从局部坐标系转换到世界坐标系
        velocity_error = setpoint_velocity_world_frame - self.robot_linvel  # 计算速度误差

        # 根据位置和速度误差计算加速度命令
        accel_command = (
            self.K_pos_tensor_current * position_error_world_frame
            + self.K_linvel_tensor_current * velocity_error
        )
        return accel_command  # 返回加速度命令

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        setpoint_angvel[:, 2] = torch.clamp(
            setpoint_angvel[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
        )  # 限制偏航角速度
        RT_Rd_quat = quat_mul(quat_inverse(self.robot_orientation), setpoint_orientation)  # 计算目标姿态与当前姿态之间的四元数差
        RT_Rd = quat_to_rotation_matrix(RT_Rd_quat)  # 转换为旋转矩阵
        rotation_error = 0.5 * compute_vee_map(torch.transpose(RT_Rd, -2, -1) - RT_Rd)  # 计算旋转误差
        angvel_error = self.robot_body_angvel - quat_rotate(RT_Rd_quat, setpoint_angvel)  # 计算角速度误差
        feed_forward_body_rates = torch.cross(
            self.robot_body_angvel,
            torch.bmm(self.robot_inertia, self.robot_body_angvel.unsqueeze(2)).squeeze(2),
            dim=1,
        )  # 前馈体动力学效应
        torque = (
            -self.K_rot_tensor_current * rotation_error
            - self.K_angvel_tensor_current * angvel_error
            + feed_forward_body_rates
        )  # 计算最终的扭矩命令
        return torque  # 返回扭矩命令


@torch.jit.script # 将 Python 函数转换为 TorchScript 的装饰器。TorchScript 是 PyTorch 提供的一种中间表示，允许你将模型导出为一种可序列化和可优化的形式，从而使得模型可以在不依赖于 Python 解释器的环境中运行
def calculate_desired_orientation_from_forces_and_yaw(forces_command, yaw_setpoint):
    c_phi_s_theta = forces_command[:, 0]  # 提取力命令中的x分量
    s_phi = -forces_command[:, 1]  # 提取力命令中的y分量
    c_phi_c_theta = forces_command[:, 2]  # 提取力命令中的z分量

    # 计算期望姿态
    pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)  # 计算俯仰角
    roll_setpoint = torch.atan2(s_phi, torch.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))  # 计算滚转角
    quat_desired = quat_from_euler_xyz_tensor(
        torch.stack((roll_setpoint, pitch_setpoint, yaw_setpoint), dim=1)
    )  # 从欧拉角生成四元数
    return quat_desired  # 返回期望四元数


# @torch.jit.script
def calculate_desired_orientation_for_position_velocity_control(
    forces_command, yaw_setpoint, rotation_matrix_desired
):
    b3_c = torch.div(forces_command, torch.norm(forces_command, dim=1).unsqueeze(1))  # 归一化力向量
    temp_dir = torch.zeros_like(forces_command)  # 创建临时方向向量
    temp_dir[:, 0] = torch.cos(yaw_setpoint)  # x轴方向
    temp_dir[:, 1] = torch.sin(yaw_setpoint)  # y轴方向

    b2_c = torch.cross(b3_c, temp_dir, dim=1)  # 计算b2_c向量
    b2_c = torch.div(b2_c, torch.norm(b2_c, dim=1).unsqueeze(1))  # 归一化b2_c向量
    b1_c = torch.cross(b2_c, b3_c, dim=1)  # 计算b1_c向量

    rotation_matrix_desired[:, :, 0] = b1_c  # 设置期望旋转矩阵的第一列
    rotation_matrix_desired[:, :, 1] = b2_c  # 设置期望旋转矩阵的第二列
    rotation_matrix_desired[:, :, 2] = b3_c  # 设置期望旋转矩阵的第三列
    q = p3d_transforms.matrix_to_quaternion(rotation_matrix_desired)  # 从旋转矩阵转换为四元数
    quat_desired = torch.stack((q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=1)  # 重新排列四元数的顺序

    sign_qw = torch.sign(quat_desired[:, 3])  # 获取四元数的符号
    # quat_desired = quat_desired * sign_qw.unsqueeze(1)

    return quat_desired  # 返回期望四元数


# quat_from_rotation_matrix(rotation_matrix_desired)


@torch.jit.script
def euler_rates_to_body_rates(robot_euler_angles, desired_euler_rates, rotmat_euler_to_body_rates):
    s_pitch = torch.sin(robot_euler_angles[:, 1])  # 计算俯仰角的正弦值
    c_pitch = torch.cos(robot_euler_angles[:, 1])  # 计算俯仰角的余弦值

    s_roll = torch.sin(robot_euler_angles[:, 0])  # 计算滚转角的正弦值
    c_roll = torch.cos(robot_euler_angles[:, 0])  # 计算滚转角的余弦值

    rotmat_euler_to_body_rates[:, 0, 0] = 1.0  # 设置旋转矩阵的元素
    rotmat_euler_to_body_rates[:, 1, 1] = c_roll  # 设置旋转矩阵的元素
    rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch  # 设置旋转矩阵的元素
    rotmat_euler_to_body_rates[:, 2, 1] = -s_roll  # 设置旋转矩阵的元素
    rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch  # 设置旋转矩阵的元素
    rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch  # 设置旋转矩阵的元素

    return torch.bmm(rotmat_euler_to_body_rates, desired_euler_rates.unsqueeze(2)).squeeze(2)  # 返回身体角速率
