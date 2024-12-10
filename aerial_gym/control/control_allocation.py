import torch
from aerial_gym.control.motor_model import MotorModel

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("control_allocation")


class ControlAllocator:
    def __init__(self, num_envs, dt, config, device):
        # 初始化控制分配器
        # num_envs: 环境数量
        # dt: 时间步长
        # config: 配置参数
        # device: 设备类型（CPU或GPU）
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.device = device
        self.force_application_level = self.cfg.force_application_level
        self.motor_directions = torch.tensor(self.cfg.motor_directions, device=self.device)
        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        )
        self.output_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        # 确保分配矩阵的行数为6，列数为电机数量
        assert (
            len(self.cfg.allocation_matrix[0]) == self.cfg.num_motors
        ), "Allocation matrix must have 6 rows and num_motors columns."

        self.force_torque_allocation_matrix = torch.tensor(
            self.cfg.allocation_matrix, device=self.device, dtype=torch.float32
        )
        alloc_matrix_rank = torch.linalg.matrix_rank(self.force_torque_allocation_matrix)
        if alloc_matrix_rank < 6:
            print("WARNING: allocation matrix is not full rank. Rank: {}".format(alloc_matrix_rank))
        # 扩展分配矩阵以适应多个环境
        self.force_torque_allocation_matrix = self.force_torque_allocation_matrix.expand(
            self.num_envs, -1, -1
        )
        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(
            torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)
        ).expand(self.num_envs, -1, -1)
        # 初始化电机模型
        self.motor_model = MotorModel(
            num_envs=self.num_envs,
            dt=self.dt,
            motors_per_robot=self.cfg.num_motors,
            config=self.cfg.motor_model_config,
            device=self.device,
        )
        logger.warning(
            f"Control allocation does not account for actuator limits. This leads to suboptimal allocation"
        )

    def allocate_output(self, command, output_mode):
        # 分配输出，计算电机的推力
        # command: 输入命令，通常是期望的力或扭矩
        # output_mode: 输出模式（"forces"或其他）
        if self.force_application_level == "motor_link":
            if output_mode == "forces":
                motor_thrusts = self.update_motor_thrusts_with_forces(command)
            else:
                motor_thrusts = self.update_motor_thrusts_with_wrench(command)
            forces, torques = self.calc_motor_forces_torques_from_thrusts(motor_thrusts)

        else:
            output_wrench = self.update_wrench(command)
            forces = output_wrench[:, 0:3].unsqueeze(1)
            torques = output_wrench[:, 3:6].unsqueeze(1)

        return forces, torques

    def update_wrench(self, ref_wrench):
        # 更新力矩（wrench），计算电机推力
        # ref_wrench: 参考力矩
        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        self.output_wrench[:] = torch.bmm(
            self.force_torque_allocation_matrix, current_motor_thrust.unsqueeze(-1)
        ).squeeze(-1)

        return self.output_wrench

    def update_motor_thrusts_with_forces(self, ref_forces):
        # 根据参考力更新电机推力
        # ref_forces: 参考力
        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_forces)
        return current_motor_thrust

    def update_motor_thrusts_with_wrench(self, ref_wrench):
        # 根据参考力矩更新电机推力
        # ref_wrench: 参考力矩
        ref_motor_thrusts = torch.bmm(
            self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)
        ).squeeze(-1)

        current_motor_thrust = self.motor_model.update_motor_thrusts(ref_motor_thrusts)

        return current_motor_thrust

    def reset_idx(self, env_ids):
        # 重置指定环境的索引
        self.motor_model.reset_idx(env_ids)
        # 此处可以根据需要随机化分配矩阵

    def reset(self):
        # 重置控制分配器
        self.motor_model.reset()
        # 此处可以根据需要随机化分配矩阵

    def calc_motor_forces_torques_from_thrusts(self, motor_thrusts):
        # 根据电机推力计算电机的力和扭矩
        # motor_thrusts: 电机推力
        motor_forces = torch.stack(
            [
                torch.zeros_like(motor_thrusts),
                torch.zeros_like(motor_thrusts),
                motor_thrusts,
            ],
            dim=2,
        )
        cq = self.cfg.motor_model_config.thrust_to_torque_ratio
        motor_torques = cq * motor_forces * (-self.motor_directions[None, :, None])
        return motor_forces, motor_torques

    def set_single_allocation_matrix(self, alloc_matrix):
        # 设置单个分配矩阵
        # alloc_matrix: 新的分配矩阵，形状应为(6, num_motors)
        if alloc_matrix.shape != (6, self.cfg.num_motors):
            raise ValueError("Allocation matrix must have shape (6, num_motors)")
        self.force_torque_allocation_matrix[:] = torch.tensor(
            alloc_matrix, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1, -1)
        self.inv_force_torque_allocation_matrix[:] = torch.linalg.pinv(
            self.force_torque_allocation_matrix
        ).expand(self.num_envs, -1, -1)
