from aerial_gym.robots.base_robot import BaseRobot

from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.registry.controller_registry import controller_registry

import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_rov")


class BaseROV(BaseRobot):
    """
    该类是一个完全驱动的ROV（遥控水下机器人）机器人的基类。
    """

    def __init__(self, robot_config, controller_name, env_config, device):
        logger.debug("Initializing BaseROV")
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )
        logger.warning(f"创建 {self.num_envs} 个ROV。")
        self.force_application_level = self.cfg.control_allocator_config.force_application_level
        if controller_name == "no_control":
            self.output_mode = "forces"
        else:
            self.output_mode = "wrench"

        if self.force_application_level == "root_link" and controller_name == "no_control":
            raise ValueError(
                "力应用级别 'root_link' 不能与 'no_control' 一起使用。"
            )

        # 初始化张量
        self.robot_state = None
        self.robot_force_tensors = None
        self.robot_torque_tensors = None
        self.action_tensor = None
        self.max_init_state = None
        self.min_init_state = None
        self.max_force_and_torque_disturbance = None
        self.max_torque_disturbance = None
        self.controller_input = None
        self.control_allocator = None
        self.output_forces = None
        self.output_torques = None

        logger.debug("[DONE] 初始化 BaseROV")

    def init_tensors(self, global_tensor_dict):
        """
        初始化机器人的状态、力、扭矩和动作的张量。
        该函数调用中使用的张量作为环境中的主张量的切片发送。
        这些切片仅确定机器人的状态、力、扭矩和动作。
        为了避免访问机器人不需要的数据，未将完整张量传递给此函数。
        """
        super().init_tensors(global_tensor_dict)
        # 向全局张量字典添加更多张量
        self.robot_vehicle_orientation = torch.zeros_like(
            self.robot_orientation, requires_grad=False, device=self.device
        )
        self.robot_vehicle_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_angvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_body_linvel = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        self.robot_euler_angles = torch.zeros_like(
            self.robot_linvel, requires_grad=False, device=self.device
        )
        # 添加到张量字典
        global_tensor_dict["robot_vehicle_orientation"] = self.robot_vehicle_orientation
        global_tensor_dict["robot_vehicle_linvel"] = self.robot_vehicle_linvel
        global_tensor_dict["robot_body_angvel"] = self.robot_body_angvel
        global_tensor_dict["robot_body_linvel"] = self.robot_body_linvel
        global_tensor_dict["robot_euler_angles"] = self.robot_euler_angles

        global_tensor_dict["num_robot_actions"] = self.controller_config.num_actions

        self.controller.init_tensors(global_tensor_dict)
        self.action_tensor = torch.zeros(
            (self.num_envs, self.controller_config.num_actions), device=self.device
        )

        # 初始化机器人状态
        # [x, y, z, roll, pitch, yaw, 1.0 (用于保持形状), vx, vy, vz, wx, wy, wz]
        self.min_init_state = torch.tensor(
            self.cfg.init_config.min_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_init_state = torch.tensor(
            self.cfg.init_config.max_init_state, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # 扰动参数
        # [fx, fy, fz, tx, ty, tz]
        self.max_force_and_torque_disturbance = torch.tensor(
            self.cfg.disturbance.max_force_and_torque_disturbance,
            device=self.device,
            requires_grad=False,
        ).expand(self.num_envs, -1)

        # 控制器参数
        self.controller_input = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, requires_grad=False
        )
        self.control_allocator = ControlAllocator(
            num_envs=self.num_envs,
            dt=self.dt,
            config=self.cfg.control_allocator_config,
            device=self.device,
        )

        self.body_vel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.body_vel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.linvel_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_linear_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_linear_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )

        self.angvel_quadratic_damping_coefficient = torch.tensor(
            self.cfg.damping.angular_quadratic_damping_coefficient,
            device=self.device,
            requires_grad=False,
        )
        if self.force_application_level == "motor_link":
            self.application_mask = torch.tensor(
                self.cfg.control_allocator_config.application_mask,
                device=self.device,
                requires_grad=False,
            )
        else:
            self.application_mask = torch.tensor([0], device=self.device)

        self.motor_directions = torch.tensor(
            self.cfg.control_allocator_config.motor_directions,
            device=self.device,
            requires_grad=False,
        )

        self.output_forces = torch.zeros_like(
            global_tensor_dict["robot_force_tensor"], device=self.device, requires_grad=False
        )
        self.output_torques = torch.zeros_like(
            global_tensor_dict["robot_torque_tensor"], device=self.device, requires_grad=False
        )

    def reset(self):
        """
        重置所有环境的状态。
        """
        self.reset_idx(torch.arange(self.num_envs))

    def reset_idx(self, env_ids):
        """
        根据给定的环境ID重置特定环境的状态。
        """
        if len(env_ids) == 0:
            return
        # robot_state 被定义为形状为 (num_envs, 13) 的张量
        # init_state 张量的格式为 [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0 (用于保持形状), vx, vy, vz, wx, wy, wz]
        random_state = torch_rand_float_tensor(self.min_init_state, self.max_init_state)

        self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
            self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
        )[env_ids]

        logger.debug(
            f"随机状态: {random_state[0]}, 最小初始化状态: {self.min_init_state[0]}, 最大初始化状态: {self.max_init_state[0]}"
        )
        logger.debug(
            f"环境边界最小值: {self.env_bounds_min[0]}, 环境边界最大值: {self.env_bounds_max[0]}"
        )

        # 四元数转换单独处理
        self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])

        self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
        self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]

        self.controller.randomize_params(env_ids=env_ids)

        # 在重置后更新状态，因为RL代理在重置后获取第一个状态
        self.update_states()

    def clip_actions(self):
        """
        将动作张量裁剪到控制器输入的范围内。
        """
        self.action_tensor[:] = torch.clamp(self.action_tensor, -10.0, 10.0)

    def apply_disturbance(self):
        """
        应用扰动到机器人状态。
        """
        if not self.cfg.disturbance.enable_disturbance:
            return
        disturbance_occurence = torch.bernoulli(
            self.cfg.disturbance.prob_apply_disturbance
            * torch.ones((self.num_envs), device=self.device)
        )
        logger.debug(
            f"对 {disturbance_occurence.sum().item()} 个环境应用扰动"
        )
        logger.debug(
            f"扰动张量的形状: {self.robot_force_tensors.shape}, {self.robot_torque_tensors.shape}"
        )
        logger.debug(f"扰动形状: {disturbance_occurence.unsqueeze(1).shape}")
        self.robot_force_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 0:3],
            self.max_force_and_torque_disturbance[:, 0:3],
        ) * disturbance_occurence.unsqueeze(1)
        self.robot_torque_tensors[:, 0, 0:3] += torch_rand_float_tensor(
            -self.max_force_and_torque_disturbance[:, 3:6],
            self.max_force_and_torque_disturbance[:, 3:6],
        ) * disturbance_occurence.unsqueeze(1)

    def control_allocation(self, command_wrench, output_mode):
        """
        将推力和扭矩命令分配给电机。电机模型也用于更新电机推力。
        """
        forces, torques = self.control_allocator.allocate_output(command_wrench, output_mode)

        self.output_forces[:, self.application_mask, :] = forces
        self.output_torques[:, self.application_mask, :] = torques

    def call_controller(self):
        """
        将动作张量转换为控制器输入。动作张量是输入，可以根据用户的需求进行参数化。
        此函数的目的是将动作张量转换为控制器输入。
        """
        self.clip_actions()
        controller_output = self.controller(self.action_tensor)
        self.control_allocation(controller_output, self.output_mode)

        self.robot_force_tensors[:] = self.output_forces
        self.robot_torque_tensors[:] = self.output_torques

    def update_states(self):
        """
        更新机器人的状态信息，包括欧拉角、车辆方向、线速度和角速度。
        """
        self.robot_euler_angles[:] = get_euler_xyz_tensor(self.robot_orientation)
        self.robot_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.robot_orientation)
        self.robot_vehicle_linvel[:] = quat_rotate_inverse(
            self.robot_vehicle_orientation, self.robot_linvel
        )
        self.robot_body_linvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_linvel)
        self.robot_body_angvel[:] = quat_rotate_inverse(self.robot_orientation, self.robot_angvel)

    def simulate_drag(self):
        """
        模拟阻力并更新力和扭矩张量。
        """
        self.robot_body_vel_drag_linear = (
            -self.body_vel_linear_damping_coefficient * self.robot_body_linvel
        )
        self.robot_body_vel_drag_quadratic = (
            -self.body_vel_quadratic_damping_coefficient
            * self.robot_body_linvel.abs()
            * self.robot_body_linvel
        )
        self.robot_body_vel_drag = (
            self.robot_body_vel_drag_linear + self.robot_body_vel_drag_quadratic
        )
        self.robot_force_tensors[:, 0, 0:3] += self.robot_body_vel_drag

        self.robot_body_angvel_drag_linear = (
            -self.angvel_linear_damping_coefficient * self.robot_body_angvel
        )
        self.robot_body_angvel_drag_quadratic = (
            -self.angvel_quadratic_damping_coefficient
            * self.robot_body_angvel.abs()
            * self.robot_body_angvel
        )
        self.robot_body_angvel_drag = (
            self.robot_body_angvel_drag_linear + self.robot_body_angvel_drag_quadratic
        )
        self.robot_torque_tensors[:, 0, 0:3] += self.robot_body_angvel_drag

    def step(self, action_tensor):
        """
        更新四旋翼的状态。该函数在每个仿真步骤中被调用。
        """
        self.update_states()
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("动作张量的环境数量不正确")
        self.action_tensor[:] = action_tensor
        # 调用控制器导致控制分配的发生
        self.call_controller()
        self.simulate_drag()
        self.apply_disturbance()
