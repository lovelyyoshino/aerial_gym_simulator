from aerial_gym.task.position_setpoint_task.position_setpoint_task import PositionSetpointTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    # 将字典转换为类，动态创建一个新类并返回
    # 参数:
    # dict: 一个包含属性和对应值的字典，用于初始化新类的属性

    return type("ClassFromDict", (object,), dict)
    # 使用type函数动态创建一个名为"ClassFromDict"的新类，
    # 该类继承自object，并将传入的字典作为类的属性。


class PositionSetpointTaskReconfigurable(PositionSetpointTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # 初始化位置设定任务，继承自PositionSetpointTask
        super().__init__(
            task_config=task_config,
            seed=seed,
            num_envs=num_envs,
            headless=headless,
            device=device,
            use_warp=use_warp,
        )

        # 设置动作限制的最小值和最大值
        self.action_limit_min = torch.tensor(
            task_config.action_limit_min, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self.action_limit_max = torch.tensor(
            task_config.action_limit_max, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

    def step(self, actions):
        # 执行一步操作，更新状态和奖励
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = self.task_config.process_actions_for_task(
            actions, self.action_limit_min, self.action_limit_max
        )

        # 在执行步骤之前设置动作
        self.sim_env.robot_manager.robot.set_dof_velocity_targets(
            self.actions[:, self.task_config.num_motors :]
        )
        self.sim_env.step(actions=self.actions[:, : self.task_config.num_motors])

        # 计算奖励和终止条件
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        # 检查是否达到最大步数以进行截断
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # 清空信息字典

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def process_obs_for_task(self):
        # 处理观察数据，将其格式化为任务所需的形式
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13 : 13 + self.task_config.action_space_dim] = self.actions
        self.task_obs["observations"][
            :,
            13
            + self.task_config.action_space_dim : 13
            + self.task_config.action_space_dim
            + self.task_config.num_joints,
        ] = self.obs_dict["dof_state_tensor"][..., 0].reshape(-1, self.task_config.num_joints)
        self.task_obs["observations"][
            :, 13 + self.task_config.action_space_dim + self.task_config.num_joints :
        ] = self.obs_dict["dof_state_tensor"][..., 1].reshape(-1, self.task_config.num_joints)

        # 打印观察张量中的NAN值位置
        if torch.isnan(self.task_obs["observations"]).any():
            logger.info(
                "NAN values in the observation tensor: ",
                torch.isnan(self.task_obs["observations"]).nonzero(),
            )

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        # 计算奖励和碰撞情况
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        # 计算车辆框架下的位置误差
        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            pos_error_vehicle_frame,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # 指数衰减函数，用于计算奖励
    # 输入：x - 输入张量，gain - 增益因子，exp - 衰减因子
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # 指数惩罚函数，用于计算惩罚
    # 输入：x - 输入张量，gain - 增益因子，exp - 衰减因子
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def compute_reward(
    pos_error,
    robot_quats,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):
    # 计算总奖励
    # 输入：
    # pos_error - 位置误差张量
    # robot_quats - 机器人四元数
    # robot_angvels - 机器人角速度
    # crashes - 碰撞标志
    # curriculum_level_multiplier - 课程级别乘数
    # current_action - 当前动作
    # prev_actions - 前一动作
    # parameter_dict - 奖励参数字典
    # 输出：总奖励和碰撞标志
    dist = torch.norm(pos_error, dim=1)  # 计算位置误差的范数
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 0.5, 1.0)  # 位置奖励
    dist_reward = (20 - dist) / 40.0  # 距离奖励

    roll, pitch, yaw = get_euler_xyz(robot_quats)  # 获取欧拉角
    roll = ssa(roll)  # 限制范围
    pitch = ssa(pitch)  # 限制范围
    up_reward = exp_func(roll, 3.0, 5.0) + exp_func(pitch, 3.0, 5.0)  # 上升奖励

    spinnage = torch.norm(robot_angvels, dim=1)  # 计算角速度的范数
    ang_vel_reward = exp_func(spinnage, 3.0, 10.5)  # 角速度奖励
    yaw_rate_special = exp_func(torch.abs(robot_angvels[:, 2]), 5.0, 20.5)  # 偏航率奖励

    total_reward = (
        pos_reward
        + dist_reward
        + yaw_rate_special
        + pos_reward * (up_reward + ang_vel_reward + yaw_rate_special)
    )  # 总奖励计算
    total_reward[:] = curriculum_level_multiplier * total_reward  # 应用课程级别乘数

    # 更新碰撞标志
    crashes[:] = torch.where(dist > 3.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(torch.abs(roll) > 1.0, torch.ones_like(crashes), crashes)
    crashes[:] = torch.where(torch.abs(pitch) > 1.0, torch.ones_like(crashes), crashes)

    # 如果发生碰撞，总奖励设为负值
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes  # 返回总奖励和碰撞标志
