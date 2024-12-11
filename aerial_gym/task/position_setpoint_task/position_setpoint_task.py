from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    # 将字典转换为类
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # 如果用户提供了参数，则覆盖默认参数
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)  # 调用父类构造函数
        self.device = self.task_config.device
        
        # 将奖励参数的每个元素设置为torch张量
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        # 构建仿真环境
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        # 初始化动作和状态
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # 从环境中获取观察值字典，以便后续使用
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        # 定义观察空间和动作空间
        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.num_envs = self.sim_env.num_envs

        # 当前只将"observations"发送给actor和critic，"priviliged_obs"尚未处理
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        # 删除环境
        self.sim_env.delete_env()

    def reset(self):
        # 重置目标位置和信息
        self.target_position[:, 0:3] = 0.0  
        self.infos = {}
        self.sim_env.reset()  # 重置仿真环境
        return self.get_return_tuple()  # 返回初始状态

    def reset_idx(self, env_ids):
        # 根据环境ID重置特定环境
        self.target_position[:, 0:3] = 0.0 
        self.infos = {}
        self.sim_env.reset_idx(env_ids)  # 重置指定环境
        return

    def render(self):
        # 渲染环境（当前不实现）
        return None

    def step(self, actions):
        # 执行动作并更新状态
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        # 使用当前动作进行一步仿真，并计算奖励
        self.sim_env.step(actions=self.actions)

        # 在计算奖励之后执行重置操作以确保正确的状态返回
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        # 检查是否达到最大步数
        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()  # 后处理步骤

        self.infos = {}  # 清空信息

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple  # 返回当前状态、奖励等信息

    def get_return_tuple(self):
        # 获取任务的返回元组，包括观察、奖励、终止、截断和信息
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        # 处理任务的观察数据
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
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
            angular_velocity,
            root_quats,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # 指数衰减函数
    # 类型: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # 指数惩罚函数
    # 类型: (Tensor, float, float) -> Tensor
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
    # 类型: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)  # 计算位置误差的范数

    # 基于距离的奖励
    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 0.5, 1.0)

    dist_reward = (20 - dist) / 40.0  # 距离奖励归一化

    ups = quat_axis(robot_quats, 2)  # 提取机器人朝上的方向
    tiltage = torch.abs(1 - ups[..., 2])  # 计算倾斜度
    up_reward = 0.2 / (0.1 + tiltage * tiltage)  # 倾斜奖励

    spinnage = torch.norm(robot_angvels, dim=1)  # 计算角速度
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 10  # 角速度奖励

    previous_action_penalty = torch.sum(
        exp_penalty_func(current_action - prev_actions, 0.02, 10.0), dim=1
    )  # 上一个动作的惩罚

    absolute_action_penalty = torch.sum(exp_penalty_func(current_action, 0.01, 5.0), dim=1)  # 当前动作的绝对惩罚

    total_reward = (
        pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    )  # 总奖励计算
    total_reward[:] = curriculum_level_multiplier * total_reward  # 应用课程级别乘法器

    # 碰撞检测
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    # 如果发生碰撞，总奖励设为负值
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes  # 返回总奖励和碰撞信息
