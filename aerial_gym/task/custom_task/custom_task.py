from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("custom_task")


def dict_to_class(dict):
    """将字典转换为类对象"""
    return type("ClassFromDict", (object,), dict)


class CustomTask(BaseTask):
    def __init__(self, task_config):
        """初始化自定义任务类
        参数:
            task_config: 任务的配置对象，包含有关仿真和环境的信息
        """
        super().__init__(task_config)  # 调用基类的初始化方法
        self.device = self.task_config.device  # 获取设备信息

        # 构建仿真环境
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            device=self.device,
            args=self.task_config.args,
        )

        # 初始化任务观察
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
        """关闭仿真环境"""
        self.sim_env.delete_env()  # 删除环境

    def reset(self):
        """重置环境状态
        返回:
            None: 这里可以返回任何与任务相关的初始状态
        """
        return None  # 这里可以实现具体的重置逻辑

    def reset_idx(self, env_ids):
        """根据环境ID重置特定环境
        参数:
            env_ids: 需要重置的环境的ID列表
        """
        # 这里可以实现具体的重置逻辑
        return

    def render(self):
        """渲染当前环境状态
        返回:
            渲染结果: 返回仿真环境的渲染输出
        """
        return self.sim_env.render()  # 调用仿真环境的渲染方法

    def step(self, actions):
        """执行一步仿真并返回观察、奖励等信息
        参数:
            actions: 当前步骤的动作
        返回:
            None: 这里可以返回与任务相关的输出
        """
        # 使用动作，获取观察结果
        # 计算奖励，返回元组
        # 在这种情况下，需要首先重置终止的回合，并返回新回合的第一个观察
        self.sim_env.step(actions=actions)  # 执行动作并更新环境状态

        return None  # 这里可以实现具体的返回逻辑


@torch.jit.script
def compute_reward(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    """计算奖励函数
    参数:
        pos_error: 位置误差
        crashes: 碰撞信息
        action: 当前动作
        prev_action: 上一动作
        curriculum_level_multiplier: 课程级别的乘数
        parameter_dict: 一些参数字典
    返回:
        float: 计算得到的奖励值
    """
    # 这里可以实现具体的奖励计算逻辑
    return 0  # 返回计算的奖励值，这里为示例返回0
