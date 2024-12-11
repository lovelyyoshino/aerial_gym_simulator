import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from gymnasium import spaces

from torch import nn


class NN_Inference_Class(nn.Module):
    def __init__(self, num_envs, num_actions, num_obs, cfg: Config) -> None:
        super().__init__()
        self.cfg = load_from_checkpoint(cfg)  # 从检查点加载配置
        self.cfg.num_envs = num_envs  # 设置环境数量
        self.num_actions = num_actions  # 动作数量
        self.num_obs = num_obs  # 观察值数量
        self.num_agents = num_envs  # 代理数量等于环境数量
        
        # 定义观察空间和动作空间
        self.observation_space = spaces.Dict(
            dict(
                obs=convert_space(
                    spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
                )
            )
        )
        self.action_space = convert_space(
            spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        )
        
        self.init_env_info()  # 初始化环境信息
        self.actor_critic = create_actor_critic(self.cfg, self.observation_space, self.action_space)  # 创建actor-critic模型
        self.actor_critic.eval()  # 将模型设置为评估模式
        self.device = torch.device("cpu" if self.cfg.device == "cpu" else "cuda")  # 根据配置选择设备
        self.actor_critic.model_to_device(self.device)  # 将模型移动到指定设备
        print("Model:\n\n", self.actor_critic)  # 打印模型结构
        
        # 加载策略到模型中
        policy_id = self.cfg.policy_index  # 获取策略索引
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]  # 确定检查点名称前缀
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*"
        )  # 获取所有检查点
        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)  # 加载最新的检查点
        self.actor_critic.load_state_dict(checkpoint_dict["model"])  # 将模型参数加载到actor-critic中
        
        # 初始化RNN状态
        self.rnn_states = torch.zeros(
            [self.num_agents, get_rnn_size(self.cfg)],
            dtype=torch.float32,
            device=self.device,
        )

    def init_env_info(self):
        # 初始化环境信息，包括观察空间、动作空间、代理数量等
        self.env_info = EnvInfo(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
            gpu_actions=self.cfg.env_gpu_actions,
            gpu_observations=self.cfg.env_gpu_observations,
            action_splits=None,
            all_discrete=None,
            frameskip=self.cfg.env_frameskip,
        )

    def reset(self, env_ids):
        # 重置指定环境的RNN状态
        self.rnn_states[env_ids] = 0.0

    def get_action(self, obs, get_np=False, get_robot_zero=False):
        with torch.no_grad():  # 禁用梯度计算以节省内存
            # 将观察值处理并归一化
            processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)  # 获取策略输出
            
            # 默认从分布中采样动作
            actions = policy_outputs["actions"]
            if self.cfg.eval_deterministic:  # 如果是确定性评估，使用最大概率的动作
                action_distribution = self.actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # 确保动作形状为[num_agents, num_actions]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)  # 对动作进行预处理

            self.rnn_states = policy_outputs["new_rnn_states"]  # 更新RNN状态
            
        if get_robot_zero:
            actions = actions[0]  # 如果需要机器人零状态，则返回第一个动作
        if get_np:
            return actions.cpu().numpy()  # 返回NumPy格式的动作
        return actions  # 返回动作
