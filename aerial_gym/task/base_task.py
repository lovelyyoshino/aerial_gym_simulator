from abc import ABC, abstractmethod

import time, torch, os, numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_task")


class BaseTask(ABC):
    def __init__(self, task_config):
        # 初始化BaseTask类，接收任务配置参数
        self.task_config = task_config  # 任务配置
        self.action_space = None  # 动作空间
        self.observation_space = None  # 观察空间
        self.reward_range = None  # 奖励范围
        self.metadata = None  # 元数据
        self.spec = None  # 规格

        seed = task_config.seed  # 从任务配置中获取种子
        if seed == -1:  # 如果种子为-1，则使用当前时间的纳秒数作为种子
            seed = time.time_ns() % (2**32)
        self.seed(seed)  # 设置随机种子

    @abstractmethod
    def render(self, mode="human"):
        # 渲染环境（抽象方法，需要在子类中实现）
        raise NotImplementedError

    def seed(self, seed):
        # 设置随机种子的方法
        if seed is None or seed < 0:  # 检查种子的有效性
            logger.info(f"Seed is not valid. Will be sampled from system time.")
            seed = time.time_ns() % (2**32)  # 使用系统时间生成种子
        np.random.seed(seed)  # 设置numpy的随机种子
        torch.manual_seed(seed)  # 设置PyTorch的随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
        os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子

        logger.info("Setting seed: {}".format(seed))  # 日志记录设置的种子

    @abstractmethod
    def reset(self):
        # 重置环境状态（抽象方法，需要在子类中实现）
        raise NotImplementedError

    @abstractmethod
    def reset_idx(self, env_ids):
        # 根据给定的环境ID重置特定环境（抽象方法，需要在子类中实现）
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        # 执行一步操作并返回结果（抽象方法，需要在子类中实现）
        raise NotImplementedError

    @abstractmethod
    def close(self):
        # 关闭环境（抽象方法，需要在子类中实现）
        raise NotImplementedError
