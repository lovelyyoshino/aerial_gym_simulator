import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image


def sample_command(args):
    use_warp = True  # 是否使用warp技术
    headless = args.headless  # 是否以无头模式运行（不显示图形界面）

    rl_task = task_registry.make_task(
        "dce_navigation_task", seed=42, use_warp=use_warp, headless=headless
    )  # 创建导航任务实例
    print("Number of environments", rl_task.num_envs)  # 打印环境数量
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim))  # 初始化命令动作
    command_actions[:, 0] = 1.5  # 设置第一个动作维度的初始值
    command_actions[:, 1] = 0.0  # 设置第二个动作维度的初始值
    command_actions[:, 2] = 0.0  # 设置第三个动作维度的初始值
    nn_model = get_network(rl_task.num_envs)  # 获取神经网络模型
    nn_model.eval()  # 将模型设置为评估模式
    nn_model.reset(torch.arange(rl_task.num_envs))  # 重置神经网络状态
    rl_task.reset()  # 重置任务环境
    for i in range(0, 50000):  # 主循环，最多执行50000次
        start_time = time.time()  # 记录开始时间
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)  # 执行一步操作并获取观察结果、奖励等信息

        obs["obs"] = obs["observations"]  # 更新观察字典中的观测数据
        action = nn_model.get_action(obs)  # 从神经网络中获取下一个动作
        action = torch.tensor(action).expand(rl_task.num_envs, -1)  # 扩展动作张量以匹配环境数量
        command_actions[:] = action  # 更新命令动作

        reset_ids = (termination + truncation).nonzero(as_tuple=True)  # 找到需要重置的环境ID
        if torch.any(termination):  # 如果有环境终止
            terminated_envs = termination.nonzero(as_tuple=True)  # 获取所有终止的环境
            print(f"Resetting environments {terminated_envs} due to Termination")  # 打印重置信息
        if torch.any(truncation):  # 如果有环境超时
            truncated_envs = truncation.nonzero(as_tuple=True)  # 获取所有超时的环境
            print(f"Resetting environments {truncated_envs} due to Timeout")  # 打印重置信息
        nn_model.reset(reset_ids)  # 重置神经网络状态

    # 以下代码用于保存每一集的帧作为GIF，但目前被注释掉
    # ...

def get_network(num_envs):
    """Script entry point."""
    cfg = parse_aerialgym_cfg(evaluation=True)  # 解析配置文件
    print("CFG is:", cfg)  # 打印配置内容
    nn_model = NN_Inference_Class(num_envs, 3, 81, cfg)  # 创建神经网络推理类实例
    return nn_model  # 返回神经网络模型

if __name__ == "__main__":
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config(
            "navigation_task"
        ),  # 使用与导航任务相同的配置
    )
    args = get_args()  # 获取命令行参数
    sample_command(args)  # 调用sample_command函数
