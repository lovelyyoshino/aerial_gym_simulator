import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

from aerial_gym.examples.rl_games_example.rl_games_inference import MLP

import time
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.print_example_message()  # 打印示例消息以确认日志记录正常工作
    start = time.time()  # 记录程序开始时间
    seed = 0  # 设置随机种子，以确保结果可重复
    torch.manual_seed(seed)  # 为PyTorch设置随机种子
    np.random.seed(seed)  # 为NumPy设置随机种子
    torch.cuda.manual_seed(seed)  # 为CUDA设置随机种子（如果使用GPU）

    plt.style.use("seaborn-v0_8-colorblind")  # 设置绘图风格为色盲友好模式
    rl_task_env = task_registry.make_task(
        "position_setpoint_task_morphy",  # 创建一个名为"position_setpoint_task_morphy"的任务环境
        # "position_setpoint_task_acceleration_sim2real",
        # 其他参数未在此处设置，使用任务配置文件中的默认值
        seed=seed,  # 使用之前定义的随机种子
        headless=False,  # 是否以无头模式运行（即不显示GUI）
        num_envs=16,  # 创建16个并行环境
        use_warp=True,  # 启用warp功能以加速模拟
    )
    rl_task_env.reset()  # 重置环境以准备进行新的仿真
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.task_config.action_space_dim,
        )
    ).to("cuda:0")  # 初始化动作张量，并将其移动到GPU上

    model = (
        MLP(  # 创建多层感知机模型
            rl_task_env.task_config.observation_space_dim,  # 输入维度为观察空间的维度
            rl_task_env.task_config.action_space_dim,  # 输出维度为动作空间的维度
            # "networks/morphy_policy_for_rigid_airframe.pth"
            "networks/morphy_policy_for_flexible_airframe_joint_aware.pth",  # 加载预训练模型权重
        )
        .to("cuda:0")  # 将模型移动到GPU上
        .eval()  # 设置模型为评估模式
    )

    actions[:] = 0.0  # 初始化所有动作为零
    counter = 0  # 初始化计数器
    action_list = []  # 用于存储动作列表
    error_list = []  # 用于存储误差列表
    joint_pos_list = []  # 用于存储关节位置列表
    joint_vel_list = []  # 用于存储关节速度列表
    
    with torch.no_grad():  # 在推理时禁用梯度计算以提高性能
        for i in range(10000):  # 循环执行10000次
            if i == 100:
                start = time.time()  # 每100步重新记录开始时间
            
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)  # 执行动作并获取反馈
            
            start_time = time.time()  # 记录当前时间
            actions[:] = torch.clamp(model.forward(obs["observations"]), -1.0, 1.0)  # 获取模型输出的动作，并限制在[-1, 1]范围内

            end_time = time.time()  # 记录结束时间
    end = time.time()  # 记录程序结束时间
