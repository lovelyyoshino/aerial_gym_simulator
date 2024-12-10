import torch

# 定义一个任务配置类，用于存储与仿真和机器人控制相关的参数
class task_config:
    seed = 1  # 随机种子，用于确保实验可重复性
    sim_name = "base_sim"  # 仿真名称
    env_name = "empty_env"  # 环境名称
    robot_name = "lmf2"  # 机器人名称
    controller_name = "lmf2_acceleration_control"  # 控制器名称
    args = {}  # 用于存储其他参数的字典
    num_envs = 16  # 并行环境数量
    use_warp = False  # 是否使用时间扭曲（warp）技术
    headless = False  # 是否以无头模式运行（不显示图形界面）
    device = "cuda:0"  # 使用的设备，通常是GPU
    observation_space_dim = 17  # 观察空间维度，即输入特征的数量
    privileged_observation_space_dim = 0  # 特权观察空间维度，通常用于额外信息
    action_space_dim = 4  # 动作空间维度，即输出动作的数量
    episode_len_steps = 800  # 每个回合的步数，实际物理时间为此值乘以sim.dt
    return_state_before_reset = False  # 在重置之前是否返回状态
    reward_parameters = {}  # 奖励参数的字典，用于定义奖励机制
