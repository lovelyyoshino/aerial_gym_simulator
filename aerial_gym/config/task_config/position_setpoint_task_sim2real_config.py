import torch

# 定义一个任务配置类，用于存储与仿真和机器人控制相关的参数
class task_config:
    seed = 1  # 随机种子，用于确保实验可重复性
    sim_name = "base_sim"  # 仿真名称
    env_name = "empty_env"  # 环境名称
    robot_name = "lmf2"  # 机器人名称
    controller_name = "lmf2_velocity_control"  # 控制器名称
    args = {}  # 用于存储额外的参数，默认为空字典
    num_envs = 16  # 同时运行的环境数量
    use_warp = False  # 是否使用图形加速（warp），默认不使用
    headless = False  # 是否以无头模式运行（即不显示图形界面），默认不启用
    device = "cuda:0"  # 使用的设备，这里指定为第一个CUDA设备（GPU）
    observation_space_dim = 17  # 观察空间的维度，即输入给算法的信息量
    privileged_observation_space_dim = 0  # 特权观察空间的维度，通常用于包含额外信息的情况，默认为0
    action_space_dim = 4  # 动作空间的维度，表示可以采取的动作数量
    episode_len_steps = 800  # 每个回合的步数，实际物理时间是这个值乘以sim.dt
    return_state_before_reset = False  # 在重置之前是否返回状态，默认为False
    reward_parameters = {}  # 奖励参数，默认为空字典，可以根据需要进行设置
