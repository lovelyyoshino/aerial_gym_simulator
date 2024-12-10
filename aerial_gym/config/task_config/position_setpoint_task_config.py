import torch

# 定义任务配置类，包含了自动驾驶仿真环境的各种参数设置
class task_config:
    seed = 1  # 随机种子，用于确保实验可重复性
    sim_name = "base_sim"  # 仿真名称
    env_name = "empty_env"  # 环境名称
    robot_name = "base_quadrotor"  # 机器人名称，这里是四旋翼无人机
    controller_name = "lee_attitude_control"  # 控制器名称，使用Lee姿态控制算法
    args = {}  # 可选参数字典，可以用于传递额外的配置
    num_envs = 16  # 并行环境数量，表示同时运行的环境实例数
    use_warp = False  # 是否使用warp技术来加速计算
    headless = False  # 是否以无头模式运行（不显示图形界面）
    device = "cuda:0"  # 使用的设备，这里指定为CUDA GPU
    observation_space_dim = 13  # 观察空间维度，表示状态信息的特征数量
    privileged_observation_space_dim = 0  # 特权观察空间维度，通常用于提供额外的信息
    action_space_dim = 4  # 动作空间维度，表示可以执行的动作数量
    episode_len_steps = 500  # 每个回合的步长，实际物理时间为此值乘以sim.dt
    return_state_before_reset = False  # 在重置之前是否返回状态
    reward_parameters = {  # 奖励参数设置
        "pos_error_gain1": [2.0, 2.0, 2.0],  # 第一组位置误差增益
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],  # 第一组位置误差指数
        "pos_error_gain2": [2.0, 2.0, 2.0],  # 第二组位置误差增益
        "pos_error_exp2": [2.0, 2.0, 2.0],  # 第二组位置误差指数
        "dist_reward_coefficient": 7.5,  # 距离奖励系数
        "max_dist": 15.0,  # 最大距离限制
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],  # 动作差异惩罚增益
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],  # 绝对动作奖励增益
        "crash_penalty": -100,  # 碰撞惩罚
    }
