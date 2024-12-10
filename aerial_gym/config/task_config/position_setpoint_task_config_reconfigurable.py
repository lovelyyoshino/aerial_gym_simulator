import torch
from aerial_gym.utils.math import torch_interpolate_ratio


class task_config:
    seed = 1  # 随机种子，用于确保实验的可重复性
    sim_name = "base_sim_2ms"  # 仿真名称
    env_name = "empty_env_2ms"  # 环境名称
    robot_name = "snakey"  # 机器人名称
    controller_name = "no_control"  # 控制器名称
    args = {}  # 额外参数
    num_envs = 1024  # 环境数量
    use_warp = False  # 是否使用变形
    headless = False  # 是否以无头模式运行（不显示图形界面）
    device = "cuda:0"  # 使用的设备，通常是GPU
    episode_len_steps = 500  # 每个回合的步数，实际物理时间为此值乘以sim.dt
    return_state_before_reset = False  # 在重置之前是否返回状态
    reward_parameters = {  # 奖励参数设置
        "pos_error_gain1": [2.0, 2.0, 2.0],  # 位置误差增益1
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],  # 位置误差指数1
        "pos_error_gain2": [2.0, 2.0, 2.0],  # 位置误差增益2
        "pos_error_exp2": [2.0, 2.0, 2.0],  # 位置误差指数2
        "dist_reward_coefficient": 7.5,  # 距离奖励系数
        "max_dist": 15.0,  # 最大距离
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],  # 动作差异惩罚增益
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],  # 绝对动作奖励增益
        "crash_penalty": -100,  # 碰撞惩罚
    }

    # 以下注释部分包含不同机器人的配置，可以根据需要启用或修改

    robot_name = "snakey6"  # 当前使用的机器人名称
    num_joints = 5 * 2  # 关节数量
    num_motors = 6  # 电机数量
    action_space_dim = num_motors + num_joints  # 动作空间维度
    observation_space_dim = 13 + action_space_dim + num_joints * 2  # 观察空间维度
    privileged_observation_space_dim = 0  # 特权观察空间维度
    # 对于速度目标的限制
    action_limit_max = [15.0] * num_motors + [-1.0, -1.0] * int(num_joints / 2)  # 动作最大限制
    action_limit_min = [0.0] * num_motors + [1.0, 1.0] * int(num_joints / 2)  # 动作最小限制

    @staticmethod
    def process_actions_for_task(actions, min_limit, max_limit):
        """
        处理任务中的动作，将其缩放到指定范围内。

        参数：
        actions (torch.Tensor): 输入的动作张量，范围在[0, 1]之间。
        min_limit (list): 动作的最小限制。
        max_limit (list): 动作的最大限制。

        返回：
        torch.Tensor: 缩放后的动作张量，范围在min_limit和max_limit之间。
        """
        actions_clipped = torch.clamp(actions, 0, 1)  # 将输入动作限制在[0, 1]区间
        scaled_actions = torch_interpolate_ratio(  # 根据给定的最小和最大限制进行插值
            min=min_limit, max=max_limit, ratio=actions_clipped
        )
        return scaled_actions  # 返回缩放后的动作
