import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1  # 随机种子，-1表示使用系统随机数生成器
    sim_name = "base_sim"  # 模拟名称
    env_name = "env_with_obstacles"  # 环境名称
    robot_name = "lmf2"  # 机器人名称
    controller_name = "lmf2_velocity_control"  # 控制器名称
    args = {}  # 额外参数
    num_envs = 1024  # 环境数量
    use_warp = True  # 是否使用warp技术
    headless = True  # 是否无头模式（不显示图形界面）
    device = "cuda:0"  # 使用的设备，CUDA支持GPU加速
    observation_space_dim = 13 + 4 + 64  # 观测空间维度：根状态 + 动作维度 + 潜在维度
    privileged_observation_space_dim = 0  # 特权观测空间维度
    action_space_dim = 4  # 动作空间维度
    episode_len_steps = 100  # 每个episode的步长，实际物理时间为此值乘以sim.dt

    return_state_before_reset = (
        False  # 通常情况下，在重置之前不会返回状态
    )
    # 用户可以根据需要将其设置为True

    target_min_ratio = [0.90, 0.1, 0.1]  # 相对于环境边界的目标比例（x,y,z）
    target_max_ratio = [0.94, 0.90, 0.90]  # 相对于环境边界的目标比例（x,y,z）

    reward_parameters = {
        "pos_reward_magnitude": 5.0,  # 位置奖励幅度
        "pos_reward_exponent": 1.0 / 3.5,  # 位置奖励指数
        "very_close_to_goal_reward_magnitude": 5.0,  # 非常接近目标的奖励幅度
        "very_close_to_goal_reward_exponent": 2.0,  # 非常接近目标的奖励指数
        "getting_closer_reward_multiplier": 10.0,  # 越来越接近目标的奖励倍增因子
        "x_action_diff_penalty_magnitude": 0.8,  # x方向动作差异惩罚幅度
        "x_action_diff_penalty_exponent": 3.333,  # x方向动作差异惩罚指数
        "z_action_diff_penalty_magnitude": 0.8,  # z方向动作差异惩罚幅度
        "z_action_diff_penalty_exponent": 5.0,  # z方向动作差异惩罚指数
        "yawrate_action_diff_penalty_magnitude": 0.8,  # 偏航率动作差异惩罚幅度
        "yawrate_action_diff_penalty_exponent": 3.33,  # 偏航率动作差异惩罚指数
        "x_absolute_action_penalty_magnitude": 0.1,  # x方向绝对动作惩罚幅度
        "x_absolute_action_penalty_exponent": 0.3,  # x方向绝对动作惩罚指数
        "z_absolute_action_penalty_magnitude": 1.5,  # z方向绝对动作惩罚幅度
        "z_absolute_action_penalty_exponent": 1.0,  # z方向绝对动作惩罚指数
        "yawrate_absolute_action_penalty_magnitude": 1.5,  # 偏航率绝对动作惩罚幅度
        "yawrate_absolute_action_penalty_exponent": 2.0,  # 偏航率绝对动作惩罚指数
        "collision_penalty": -100.0,  # 碰撞惩罚
    }

    class vae_config:
        use_vae = True  # 是否使用变分自编码器
        latent_dims = 64  # 潜在维度
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )  # VAE模型文件路径
        model_folder = AERIAL_GYM_DIRECTORY  # 模型文件夹路径
        image_res = (270, 480)  # 图像分辨率
        interpolation_mode = "nearest"  # 插值模式
        return_sampled_latent = True  # 是否返回采样的潜在变量

    class curriculum:
        min_level = 15  # 最小课程级别
        max_level = 50  # 最大课程级别
        check_after_log_instances = 2048  # 日志实例后检查次数
        increase_step = 2  # 增加步骤
        decrease_step = 1  # 减少步骤
        success_rate_for_increase = 0.7  # 增加级别所需成功率
        success_rate_for_decrease = 0.6  # 减少级别所需成功率

        def update_curriculim_level(self, success_rate, current_level):
            """
            更新课程级别，根据成功率调整当前级别。
            
            :param success_rate: 当前成功率
            :param current_level: 当前课程级别
            :return: 更新后的课程级别
            """
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)  # 成功率高于阈值时增加级别
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)  # 成功率低于阈值时减少级别
            return current_level  # 否则保持当前级别

    def action_transformation_function(action):
        """
        将原始动作转换为处理过的动作，以适应机器人的控制需求。

        :param action: 原始动作输入，包含速度和偏航率信息
        :return: 处理后的动作输出
        """
        clamped_action = torch.clamp(action, -1.0, 1.0)  # 限制动作范围在[-1, 1]
        max_speed = 2.0  # 最大速度 [m/s]
        max_yawrate = torch.pi / 3  # 最大偏航率 [rad/s]

        max_inclination_angle = torch.pi / 4  # 最大倾斜角度 [rad]

        clamped_action[:, 0] += 1.0  # 对第一个动作进行偏移

        processed_action = torch.zeros(
            (clamped_action.shape[0], 4), device=task_config.device, requires_grad=False
        )  # 初始化处理后的动作张量
        processed_action[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0  # 计算x方向的速度
        )
        processed_action[:, 1] = 0  # y方向速度设为0
        processed_action[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0  # 计算z方向的速度
        )
        processed_action[:, 3] = clamped_action[:, 2] * max_yawrate  # 计算偏航率
        return processed_action  # 返回处理后的动作
