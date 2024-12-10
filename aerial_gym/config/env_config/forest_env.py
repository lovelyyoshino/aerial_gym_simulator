from aerial_gym.config.asset_config.env_object_config import (
    tree_asset_params,
    object_asset_params,
    bottom_wall,
)

import numpy as np


class ForestEnvCfg:
    class env:
        num_envs = 64  # 环境的数量
        num_env_actions = 4  # 环境处理的动作数量
        # 这些动作可能来自于RL代理，用于控制机器人，或者用于控制环境中的各种实体，例如障碍物的运动等。
        env_spacing = 5.0  # 环境之间的间距（在使用高度场/三角网格时未使用）

        num_physics_steps_per_env_step_mean = 10  # 每个环境步骤之间的平均物理步数
        num_physics_steps_per_env_step_std = 0  # 每个环境步骤之间的物理步数标准差

        render_viewer_every_n_steps = 1  # 每n步渲染一次观察者视图
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        collision_force_threshold = 0.005  # 碰撞力阈值 [N]
        create_ground_plane = False  # 是否创建地面平面
        sample_timestep_for_latency = True  # 是否为延迟噪声采样时间步长
        perturb_observations = True  # 是否扰动观测值
        keep_same_env_for_num_episodes = 1  # 在多个回合中保持相同的环境
        write_to_sim_at_every_timestep = False  # 是否在每个时间步写入模拟数据

        use_warp = True  # 是否使用扭曲效果
        lower_bound_min = [-5.0, -5.0, -1.0]  # 环境空间的下边界最小值
        lower_bound_max = [-5.0, -5.0, -1.0]  # 环境空间的下边界最大值
        upper_bound_min = [5.0, 5.0, 3.0]  # 环境空间的上边界最小值
        upper_bound_max = [5.0, 5.0, 3.0]  # 环境空间的上边界最大值

    class env_config:
        include_asset_type = {
            "trees": True,  # 包含树木资产类型
            "objects": True,  # 包含其他对象资产类型
            "bottom_wall": True,  # 包含底部墙壁资产类型
        }

        # 将上述名称映射到定义资产的类。可以在include_asset_type中启用和禁用它们
        asset_type_to_dict_map = {
            "trees": tree_asset_params,  # 树木资产参数
            "objects": object_asset_params,  # 其他对象资产参数
            "bottom_wall": bottom_wall,  # 底部墙壁资产参数
        }
