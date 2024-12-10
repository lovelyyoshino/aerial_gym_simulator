from aerial_gym.config.asset_config.env_object_config import (
    panel_asset_params,  # 面板资产参数
    thin_asset_params,   # 薄物体资产参数
    tree_asset_params,   # 树木资产参数
    object_asset_params, # 其他对象资产参数
    tile_asset_params,   # 瓦片资产参数
)
from aerial_gym.config.asset_config.env_object_config import (
    left_wall,          # 左墙参数
    right_wall,         # 右墙参数
    back_wall,          # 后墙参数
    front_wall,         # 前墙参数
    bottom_wall,        # 底墙参数
    top_wall,           # 顶墙参数
)

import numpy as np


class EnvWithObstaclesCfg:
    class env:
        num_envs = 64  # 环境数量，如果在任务配置中使用，则由num_envs参数覆盖
        num_env_actions = 4  # 环境处理的动作数量
        # 这些动作可能来自于RL代理用于控制机器人，
        # 也可以用于控制环境中的各种实体，例如障碍物的运动等。
        env_spacing = 5.0  # 在高度场/三角网格中未使用

        num_physics_steps_per_env_step_mean = 10  # 每个环境步骤之间的平均物理步数
        num_physics_steps_per_env_step_std = 0  # 每个环境步骤之间的物理步数标准差

        render_viewer_every_n_steps = 1  # 每n步渲染一次观察者
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        collision_force_threshold = 0.05  # 碰撞力阈值 [N]
        create_ground_plane = False  # 是否创建地面平面
        sample_timestep_for_latency = True  # 为延迟噪声采样时间步长
        perturb_observations = True  # 是否扰动观测数据
        keep_same_env_for_num_episodes = 1  # 保持相同环境的回合数
        write_to_sim_at_every_timestep = False  # 是否在每个时间步写入模拟

        use_warp = True  # 是否使用扭曲
        lower_bound_min = [-2.0, -4.0, -3.0]  # 环境空间的下界最小值
        lower_bound_max = [-1.0, -2.5, -2.0]  # 环境空间的下界最大值
        upper_bound_min = [9.0, 2.5, 2.0]  # 环境空间的上界最小值
        upper_bound_max = [10.0, 4.0, 3.0]  # 环境空间的上界最大值

    class env_config:
        include_asset_type = {
            "panels": True,  # 包含面板类型
            "tiles": False,  # 不包含瓦片类型
            "thin": False,   # 不包含薄物体类型
            "trees": False,  # 不包含树木类型
            "objects": True, # 包含其他对象类型
            "left_wall": True,  # 包含左墙
            "right_wall": True, # 包含右墙
            "back_wall": True,  # 包含后墙
            "front_wall": True, # 包含前墙
            "top_wall": True,   # 包含顶墙
            "bottom_wall": True, # 包含底墙
        }

        # 将上述名称映射到定义资产的类。它们可以在include_asset_type中启用和禁用
        asset_type_to_dict_map = {
            "panels": panel_asset_params,  # 面板资产参数映射
            "thin": thin_asset_params,      # 薄物体资产参数映射
            "trees": tree_asset_params,      # 树木资产参数映射
            "objects": object_asset_params,  # 其他对象资产参数映射
            "left_wall": left_wall,          # 左墙参数映射
            "right_wall": right_wall,        # 右墙参数映射
            "back_wall": back_wall,          # 后墙参数映射
            "front_wall": front_wall,        # 前墙参数映射
            "bottom_wall": bottom_wall,      # 底墙参数映射
            "top_wall": top_wall,            # 顶墙参数映射
            "tiles": tile_asset_params,      # 瓦片资产参数映射
        }
