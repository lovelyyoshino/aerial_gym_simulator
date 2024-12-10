from aerial_gym.config.asset_config.dynamic_env_object_config import *

import numpy as np


class DynamicEnvironmentCfg:
    class env:
        num_envs = 64  # 环境数量，如果在任务配置中使用，则由num_envs参数覆盖
        num_env_actions = 6  # 环境处理的动作数量
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
        create_ground_plane = True  # 创建地面平面
        sample_timestep_for_latency = True  # 为延迟噪声采样时间步长
        perturb_observations = True  # 扰动观测值
        keep_same_env_for_num_episodes = 1  # 保持相同环境进行的回合数
        write_to_sim_at_every_timestep = True  # 在每个时间步写入模拟数据

        use_warp = True  # 是否使用扭曲效果
        lower_bound_min = [-2.0, -4.0, 0.0]  # 环境空间的下界最小值
        lower_bound_max = [-1.0, -2.5, 0.0]  # 环境空间的下界最大值
        upper_bound_min = [9.0, 2.5, 4.0]  # 环境空间的上界最小值
        upper_bound_max = [10.0, 4.0, 5.0]  # 环境空间的上界最大值

    class env_config:
        include_asset_type = {
            "panels": False,  # 面板资产是否包含
            "thin": False,  # 薄资产是否包含
            "trees": False,  # 树木资产是否包含
            "objects": True,  # 对象资产是否包含
            "left_wall": False,  # 左墙资产是否包含
            "right_wall": False,  # 右墙资产是否包含
            "back_wall": False,  # 后墙资产是否包含
            "front_wall": False,  # 前墙资产是否包含
            "top_wall": False,  # 顶墙资产是否包含
            "bottom_wall": False,  # 底墙资产是否包含
        }

        # 将上述名称映射到定义资产的类。它们可以在include_asset_type中启用和禁用
        asset_type_to_dict_map = {
            "panels": panel_asset_params,  # 面板资产参数
            "thin": thin_asset_params,  # 薄资产参数
            "trees": tree_asset_params,  # 树木资产参数
            "objects": object_asset_params,  # 对象资产参数
            "left_wall": left_wall,  # 左墙参数
            "right_wall": right_wall,  # 右墙参数
            "back_wall": back_wall,  # 后墙参数
            "front_wall": front_wall,  # 前墙参数
            "bottom_wall": bottom_wall,  # 底墙参数
            "top_wall": top_wall,  # 顶墙参数
        }
