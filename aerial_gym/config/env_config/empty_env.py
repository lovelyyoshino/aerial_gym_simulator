# 这里不需要导入任何模块，因为定义基类时不需要其他模块


class EmptyEnvCfg:
    class env:
        num_envs = 3  # 环境的数量
        num_env_actions = 0  # 环境处理的动作数量
        # 这些是发送给环境实体的动作
        # 其中一些可能用于控制环境中的各种实体
        # 例如，障碍物的运动等。
        env_spacing = 1.0  # 在高度场/三角网格中未使用
        num_physics_steps_per_env_step_mean = 1  # 每个环境步骤之间的平均物理步数
        num_physics_steps_per_env_step_std = 0  # 每个环境步骤之间的物理步数标准差
        render_viewer_every_n_steps = 10  # 每n步渲染一次观察者
        collision_force_threshold = 0.010  # 碰撞力阈值
        manual_camera_trigger = False  # 手动触发相机捕捉
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        create_ground_plane = False  # 创建地面平面
        sample_timestep_for_latency = True  # 为延迟噪声采样时间步长
        perturb_observations = True  # 扰动观测值
        keep_same_env_for_num_episodes = 1  # 在多个回合中保持相同的环境
        write_to_sim_at_every_timestep = False  # 是否在每个时间步写入模拟

        use_warp = False  # 是否使用扭曲
        e_s = env_spacing  # 将环境间距赋值给e_s
        lower_bound_min = [-e_s, -e_s, -e_s]  # 环境空间的下界最小值
        lower_bound_max = [-e_s, -e_s, -e_s]  # 环境空间的下界最大值
        upper_bound_min = [e_s, e_s, e_s]  # 环境空间的上界最小值
        upper_bound_max = [e_s, e_s, e_s]  # 环境空间的上界最大值

    class env_config:
        include_asset_type = {}  # 包含资产类型的字典

        asset_type_to_dict_map = {}  # 资产类型到字典映射的字典
