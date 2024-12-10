class BaseSimConfig:
    # 基础仿真配置类

    class viewer:
        # 视图设置
        headless = False  # 是否以无头模式运行（不显示窗口）
        ref_env = 0  # 参考环境的索引
        camera_position = [-5, -5, 4]  # 相机位置 [m]
        lookat = [0, 0, 0]  # 相机注视点 [m]
        camera_orientation_euler_deg = [0, 0, 0]  # 相机朝向（欧拉角）[deg]
        camera_follow_type = "FOLLOW_TRANSFORM"  # 相机跟随类型
        width = 1280  # 窗口宽度
        height = 720  # 窗口高度
        max_range = 100.0  # 最大可视范围 [m]
        min_range = 0.1  # 最小可视范围 [m]
        horizontal_fov_deg = 90  # 水平视场角 [deg]
        use_collision_geometry = False  # 是否使用碰撞几何体
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.3]  # 相机跟随变换的局部偏移 [m]
        camera_follow_position_global_offset = [-1.0, 0.0, 0.3]  # 相机跟随位置的全局偏移 [m]

    class sim:
        # 仿真参数设置
        dt = 0.01  # 时间步长 [s]
        substeps = 1  # 子步骤数量
        gravity = [0.0, 0.0, -9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 向上轴，0表示y轴，1表示z轴
        use_gpu_pipeline = True  # 是否使用GPU管道进行计算

        class physx:
            # 物理引擎参数设置
            num_threads = 10  # 使用的线程数
            solver_type = 1  # 求解器类型，0: pgs, 1: tgs
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 1  # 速度迭代次数
            contact_offset = 0.002  # 接触偏移量 [m]
            rest_offset = 0.001  # 静止偏移量 [m]
            bounce_threshold_velocity = 0.1  # 弹跳阈值速度 [m/s]
            max_depenetration_velocity = 1.0  # 最大穿透速度
            max_gpu_contact_pairs = 2**24  # GPU最大接触对数，用于处理8000个及以上环境
            default_buffer_size_multiplier = 10  # 默认缓冲区大小乘数
            contact_collection = 1  # 接触收集策略，0: 从不，1: 最后一个子步骤，2: 所有子步骤（默认=2）
