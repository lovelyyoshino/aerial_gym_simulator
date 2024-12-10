import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class SnakeyCfg:
    # SnakeyCfg类用于配置Snakey机器人的各种参数和设置。

    class init_config:
        # 初始化状态张量的格式为 [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (保持形状), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.0,
            0.0,
            0.0,
            0,  # -np.pi / 6，最小滚转角度
            0,  # -np.pi / 6，最小俯仰角度
            -np.pi,  # 最小偏航角度
            1.0,  # 保持形状的常数
            -0.2,  # 最小线速度x
            -0.2,  # 最小线速度y
            -0.2,  # 最小线速度z
            -0.2,  # 最小角速度x
            -0.2,  # 最小角速度y
            -0.2,  # 最小角速度z
        ]
        max_init_state = [
            1.0,
            1.0,
            1.0,
            0,  # np.pi / 6，最大滚转角度
            0,  # np.pi / 6，最大俯仰角度
            np.pi,  # 最大偏航角度
            1.0,  # 保持形状的常数
            0.2,  # 最大线速度x
            0.2,  # 最大线速度y
            0.2,  # 最大线速度z
            0.2,  # 最大角速度x
            0.2,  # 最大角速度y
            0.2,  # 最大角速度z
        ]

    class reconfiguration_config:
        # 重配置相关的设置，包括自由度模式、初始状态范围等。
        dof_mode = "velocity"  # 自由度模式可以是 "position", "velocity" 或 "effort"
        # 选择 effort 意味着你可以实现自己的非线性响应模型

        init_state_min = [
            [-np.pi / 2.0, -0.3, -np.pi / 2.0, -0.3, -np.pi / 2.0, -0.3],  # 位置状态的最小值
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],  # 速度状态的最小值
        ]
        init_state_max = [
            [np.pi / 2.0, +0.3, np.pi / 2.0, +0.3, np.pi / 2.0, +0.3],  # 位置状态的最大值
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 速度状态的最大值
        ]

        if dof_mode == "position":
            stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]  # Kp，位置控制增益
            damping = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # Kd，阻尼系数

        elif dof_mode == "velocity":
            stiffness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Kp，速度控制增益
            damping = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # Kd，阻尼系数

        elif dof_mode == "effort":
            stiffness = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # Kp，努力控制增益
            damping = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]  # Kd，阻尼系数

    class sensor_config:
        # 传感器配置，包括相机、激光雷达和IMU的启用与配置
        enable_camera = False  # 是否启用相机
        camera_config = BaseDepthCameraConfig  # 相机配置类

        enable_lidar = False  # 是否启用激光雷达
        lidar_config = BaseLidarConfig  # 激光雷达配置类（OSDome_64_Config）

        enable_imu = False  # 是否启用IMU
        imu_config = BaseImuConfig  # IMU配置类

    class disturbance:
        # 干扰配置，包括是否启用干扰及其概率和强度
        enable_disturbance = True  # 是否启用干扰
        prob_apply_disturbance = 0.02  # 应用干扰的概率
        max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]  # 最大力和扭矩干扰

    class damping:
        # 阻尼系数配置，用于线性和角动量的阻尼
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿身体[x, y, z]轴的线性阻尼系数
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿身体[x, y, z]轴的二次阻尼系数
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿身体[x, y, z]轴的角动量线性阻尼系数
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿身体[x, y, z]轴的角动量二次阻尼系数

    class robot_asset:
        # 机器人资产配置，包括模型文件路径、名称、重力设置等
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/snakey"  # 资产文件夹路径
        file = "model.urdf"  # 模型文件名
        name = "base_quadrotor"  # 角色名称
        base_link_name = "base_link"  # 基础链接名称
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = False  # 合并固定关节连接的物体
        fix_base_link = False  # 固定机器人的基础部分
        collision_mask = 0  # 碰撞掩码，1表示禁用，0表示启用...位过滤
        replace_cylinder_with_capsule = False  # 用胶囊替换碰撞圆柱体，提高模拟稳定性
        flip_visual_attachments = True  # 一些.obj网格需要从y-up翻转到z-up
        density = 0.000001  # 密度
        angular_damping = 0.01  # 角动量阻尼
        linear_damping = 0.01  # 线性阻尼
        max_angular_velocity = 100.0  # 最大角速度
        max_linear_velocity = 100.0  # 最大线速度
        armature = 0.001  # 骨架质量

        semantic_id = 0  # 语义ID
        per_link_semantic = False  # 每个链接的语义标识

        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # 状态比例的最小值
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # 状态比例的最大值

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # 最大力和扭矩干扰

        color = None  # 颜色
        semantic_masked_links = {}  # 语义遮罩链接
        keep_in_env = True  # 在环境中保持此设置对机器人没有影响

        min_position_ratio = None  # 最小位置比例
        max_position_ratio = None  # 最大位置比例

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # 最小欧拉角
        max_euler_angles = [np.pi, np.pi, np.pi]  # 最大欧拉角

        place_force_sensor = True  # 如果需要IMU则设为True
        force_sensor_parent_link = "base_link"  # 力传感器父链接
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 力传感器变换 [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # 对机器人无效

    class control_allocator_config:
        # 控制分配配置，包括电机数量、力应用级别等
        num_motors = 4  # 电机数量
        force_application_level = "motor_link"  # 力应用级别，可以是 "motor_link" 或 "root_link"

        application_mask = [14, 13, 12, 11]  # 应用掩码
        motor_directions = [-1, 1, -1, 1]  # 电机方向

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [0.01, -0.01, 0.01, -0.01],
        ]  # 分配矩阵

        class motor_model_config:
            # 电机模型配置，包括推力常数、时间常数等
            use_rps = False  # 是否使用转速
            motor_thrust_constant_min = 0.00000926312  # 最小推力常数
            motor_thrust_constant_max = 0.00001826312  # 最大推力常数
            motor_time_constant_increasing_min = 0.005  # 增加时的时间常数最小值
            motor_time_constant_increasing_max = 0.005  # 增加时的时间常数最大值
            motor_time_constant_decreasing_min = 0.005  # 减少时的时间常数最小值
            motor_time_constant_decreasing_max = 0.005  # 减少时的时间常数最大值
            max_thrust = 15  # 最大推力
            min_thrust = 0  # 最小推力
            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = 0.01  # 推力与扭矩比
            use_discrete_approximation = (
                True  # 设置为False将基于差异和时间常数计算f'
            )