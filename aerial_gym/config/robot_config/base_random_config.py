import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseRandCfg:
    # 基础随机配置类，包含初始化状态、传感器配置、干扰、阻尼、机器人资产和控制分配等设置。

    class init_config:
        # 初始化状态张量格式为 [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (用于保持形状), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.0,
            0.0,
            0.0,
            0,  # 最小滚转角度（弧度）
            0,  # 最小俯仰角度（弧度）
            -np.pi,  # 最小偏航角度（弧度）
            1.0,  # 保持形状的常数
            -0.2,  # 最小线速度 x 分量
            -0.2,  # 最小线速度 y 分量
            -0.2,  # 最小线速度 z 分量
            -0.2,  # 最小角速度 x 分量
            -0.2,  # 最小角速度 y 分量
            -0.2,  # 最小角速度 z 分量
        ]
        max_init_state = [
            1.0,
            1.0,
            1.0,
            0,  # 最大滚转角度（弧度）
            0,  # 最大俯仰角度（弧度）
            np.pi,  # 最大偏航角度（弧度）
            1.0,  # 保持形状的常数
            0.2,  # 最大线速度 x 分量
            0.2,  # 最大线速度 y 分量
            0.2,  # 最大线速度 z 分量
            0.2,  # 最大角速度 x 分量
            0.2,  # 最大角速度 y 分量
            0.2,  # 最大角速度 z 分量
        ]

    class sensor_config:
        # 传感器配置，包括相机、激光雷达和IMU的启用与配置
        enable_camera = False  # 是否启用相机
        camera_config = BaseDepthCameraConfig  # 相机配置类

        enable_lidar = False  # 是否启用激光雷达
        lidar_config = BaseLidarConfig  # 激光雷达配置类

        enable_imu = False  # 是否启用IMU
        imu_config = BaseImuConfig  # IMU配置类

    class disturbance:
        # 干扰配置，包括是否启用干扰及其参数
        enable_disturbance = True  # 是否启用干扰
        prob_apply_disturbance = 0.05  # 应用干扰的概率
        max_force_and_torque_disturbance = [1.5, 1.5, 1.5, 0.25, 0.25, 0.25]  # 最大力和扭矩干扰

    class damping:
        # 阻尼系数配置，用于线性和角动量的阻尼
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 线性速度的线性阻尼系数 [x, y, z]
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 线性速度的二次阻尼系数 [x, y, z]
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 角速度的线性阻尼系数 [x, y, z]
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 角速度的二次阻尼系数 [x, y, z]

    class robot_asset:
        # 机器人资产配置，包括模型文件路径、名称、重力设置等
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/random"  # 资源文件夹路径
        file = "random.urdf"  # 机器人模型文件
        name = "base_random"  # 机器人名称
        base_link_name = "base_link"  # 基本链接名称
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = False  # 是否合并固定关节连接的身体
        fix_base_link = False  # 是否固定机器人的基座
        collision_mask = 0  # 碰撞掩码，1表示禁用，0表示启用
        replace_cylinder_with_capsule = False  # 是否将碰撞圆柱替换为胶囊，以提高模拟稳定性
        flip_visual_attachments = True  # 是否翻转某些.obj网格从y-up到z-up
        density = 0.000001  # 密度
        angular_damping = 0.0000001  # 角阻尼
        linear_damping = 0.0000001  # 线阻尼
        max_angular_velocity = 100.0  # 最大角速度
        max_linear_velocity = 100.0  # 最大线速度
        armature = 0.001  # 骨架值

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
            -0.5,
            -0.5,
            -0.5,
            -0.2,
            -0.2,
            -0.2,
        ]  # 状态比例的最小值
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0.5,
            0.5,
            0.5,
            0.2,
            0.2,
            0.2,
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
        keep_in_env = True  # 在环境中保留此机器人

        min_position_ratio = None  # 最小位置比例
        max_position_ratio = None  # 最大位置比例

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # 欧拉角最小值
        max_euler_angles = [np.pi, np.pi, np.pi]  # 欧拉角最大值

        place_force_sensor = True  # 如果需要IMU，则设置为True
        force_sensor_parent_link = "base_link"  # 力传感器父链接
        force_sensor_transform = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # 力传感器变换 [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # 对于机器人无效

    class control_allocator_config:
        # 控制分配配置，包括电机数量、应用级别和分配矩阵等
        num_motors = 8  # 电机数量
        force_application_level = "motor_link"  # 力应用级别，可以是 "motor_link" 或 "root_link"

        application_mask = [1 + 8 + i for i in range(0, 8)]  # 应用掩码
        motor_directions = [-1, 1, -1, 1, -1, 1, -1, 1]  # 电机方向

        allocation_matrix = [
            [
                5.55111512e-17,
                -3.21393805e-01,
                -4.54519478e-01,
                -3.42020143e-01,
                9.69846310e-01,
                3.42020143e-01,
                8.66025404e-01,
                -7.54406507e-01,
            ],
            [
                1.00000000e00,
                -3.42020143e-01,
                -7.07106781e-01,
                0.00000000e00,
                -1.73648178e-01,
                9.39692621e-01,
                5.00000000e-01,
                -1.73648178e-01,
            ],
            [
                1.66533454e-16,
                -8.83022222e-01,
                5.41675220e-01,
                9.39692621e-01,
                1.71010072e-01,
                1.11022302e-16,
                1.11022302e-16,
                6.33022222e-01,
            ],
            [
                1.75000000e-01,
                1.23788742e-01,
                -5.69783368e-02,
                1.34977168e-01,
                3.36959042e-02,
                -2.66534135e-01,
                -7.88397460e-02,
                -2.06893989e-02,
            ],
            [
                1.00000000e-02,
                2.78845133e-01,
                -4.32852308e-02,
                -2.72061766e-01,
                -1.97793856e-01,
                8.63687139e-02,
                1.56554446e-01,
                -1.71261290e-01,
            ],
            [
                2.82487373e-01,
                -1.41735490e-01,
                -8.58541103e-02,
                3.84858939e-02,
                -3.33468026e-01,
                8.36741468e-02,
                8.46777988e-03,
                -8.74336259e-02,
            ],
        ]

        class motor_model_config:
            # 电机模型配置，包括推力常数、时间常数等
            use_rps = False  # 是否使用转速
            motor_thrust_constant_min = 0.00000926312  # 最小推力常数
            motor_thrust_constant_max = 0.00001826312  # 最大推力常数
            motor_time_constant_increasing_min = 0.01  # 增加时的最小时间常数
            motor_time_constant_increasing_max = 0.03  # 增加时的最大时间常数
            motor_time_constant_decreasing_min = 0.005  # 减少时的最小时间常数
            motor_time_constant_decreasing_max = 0.005  # 减少时的最大时间常数
            max_thrust = 5.0  # 最大推力
            min_thrust = -5.0  # 最小推力
            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = (
                0.01  # 推力与扭矩比，与惯性矩阵相关，不要更改
            )
            use_discrete_approximation = (
                True  # 设置为False将根据差异和时间常数计算f'
            )
