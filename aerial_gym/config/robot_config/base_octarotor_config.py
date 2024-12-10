import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseOctarotorCfg:
    # 基础八旋翼配置类

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
        # 传感器配置
        enable_camera = False  # 是否启用相机
        camera_config = BaseDepthCameraConfig  # 相机配置类

        enable_lidar = False  # 是否启用激光雷达
        lidar_config = BaseLidarConfig  # 激光雷达配置类

        enable_imu = False  # 是否启用惯性测量单元
        imu_config = BaseImuConfig  # IMU 配置类

    class disturbance:
        # 扰动配置
        enable_disturbance = True  # 是否启用扰动
        prob_apply_disturbance = 0.05  # 应用扰动的概率
        max_force_and_torque_disturbance = [1.5, 1.5, 1.5, 0.25, 0.25, 0.25]  # 最大力和扭矩扰动

    class damping:
        # 阻尼配置
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体 [x, y, z] 轴的线性阻尼系数
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体 [x, y, z] 轴的二次阻尼系数
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体 [x, y, z] 轴的角度线性阻尼系数
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体 [x, y, z] 轴的角度二次阻尼系数

    class robot_asset:
        # 机器人资产配置
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/octarotor"  # 资源文件夹路径
        file = "octarotor.urdf"  # URDF 文件名
        name = "base_octarotor"  # 机器人名称
        base_link_name = "base_link"  # 基本链接名称
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = False  # 合并固定关节连接的物体
        fix_base_link = False  # 固定机器人的基座
        collision_mask = 0  # 碰撞掩码，1 禁用，0 启用...位过滤
        replace_cylinder_with_capsule = False  # 用胶囊替换碰撞圆柱体，提高模拟速度和稳定性
        flip_visual_attachments = True  # 一些 .obj 网格需要从 y-up 翻转到 z-up
        density = 0.000001  # 密度
        angular_damping = 0.0000001  # 角阻尼
        linear_damping = 0.0000001  # 线阻尼
        max_angular_velocity = 100.0  # 最大角速度
        max_linear_velocity = 100.0  # 最大线速度
        armature = 0.001  # 骨架参数

        semantic_id = 0  # 语义 ID
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
        ]  # 状态比例最小值 [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
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
        ]  # 状态比例最大值 [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # 最大力和扭矩扰动 [fx, fy, fz, tx, ty, tz]

        color = None  # 颜色
        semantic_masked_links = {}  # 语义遮罩链接
        keep_in_env = True  # 在环境中保留此设置对机器人没有影响

        min_position_ratio = None  # 最小位置比例
        max_position_ratio = None  # 最大位置比例

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # 欧拉角最小值
        max_euler_angles = [np.pi, np.pi, np.pi]  # 欧拉角最大值

        place_force_sensor = True  # 如果需要 IMU，则将其设置为 True
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
        # 控制分配器配置
        num_motors = 8  # 电机数量
        force_application_level = "motor_link"  # 力应用级别，可以是 "motor_link" 或 "root_link"

        application_mask = [1 + 8 + i for i in range(0, 8)]  # 应用掩码
        motor_directions = [1, -1, 1, -1, 1, -1, 1, -1]  # 电机方向

        allocation_matrix = [
            [
                -0.78867513,
                0.21132487,
                -0.21132487,
                0.78867513,
                0.78867513,
                -0.21132487,
                0.21132487,
                -0.78867513,
            ],
            [
                0.21132487,
                0.78867513,
                -0.78867513,
                -0.21132487,
                -0.21132487,
                -0.78867513,
                0.78867513,
                0.21132487,
            ],
            [
                0.57735027,
                -0.57735027,
                -0.57735027,
                0.57735027,
                0.57735027,
                -0.57735027,
                -0.57735027,
                0.57735027,
            ],
            [
                0.14226497,
                -0.21547005,
                0.25773503,
                0.01547005,
                -0.01547005,
                -0.25773503,
                0.21547005,
                -0.14226497,
            ],
            [
                -0.25773503,
                0.01547005,
                0.14226497,
                0.21547005,
                -0.21547005,
                -0.14226497,
                -0.01547005,
                0.25773503,
            ],
            [
                0.11547005,
                -0.23094011,
                -0.11547005,
                0.23094011,
                -0.23094011,
                0.11547005,
                0.23094011,
                -0.11547005,
            ],
        ]

        class motor_model_config:
            # 电机模型配置
            use_rps = False  # 是否使用转速
            motor_thrust_constant_min = 0.00000926312  # 电机推力常数最小值
            motor_thrust_constant_max = 0.00001826312  # 电机推力常数最大值
            motor_time_constant_increasing_min = 0.01  # 电机增大时间常数最小值
            motor_time_constant_increasing_max = 0.03  # 电机增大时间常数最大值
            motor_time_constant_decreasing_min = 0.005  # 电机减小时间常数最小值
            motor_time_constant_decreasing_max = 0.005  # 电机减小时间常数最大值
            max_thrust = 6.25  # 最大推力
            min_thrust = -6.25  # 最小推力
            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = (
                0.01  # 推力与扭矩比，与惯性矩阵相关，不要更改
            )
            use_discrete_approximation = (
                True  # 设置为 false 将根据差异和时间常数计算 f'
            )
