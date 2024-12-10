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


class LMF2Cfg:
    # LMF2Cfg类用于配置LMF2机器人，包括初始状态、传感器设置、扰动、阻尼、机器人资产和控制分配等。

    class init_config:
        # 初始化配置，定义机器人的最小和最大初始状态。
        # init_state tensor的格式为 [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (保持形状), vx, vy, vz, wx, wy, wz]
        
        min_init_state = [
            0.1,
            0.15,
            0.15,
            0,  # 滚转角（roll）最小值
            0,  # 俯仰角（pitch）最小值
            -np.pi / 6,  # 偏航角（yaw）最小值
            1.0,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
            -0.2,
        ]
        max_init_state = [
            0.2,
            0.85,
            0.85,
            0,  # 滚转角（roll）最大值
            0,  # 俯仰角（pitch）最大值
            np.pi / 6,  # 偏航角（yaw）最大值
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]

    class sensor_config:
        # 传感器配置，定义是否启用相机、激光雷达和IMU，以及相关的配置类。
        
        enable_camera = True  # 启用相机
        camera_config = BaseDepthCameraConfig  # 相机配置类

        enable_lidar = False  # 禁用激光雷达
        lidar_config = BaseLidarConfig  # 激光雷达配置类（可选）

        enable_imu = False  # 禁用IMU
        imu_config = BaseImuConfig  # IMU配置类

    class disturbance:
        # 扰动配置，定义是否启用扰动及其参数。
        
        enable_disturbance = True  # 启用扰动
        prob_apply_disturbance = 0.05  # 应用扰动的概率
        max_force_and_torque_disturbance = [4.75, 4.75, 4.75, 0.03, 0.03, 0.03]  # 最大力和扭矩扰动

    class damping:
        # 阻尼配置，定义线性和角度运动的阻尼系数。
        
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 线性速度的线性阻尼系数
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 线性速度的二次阻尼系数
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 角速度的线性阻尼系数
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 角速度的二次阻尼系数

    class robot_asset:
        # 机器人资产配置，定义机器人的模型文件、名称、物理属性等。
        
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/lmf2"  # 资源文件夹路径
        file = "model.urdf"  # 模型文件名
        name = "base_quadrotor"  # 机器人名称
        base_link_name = "base_link"  # 基础链接名称
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = True  # 合并固定关节连接的身体
        fix_base_link = False  # 是否固定机器人的基础链接
        collision_mask = 0  # 碰撞掩码，1表示禁用，0表示启用
        replace_cylinder_with_capsule = False  # 用胶囊替换碰撞圆柱体，以提高模拟稳定性
        flip_visual_attachments = True  # 翻转某些.obj网格以适应坐标系
        density = 0.000001  # 密度
        angular_damping = 0.01  # 角阻尼
        linear_damping = 0.01  # 线性阻尼
        max_angular_velocity = 100.0  # 最大角速度
        max_linear_velocity = 100.0  # 最大线速度
        armature = 0.001  # 骨架刚度

        semantic_id = 0  # 语义ID
        per_link_semantic = False  # 每个链接的语义信息

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
        ]  # 最小状态比例
        max_state_ratio = [
            0.9,
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
        ]  # 最大状态比例

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # 最大力和扭矩扰动

        color = None  # 颜色
        semantic_masked_links = {}  # 语义遮罩链接
        keep_in_env = True  # 保持在环境中（对机器人无效）

        min_position_ratio = None  # 最小位置比例
        max_position_ratio = None  # 最大位置比例

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # 最小欧拉角
        max_euler_angles = [np.pi, np.pi, np.pi]  # 最大欧拉角

        place_force_sensor = True  # 如果需要IMU，则设置为True
        force_sensor_parent_link = "base_link"  # 力传感器父链接
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 力传感器变换

        use_collision_mesh_instead_of_visual = False  # 对于机器人没有效果

    class control_allocator_config:
        # 控制分配配置，定义电机数量、力应用级别和电机方向等。
        
        num_motors = 4  # 电机数量
        force_application_level = "base_link"  # 力应用级别，可以是"motor_link"或"root_link"

        application_mask = [1 + 4 + i for i in range(0, 4)]  # 应用掩码
        motor_directions = [1, -1, 1, -1]  # 电机方向

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],  # 分配给每个电机的力
            [-0.13, -0.13, 0.13, 0.13],  # 反向分配
            [-0.13, 0.13, 0.13, -0.13],  # 交叉分配
            [-0.07, 0.07, -0.07, 0.07],  # 其他分配方式
        ]

        class motor_model_config:
            # 电机模型配置，定义电机的动力学特性。
            
            use_rps = True  # 使用转速（RPS）
            motor_thrust_constant_min = 0.00000926312  # 最小推力常数
            motor_thrust_constant_max = 0.00001826312  # 最大推力常数
            motor_time_constant_increasing_min = 0.05  # 增加时的时间常数最小值
            motor_time_constant_increasing_max = 0.08  # 增加时的时间常数最大值
            motor_time_constant_decreasing_min = 0.005  # 减少时的时间常数最小值
            motor_time_constant_decreasing_max = 0.005  # 减少时的时间常数最大值
            max_thrust = 10.0  # 最大推力
            min_thrust = 0.1  # 最小推力
            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = 0.07  # 推力与扭矩比
            use_discrete_approximation = True  # 使用离散近似来描述电机动态
