import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

# 导入深度相机配置类
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
# 导入激光雷达配置类
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)

# 导入面部识别相机配置类
from aerial_gym.config.sensor_config.camera_config.base_normal_faceID_camera_config import (
    BaseNormalFaceIDCameraConfig,
)

# 导入OSDome 64激光雷达配置类
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
# 导入IMU配置类
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class BaseQuadCfg:
    # 基础四旋翼配置类

    class init_config:
        # 初始化状态张量格式为 [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (用于保持形状), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.1,
            0.15,
            0.15,
            0,  # -np.pi / 6,
            0,  # -np.pi / 6,
            -np.pi / 6,
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
            0,  # np.pi / 6,
            0,  # np.pi / 6,
            np.pi / 6,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]

    class sensor_config:
        # 传感器配置
        enable_camera = False  # 是否启用相机
        camera_config = BaseDepthCameraConfig  # 使用的相机配置类

        enable_lidar = False  # 是否启用激光雷达
        lidar_config = BaseLidarConfig  # 使用的激光雷达配置类

        enable_imu = False  # 是否启用IMU
        imu_config = BaseImuConfig  # 使用的IMU配置类

    class disturbance:
        # 扰动配置
        enable_disturbance = False  # 是否启用扰动
        prob_apply_disturbance = 0.02  # 应用扰动的概率
        max_force_and_torque_disturbance = [0.75, 0.75, 0.75, 0.004, 0.004, 0.004]  # 最大力和扭矩扰动

    class damping:
        # 阻尼配置
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体[x, y, z]轴的线性阻尼系数
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体[x, y, z]轴的二次阻尼系数
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体[x, y, z]轴的角速度线性阻尼系数
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # 沿着机体[x, y, z]轴的角速度二次阻尼系数

    class robot_asset:
        # 机器人资产配置
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/quad"  # 资源文件夹路径
        file = "quad.urdf"  # 机器人模型文件
        name = "base_quadrotor"  # 机器人名称
        base_link_name = "base_link"  # 基本链接名称
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = False  # 合并固定关节连接的物体
        fix_base_link = False  # 固定机器人的基座
        collision_mask = 0  # 碰撞掩码，1表示禁用，0表示启用...位过滤
        replace_cylinder_with_capsule = False  # 用胶囊替换碰撞圆柱体，提高模拟稳定性
        flip_visual_attachments = True  # 一些.obj网格需要从y-up翻转到z-up
        density = 0.000001  # 密度
        angular_damping = 0.01  # 角阻尼
        linear_damping = 0.01  # 线阻尼
        max_angular_velocity = 100.0  # 最大角速度
        max_linear_velocity = 100.0  # 最大线速度
        armature = 0.001  # 骨架参数

        semantic_id = 0  # 语义ID
        per_link_semantic = False  # 每个链接的语义标记

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
        keep_in_env = True  # 保持在环境中（对机器人没有影响）

        min_position_ratio = None  # 最小位置比例
        max_position_ratio = None  # 最大位置比例

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # 最小欧拉角
        max_euler_angles = [np.pi, np.pi, np.pi]  # 最大欧拉角

        place_force_sensor = True  # 如果需要IMU则设置为True
        force_sensor_parent_link = "base_link"  # 力传感器父链接
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 力传感器变换 [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # 对于机器人无效

    class control_allocator_config:
        # 控制分配配置
        num_motors = 4  # 电机数量
        force_application_level = "motor_link"  # 力应用级别，可以是"motor_link"或"root_link"

        application_mask = [1 + 4 + i for i in range(0, 4)]  # 应用掩码
        motor_directions = [1, -1, 1, -1]  # 电机方向

        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.01, 0.01, -0.01, 0.01],
        ]  # 分配矩阵

        class motor_model_config:
            # 电机模型配置
            use_rps = True  # 是否使用转速

            motor_thrust_constant_min = 0.00000926312  # 最小推力常数
            motor_thrust_constant_max = 0.00001826312  # 最大推力常数

            motor_time_constant_increasing_min = 0.09  # 增加时最小时间常数
            motor_time_constant_increasing_max = 0.12  # 增加时最大时间常数

            motor_time_constant_decreasing_min = 0.03  # 减少时最小时间常数
            motor_time_constant_decreasing_max = 0.05  # 减少时最大时间常数

            max_thrust = 2  # 最大推力
            min_thrust = 0  # 最小推力

            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = 0.01  # 推力与扭矩比
            use_discrete_approximation = (
                False  # 设置为False将根据差异和时间常数计算f'
            )
