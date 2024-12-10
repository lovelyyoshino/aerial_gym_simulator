import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

# 导入基础深度相机配置类
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
# 导入基础激光雷达配置类
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
# 导入OSDome 64激光雷达的具体配置类
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
# 导入基础IMU（惯性测量单元）配置类
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig

# 导入基础四旋翼机器人配置类
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


class BaseQuadRootLinkControlCfg(BaseQuadCfg):
    """
    基础四旋翼根链接控制配置类，继承自BaseQuadCfg。
    此类主要用于定义四旋翼机器人的控制参数和模型配置。
    """

    # 机器人资产配置，基本与BaseQuadCfg相同，但有以下不同之处
    class robot_asset(BaseQuadCfg.robot_asset):
        file = "model.urdf"  # 机器人模型文件路径
        name = "base_quadrotor"  # 机器人名称
        base_link_name = "base_link"  # 根链接名称

    class control_allocator_config:
        """
        控制分配器配置类，用于定义电机数量、力应用级别等控制相关参数。
        """
        num_motors = 4  # 电机数量
        force_application_level = "root_link"  # 力应用级别，可以是"motor_link"或"root_link"

        # 应用掩码，决定哪些电机参与控制
        application_mask = [1 + 4 + i for i in range(0, 4)]
        motor_directions = [1, -1, 1, -1]  # 每个电机的方向设置

        # 分配矩阵，定义如何将控制信号分配给各个电机
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],  # 第三行表示所有电机都接收相同的输入
            [-0.13, -0.13, 0.13, 0.13],  # 用于控制俯仰
            [-0.13, 0.13, 0.13, -0.13],  # 用于控制滚转
            [-0.01, 0.01, -0.01, 0.01],  # 用于控制偏航
        ]

        class motor_model_config:
            """
            电机模型配置类，定义电机的物理特性和行为。
            """
            use_rps = True  # 是否使用转速（RPS）
            motor_thrust_constant_min = 0.00000926312  # 最小推力常数
            motor_thrust_constant_max = 0.00001826312  # 最大推力常数
            motor_time_constant_increasing_min = 0.01  # 增加时的最小时间常数
            motor_time_constant_increasing_max = 0.03  # 增加时的最大时间常数
            motor_time_constant_decreasing_min = 0.005  # 减少时的最小时间常数
            motor_time_constant_decreasing_max = 0.005  # 减少时的最大时间常数
            max_thrust = 10  # 最大推力
            min_thrust = 0  # 最小推力
            max_thrust_rate = 100000.0  # 最大推力变化率
            thrust_to_torque_ratio = 0.01  # 推力与扭矩比
            use_discrete_approximation = (
                True  # 设置为False将基于差值和时间常数计算f'
            )
