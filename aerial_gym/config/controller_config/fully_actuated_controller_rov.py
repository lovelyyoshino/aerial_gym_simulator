import numpy as np


class control:
    """
    Controller configuration for the fully-actuated controller
    控制器配置，用于完全驱动控制器
    """

    num_actions = 7  # 定义动作的数量，这里是7个动作
    max_inclination_angle_rad = np.pi / 3.0  # 最大倾斜角度，单位为弧度（约60度）
    max_yaw_rate = np.pi / 3.0  # 最大偏航速率，单位为弧度（约60度）

    K_pos_tensor_max = [1.0, 1.0, 1.0]  # 用于位置控制的最大增益，仅用于lee_position_control
    K_pos_tensor_min = [1.0, 1.0, 1.0]  # 用于位置控制的最小增益，仅用于lee_position_control

    K_vel_tensor_max = [
        8.0,
        8.0,
        8.0,
    ]  # 用于速度控制的最大增益，仅用于lee_position_control和lee_velocity_control
    K_vel_tensor_min = [8.0, 8.0, 8.0]  # 用于速度控制的最小增益，仅用于lee_position_control和lee_velocity_control

    K_rot_tensor_max = [
        2.2,
        2.2,
        2.6,
    ]  # 用于旋转控制的最大增益，仅用于lee_position_control、lee_velocity_control和lee_attitude_control
    K_rot_tensor_min = [2.2, 2.2, 2.6]  # 用于旋转控制的最小增益，仅用于lee_position_control、lee_velocity_control和lee_attitude_control

    K_angvel_tensor_max = [
        2.2,
        2.2,
        2.2,
    ]  # 用于角速度控制的最大增益，仅用于lee_position_control、lee_velocity_control和lee_attitude_control
    K_angvel_tensor_min = [2.1, 2.1, 2.1]  # 用于角速度控制的最小增益，仅用于lee_position_control、lee_velocity_control和lee_attitude_control

    randomize_params = True  # 是否随机化参数，默认为True
