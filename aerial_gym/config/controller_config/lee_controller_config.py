import numpy as np


class control:
    """
    控制参数类
    controller:
        lee_position_control: 命令动作 = [x, y, z, yaw] 在环境坐标系中，缩放范围在 -1 到 1 之间
        lee_velocity_control: 命令动作 = [vx, vy, vz, yaw_rate] 在车辆坐标系中，缩放范围在 -1 到 1 之间
        lee_attitude_control: 命令动作 = [thrust, roll, pitch, yaw_rate] 在车辆坐标系中，缩放范围在 -1 到 1 之间
    kP: 位置控制增益
    kV: 速度控制增益
    kR: 姿态控制增益
    kOmega: 角速度控制增益
    """

    num_actions = 4  # 定义控制动作的数量为4

    max_inclination_angle_rad = np.pi / 3.0  # 最大倾斜角度（弧度）
    max_yaw_rate = np.pi / 3.0  # 最大偏航速率（弧度）

    K_pos_tensor_max = [3.0, 3.0, 2.0]  # 用于lee_position_control的最大位置控制增益
    K_pos_tensor_min = [2.0, 2.0, 1.0]  # 用于lee_position_control的最小位置控制增益

    K_vel_tensor_max = [
        3.0,
        3.0,
        3.0,
    ]  # 用于lee_position_control和lee_velocity_control的最大速度控制增益
    K_vel_tensor_min = [2.0, 2.0, 2.0]  # 用于lee_velocity_control的最小速度控制增益

    K_rot_tensor_max = [
        1.2,
        1.2,
        0.6,
    ]  # 用于lee_position_control、lee_velocity_control和lee_attitude_control的最大旋转控制增益
    K_rot_tensor_min = [0.8, 0.8, 0.4]  # 用于lee_attitude_control的最小旋转控制增益

    K_angvel_tensor_max = [
        0.2,
        0.2,
        0.2,
    ]  # 用于lee_position_control、lee_velocity_control和lee_attitude_control的最大角速度控制增益
    K_angvel_tensor_min = [0.1, 0.1, 0.1]  # 用于lee_attitude_control的最小角速度控制增益

    randomize_params = False  # 是否随机化参数的布尔值
