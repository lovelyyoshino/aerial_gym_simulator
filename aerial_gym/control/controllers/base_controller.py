import torch

class BaseController:
    def __init__(self, control_config, num_envs, device, mode="robot"):
        # 初始化控制器的基本参数
        # control_config: 控制配置，包含控制相关的超参数
        # num_envs: 环境数量，用于并行处理多个环境
        # device: 设备类型（如CPU或GPU）
        # mode: 模式选择，可以是"robot"或"obstacle"，决定了控制器的行为
        self.cfg = control_config
        self.num_envs = num_envs
        self.device = device
        self.mode = mode

    def init_tensors(self, global_tensor_dict):
        # 初始化张量，根据模式设置不同的机器人或障碍物状态信息
        # global_tensor_dict: 包含全局张量数据的字典
        
        if self.mode == "robot":
            # 如果模式为"robot"，则从字典中提取机器人的状态信息
            self.robot_position = global_tensor_dict["robot_position"]  # 机器人的位置
            self.robot_orientation = global_tensor_dict["robot_orientation"]  # 机器人的朝向
            self.robot_linvel = global_tensor_dict["robot_linvel"]  # 机器人的线速度
            self.robot_angvel = global_tensor_dict["robot_angvel"]  # 机器人的角速度
            self.robot_vehicle_orientation = global_tensor_dict["robot_vehicle_orientation"]  # 车辆的朝向
            self.robot_vehicle_linvel = global_tensor_dict["robot_vehicle_linvel"]  # 车辆的线速度
            self.robot_body_angvel = global_tensor_dict["robot_body_angvel"]  # 车身的角速度
            self.robot_body_linvel = global_tensor_dict["robot_body_linvel"]  # 车身的线速度
            self.robot_euler_angles = global_tensor_dict["robot_euler_angles"]  # 欧拉角表示的姿态
            self.mass = global_tensor_dict["robot_mass"].unsqueeze(1)  # 机器人的质量，并增加一个维度以便后续计算
            self.robot_inertia = global_tensor_dict["robot_inertia"]  # 机器人的惯性矩阵
            self.gravity = global_tensor_dict["gravity"]  # 重力加速度
            
        if self.mode == "obstacle":
            # 如果模式为"obstacle"，则从字典中提取障碍物的状态信息
            self.robot_position = global_tensor_dict["obstacle_position"]  # 障碍物的位置
            self.robot_orientation = global_tensor_dict["obstacle_orientation"]  # 障碍物的朝向
            self.robot_linvel = global_tensor_dict["obstacle_linvel"]  # 障碍物的线速度
            self.robot_angvel = global_tensor_dict["obstacle_angvel"]  # 障碍物的角速度
            self.robot_vehicle_orientation = global_tensor_dict["obstacle_vehicle_orientation"]  # 障碍物的车辆朝向
            self.robot_vehicle_linvel = global_tensor_dict["obstacle_vehicle_linvel"]  # 障碍物的车辆线速度
            self.robot_body_angvel = global_tensor_dict["obstacle_body_angvel"]  # 障碍物的车身角速度
            self.robot_body_linvel = global_tensor_dict["obstacle_body_linvel"]  # 障碍物的车身线速度
            self.robot_euler_angles = global_tensor_dict["obstacle_euler_angles"]  # 障碍物的欧拉角
            self.mass = 1.0  # 假设障碍物的质量为1.0
            self.robot_inertia = torch.eye(3, device=self.device)  # 障碍物的惯性矩阵为单位矩阵
            self.gravity = global_tensor_dict["gravity"]  # 重力加速度
