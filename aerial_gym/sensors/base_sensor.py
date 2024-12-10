from abc import ABC, abstractmethod
import math


class BaseSensor(ABC):
    def __init__(self, sensor_config, num_envs, device):
        """
        初始化传感器基类
        
        参数:
        sensor_config: 传感器配置，包括传感器类型等信息
        num_envs: 环境数量，表示同时运行的环境实例数
        device: 设备信息，例如 CPU 或 GPU
        """
        self.cfg = sensor_config  # 保存传感器配置
        self.device = device  # 保存设备信息
        self.num_envs = num_envs  # 保存环境数量
        self.robot_position = None  # 机器人位置（初始化为 None）
        self.robot_orientation = None  # 机器人朝向（初始化为 None）
        self.robot_linvel = None  # 机器人线速度（初始化为 None）
        self.robot_angvel = None  # 机器人角速度（初始化为 None）

    @abstractmethod
    def init_tensors(self, global_tensor_dict):
        """
        初始化传感器的张量
        
        参数:
        global_tensor_dict: 全局张量字典，包含机器人状态和传感器相关信息
        
        重要逻辑:
        该方法用于根据传感器类型初始化不同的张量，包括机器人的位置、朝向、重力、时间步长等。
        对于不同类型的传感器（如激光雷达、相机、IMU等），会从全局字典中提取相应的数据。
        """
        # 对于变形传感器
        self.robot_position = global_tensor_dict["robot_position"]  # 获取机器人位置
        self.robot_orientation = global_tensor_dict["robot_orientation"]  # 获取机器人朝向

        # 对于IMU传感器
        self.gravity = global_tensor_dict["gravity"]  # 获取重力
        self.dt = global_tensor_dict["dt"]  # 获取时间步长
        self.sqrt_dt = math.sqrt(self.dt)  # 计算时间步长的平方根
        self.robot_masses = global_tensor_dict["robot_mass"]  # 获取机器人质量

        if self.cfg.sensor_type in ["lidar", "camera"]:
            # 对于IGE和变形传感器
            self.pixels = global_tensor_dict["depth_range_pixels"]  # 获取深度范围像素
            if self.cfg.segmentation_camera:
                self.segmentation_pixels = global_tensor_dict["segmentation_pixels"]  # 获取分割像素
            else:
                self.segmentation_pixels = None  # 如果没有分割像素，则设置为 None
        elif self.cfg.sensor_type in ["normal_faceID_lidar", "normal_faceID_camera"]:
            self.pixels = global_tensor_dict["depth_range_pixels"]  # 获取深度范围像素
            self.segmentation_pixels = global_tensor_dict["segmentation_pixels"]  # 获取分割像素
        else:
            # 对于IMU传感器（可能用于运动模糊）
            self.robot_linvel = global_tensor_dict["robot_linvel"]  # 获取机器人线速度
            self.robot_angvel = global_tensor_dict["robot_angvel"]  # 获取机器人角速度
            self.robot_body_angvel = global_tensor_dict["robot_body_angvel"]  # 获取机器人身体角速度
            self.robot_body_linvel = global_tensor_dict["robot_body_linvel"]  # 获取机器人身体线速度
            self.robot_euler_angles = global_tensor_dict["robot_euler_angles"]  # 获取机器人欧拉角

    @abstractmethod
    def update(self):
        """
        更新传感器状态
        
        抛出:
        NotImplementedError: 如果没有实现该方法
        """
        raise NotImplementedError("update not implemented")

    @abstractmethod
    def reset_idx(self):
        """
        重置传感器索引
        
        抛出:
        NotImplementedError: 如果没有实现该方法
        """
        raise NotImplementedError("reset_idx not implemented")

    @abstractmethod
    def reset(self):
        """
        重置传感器状态
        
        抛出:
        NotImplementedError: 如果没有实现该方法
        """
        raise NotImplementedError("reset not implemented")

    @staticmethod
    def print_params(self):
        """
        打印传感器参数
        
        参数:
        self: 传感器实例
        
        重要逻辑:
        遍历实例的属性，并打印每个属性的名称及其类型。如果属性有 dtype 字段，则也打印 dtype。
        """
        for name, value in vars(self).items():
            # 如果dtype是有效字段，打印它
            if hasattr(value, "dtype"):
                print(name, type(value), value.dtype)  # 打印属性名称、类型及其数据类型
            else:
                print(name, type(value))  # 打印属性名称及其类型
