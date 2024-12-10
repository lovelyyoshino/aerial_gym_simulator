import warp as wp
from aerial_gym.sensors.base_sensor import BaseSensor

from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    tf_apply,
    torch_rand_float_tensor,
    quat_from_euler_xyz_tensor,
)

import torch

from aerial_gym.sensors.warp.warp_cam import WarpCam
from aerial_gym.sensors.warp.warp_lidar import WarpLidar
from aerial_gym.sensors.warp.warp_normal_faceID_cam import WarpNormalFaceIDCam
from aerial_gym.sensors.warp.warp_normal_faceID_lidar import WarpNormalFaceIDLidar

from aerial_gym.utils.logging import CustomLogger, logging

logger = CustomLogger("WarpSensor")
logger.setLoggerLevel(logging.INFO)

# WarpSensor类继承自BaseSensor类，负责初始化和管理传感器（如相机和激光雷达）的状态和数据。
class WarpSensor(BaseSensor):
    def __init__(self, sensor_config, num_envs, mesh_id_list, device):
        # 调用父类构造函数初始化基本传感器设置
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.mesh_id_list = mesh_id_list  # 传感器所使用的网格ID列表
        self.device = device  # 设备类型（CPU/GPU）
        self.num_sensors = self.cfg.num_sensors  # 传感器数量

        # 将mesh_id_list转换为warp数组
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)

        # 根据传感器类型实例化相应的传感器
        if self.cfg.sensor_type == "lidar":
            self.sensor = WarpLidar(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Lidar sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "camera":
            self.sensor = WarpCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "normal_faceID_lidar":
            self.sensor = WarpNormalFaceIDLidar(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Normal FaceID Lidar sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "normal_faceID_camera":
            self.sensor = WarpNormalFaceIDCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Normal FaceID Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        else:
            raise NotImplementedError

    def init_tensors(self, global_tensor_dict):
        # 初始化传感器的张量
        super().init_tensors(global_tensor_dict)
        logger.debug(f"Initializing sensor tensors")
        # 这里为机器人位置和方向创建新的视图，因为机器人有多个传感器
        self.robot_position = self.robot_position.unsqueeze(1).expand(-1, self.num_sensors, -1)
        self.robot_orientation = self.robot_orientation.unsqueeze(1).expand(
            -1, self.num_sensors, -1
        )

        # 为传感器的最小和最大平移初始化张量
        self.sensor_min_translation = torch.tensor(
            self.cfg.min_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_translation = torch.tensor(
            self.cfg.max_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_min_rotation = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_rotation = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)

        # 计算传感器框架的旋转
        euler_sensor_frame_rot = self.cfg.euler_frame_rot_deg
        sensor_frame_rot_rad = torch.deg2rad(
            torch.tensor(euler_sensor_frame_rot, device=self.device, requires_grad=False)
        )
        sensor_quat = quat_from_euler_xyz_tensor(sensor_frame_rot_rad)
        self.sensor_data_frame_quat = sensor_quat.expand(self.num_envs, self.num_sensors, -1)

        # 初始化传感器局部位置和方向
        self.sensor_local_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation[..., 3] = 1.0  # 单位四元数的w分量

        # 计算传感器局部方向的均值
        mean_euler_rotation = (self.sensor_min_rotation + self.sensor_max_rotation) / 2.0
        self.sensor_local_orientation[:] = quat_from_euler_xyz(
            mean_euler_rotation[..., 0],
            mean_euler_rotation[..., 1],
            mean_euler_rotation[..., 2],
        )

        # 初始化传感器的全局位置和方向
        self.sensor_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation[..., 3] = 1.0  # 单位四元数的w分量

        # 设置传感器的位姿和图像张量
        self.sensor.set_pose_tensor(
            positions=self.sensor_position, orientations=self.sensor_orientation
        )
        self.sensor.set_image_tensors(
            pixels=self.pixels, segmentation_pixels=self.segmentation_pixels
        )
        self.reset()

        logger.debug(f"[DONE] Initializing sensor tensors")

    def reset(self):
        # 重置传感器状态
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        if self.cfg.randomize_placement == True:
            # 从最小和最大平移中随机采样局部位置
            self.sensor_local_position[env_ids] = torch_rand_float_tensor(
                self.sensor_min_translation[env_ids],
                self.sensor_max_translation[env_ids],
            )
            # 从最小和最大旋转中随机采样局部方向
            local_euler_rotation = torch_rand_float_tensor(
                self.sensor_min_rotation[env_ids], self.sensor_max_rotation[env_ids]
            )
            self.sensor_local_orientation[env_ids] = quat_from_euler_xyz(
                local_euler_rotation[..., 0],
                local_euler_rotation[..., 1],
                local_euler_rotation[..., 2],
            )
        else:
            # 不进行任何操作
            pass
        return

    def initialize_sensor(self):
        # 初始化传感器并捕获数据
        self.sensor.capture()

    def update(self):
        # 在执行射线投射之前，将局部位置和方向转换到世界坐标系
        self.sensor_position[:] = tf_apply(
            self.robot_orientation, self.robot_position, self.sensor_local_position
        )
        self.sensor_orientation[:] = quat_mul(
            self.robot_orientation,
            quat_mul(self.sensor_local_orientation, self.sensor_data_frame_quat),
        )

        logger.debug(
            f"Sensor position: {self.sensor_position[0]}, Sensor orientation: {self.sensor_orientation[0]}"
        )

        logger.debug("Capturing sensor data")
        self.sensor.capture()
        logger.debug("[DONE] Capturing sensor data")

        # 应用噪声和范围限制
        self.apply_noise()
        if self.cfg.sensor_type in ["camera", "lidar"]:
            self.apply_range_limits()
            self.normalize_observation()

    def apply_range_limits(self):
        # 应用传感器范围限制
        if self.cfg.return_pointcloud == True:
            # 如果点云在世界坐标系中，则不进行归一化
            if self.cfg.pointcloud_in_world_frame == False:
                logger.debug("Pointcloud is not in world frame")
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    > self.cfg.max_range
                ] = self.cfg.far_out_of_range_value
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    < self.cfg.min_range
                ] = self.cfg.near_out_of_range_value
                logger.debug("[DONE] Clipping pointcloud values to sensor range")
        else:
            logger.debug("Pointcloud is in world frame")
            self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
            self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value
            logger.debug("[DONE] Clipping pointcloud values to sensor range")

    def normalize_observation(self):
        # 归一化观察值
        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            logger.debug("Normalizing pointcloud values")
            self.pixels[:] = self.pixels / self.cfg.max_range
        if self.cfg.pointcloud_in_world_frame == True:
            logger.debug("Pointcloud is in world frame. not normalizing")

    def apply_noise(self):
        # 应用传感器噪声
        if self.cfg.sensor_noise.enable_sensor_noise == True:
            logger.debug("Applying sensor noise")
            self.pixels[:] = torch.normal(
                mean=self.pixels,
                std=self.cfg.sensor_noise.pixel_std_dev_multiplier * self.pixels,
            )
            self.pixels[
                torch.bernoulli(
                    torch.ones_like(self.pixels) * self.cfg.sensor_noise.pixel_dropout_prob
                )
                > 0
            ] = self.cfg.near_out_of_range_value

    def get_observation(self):
        # 获取传感器的观察值
        return self.pixels, self.segmentation_pixels
