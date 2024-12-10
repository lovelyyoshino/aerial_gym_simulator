from aerial_gym.sensors.base_sensor import BaseSensor

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacGymCameraSensor")


class IsaacGymCameraSensor(BaseSensor):
    """
    Camera sensor class for Isaac Gym. Inherits from BaseSensor.
    Supports depth and semantic segmentation images. Color image support is not yet implemented.
    """

    def __init__(self, sensor_config, num_envs, gym, sim, device):
        """
        初始化相机传感器的参数和配置。

        Args:
        - sensor_config: 传感器的配置参数
        - num_envs: 环境数量
        - gym: gym的实例
        - sim: 模拟器的实例
        - device: 设备信息（CPU或GPU）

        Returns:
        - None
        """
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.device = device
        self.num_envs = num_envs
        self.cfg = sensor_config
        self.gym = gym
        self.sim = sim
        logger.warning("Initializing Isaac Gym Camera Sensor")
        logger.debug(f"Camera sensor config: {self.cfg.__dict__}")
        self.init_cam_config()  # 初始化相机配置
        self.depth_tensors = []  # 深度图像张量列表
        self.segmentation_tensors = []  # 语义分割图像张量列表
        self.color_tensors = []  # 颜色图像张量列表（尚未实现）
        self.cam_handles = []  # 相机句柄列表

    def init_cam_config(self):
        """
        初始化相机的属性和本地变换。使用配置文件中的参数。

        Args:
        - None

        Returns:
        - None
        """
        # 设置相机属性
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True  # 启用张量
        camera_props.width = self.cfg.width  # 相机宽度
        camera_props.height = self.cfg.height  # 相机高度
        camera_props.far_plane = self.cfg.max_range  # 最大范围
        camera_props.near_plane = self.cfg.min_range  # 最小范围
        camera_props.horizontal_fov = self.cfg.horizontal_fov_deg  # 水平视场角
        camera_props.use_collision_geometry = self.cfg.use_collision_geometry  # 使用碰撞几何体
        self.camera_properties = camera_props  # 保存相机属性

        # 本地变换
        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(
            self.cfg.nominal_position[0],
            self.cfg.nominal_position[1],
            self.cfg.nominal_position[2],
        )
        angle_euler = torch.deg2rad(
            torch.tensor(
                self.cfg.nominal_orientation_euler_deg,
                device=self.device,
                requires_grad=False,
            )
        )
        angle_quat = quat_from_euler_xyz(angle_euler[0], angle_euler[1], angle_euler[2])  # 欧拉角转四元数
        self.local_transform.r = gymapi.Quat(
            angle_quat[0], angle_quat[1], angle_quat[2], angle_quat[3]
        )

    def add_sensor_to_env(self, env_id, env_handle, actor_handle):
        """
        将相机传感器添加到环境中。为每个相机传感器设置适当的属性，并将其附加到actor。

        Args:
        - env_id: 环境ID
        - env_handle: 环境句柄
        - actor_handle: actor句柄

        Returns:
        - None
        """
        logger.debug(f"Adding camera sensor to env {env_handle} and actor {actor_handle}")
        if len(self.cam_handles) == env_id:  # 检查是否为新环境
            self.depth_tensors.append([])  # 添加深度张量
            self.segmentation_tensors.append([])  # 添加分割张量
        self.cam_handle = self.gym.create_camera_sensor(env_handle, self.camera_properties)  # 创建相机传感器
        self.cam_handles.append(self.cam_handle)  # 保存相机句柄
        self.gym.attach_camera_to_body(
            self.cam_handle,
            env_handle,
            actor_handle,
            self.local_transform,
            gymapi.FOLLOW_TRANSFORM,  # 相机跟随actor的变换
        )
        # 获取深度和分割图像的张量
        self.depth_tensors[env_id].append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.cam_handle, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.segmentation_tensors[env_id].append(
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, self.cam_handle, gymapi.IMAGE_SEGMENTATION
                )
            )
        )
        logger.debug(f"Camera sensor added to env {env_handle} and actor {actor_handle}")

    def init_tensors(self, global_tensor_dict):
        """
        初始化相机传感器的张量。深度张量是必需的，语义张量是可选的。

        Args:
        - global_tensor_dict: 全局张量字典

        Returns:
        - None
        """
        super().init_tensors(global_tensor_dict)  # 调用父类的初始化方法

        # RGB相机支持将在将来添加。请在此之前使用Isaac Gym的原生RGB相机。
        # self.color_tensors = global_tensor_dict["color_tensor"]

    def capture(self):
        """
        触发传感器捕获图像。在fetch_results运行后执行此操作。
        随后，图像必须分别存储在相关的张量切片中。
        """
        self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机传感器
        self.gym.start_access_image_tensors(self.sim)  # 开始访问图像张量
        for env_id in range(self.num_envs):
            for cam_id in range(self.cfg.num_sensors):
                # 深度值在-z轴，因此需要翻转为正值
                self.pixels[env_id, cam_id] = -self.depth_tensors[env_id][cam_id]
                if self.cfg.segmentation_camera:  # 如果启用分割相机
                    self.segmentation_pixels[env_id, cam_id] = self.segmentation_tensors[env_id][
                        cam_id
                    ]
        self.gym.end_access_image_tensors(self.sim)  # 结束访问图像张量

    def update(self):
        """
        更新相机传感器。捕获图像，应用与其他相机相同的后处理。
        深度张量中的值设置为可接受的限制，并在需要时进行归一化。
        """
        self.capture()  # 捕获图像
        self.apply_noise()  # 应用噪声
        self.apply_range_limits()  # 应用范围限制
        self.normalize_observation()  # 归一化观察值

    def apply_range_limits(self):
        """
        应用深度范围限制，确保深度值在设置的最大和最小范围之间。
        """
        logger.debug("Applying range limits")
        self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value  # 超过最大范围的值设为远离范围值
        self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value  # 低于最小范围的值设为接近范围值
        logger.debug("[DONE] Applying range limits")

    def normalize_observation(self):
        """
        归一化观察值。如果配置要求且点云不在世界坐标系中，则进行归一化。
        """
        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            logger.debug("Normalizing pointcloud values")
            self.pixels[:] = self.pixels / self.cfg.max_range  # 归一化点云值
        if self.cfg.pointcloud_in_world_frame == True:
            logger.error("Pointcloud is in world frame. Not supported for this sensor")

    def apply_noise(self):
        """
        应用传感器噪声。如果启用传感器噪声，则对像素值添加噪声。
        """
        if self.cfg.sensor_noise.enable_sensor_noise == True:
            logger.debug("Applying sensor noise")
            self.pixels[:] = torch.normal(
                mean=self.pixels, std=self.cfg.pixel_std_dev_multiplier * self.pixels
            )  # 添加高斯噪声
            self.pixels[
                torch.bernoulli(torch.ones_like(self.pixels) * self.cfg.pixel_dropout_prob) > 0
            ] = self.cfg.near_out_of_range_value  # 应用丢失概率
     
    def reset_idx(self, env_ids):
        """
        重置指定env_ids的相机姿势。对于Isaac Gym的相机传感器无需执行任何操作。
        """
        # 对于Isaac Gym的相机传感器，无需执行任何操作
        pass

    def reset(self):
        """
        重置所有环境的相机姿势。对于Isaac Gym的相机传感器无需执行任何操作。
        """
        # 对于Isaac Gym的相机传感器，无需执行任何操作
        pass

    def get_observation(self):
        """
        获取当前的观察值，包括深度和语义分割图像。

        Returns:
        - pixels: 深度图像张量
        - segmentation_pixels: 语义分割图像张量
        """
        return self.pixels, self.segmentation_pixels
