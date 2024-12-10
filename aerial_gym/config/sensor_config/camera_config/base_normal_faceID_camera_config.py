from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
import numpy as np


class BaseNormalFaceIDCameraConfig(BaseDepthCameraConfig):
    num_sensors = 1  # 传感器数量

    sensor_type = "normal_faceID_camera"  # 传感器类型

    # 如果使用多个传感器，需要为每个传感器指定放置位置
    # 可以在这里添加，但用户也可以根据需要自行实现。

    height = 270  # 相机图像的高度（像素）
    width = 480   # 相机图像的宽度（像素）
    horizontal_fov_deg = 87.000  # 水平视场角（度）
    
    # 相机参数，垂直视场角通过纵横比和水平视场角计算得出
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    
    max_range = 10.0  # 最大探测范围（米）
    min_range = 0.2   # 最小探测范围（米）

    return_pointcloud = True  # 正常信息以点云形式返回

    normal_in_world_frame = True  # 法线是否在世界坐标系中表示
    # 从传感器元素坐标系到传感器基座链接框架的变换
    # euler_frame_rot_deg = [-90.0, 0, -90.0]

    # 随机化传感器的位置
    randomize_placement = False  # 是否随机放置传感器
    min_translation = [0.07, -0.06, 0.01]  # 最小平移量（米）
    max_translation = [0.12, 0.03, 0.04]   # 最大平移量（米）
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]  # 最小欧拉旋转角度（度）
    max_euler_rotation_deg = [5.0, 5.0, 5.0]     # 最大欧拉旋转角度（度）

    use_collision_geometry = False  # 是否使用碰撞几何体

    class sensor_noise:
        enable_sensor_noise = False  # 是否启用传感器噪声
        pixel_dropout_prob = 0.01  # 像素丢失概率
        pixel_std_dev_multiplier = 0.01  # 像素标准差乘数
