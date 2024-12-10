from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
import numpy as np


class RsD455Config(BaseDepthCameraConfig):
    num_sensors = 1  # 传感器数量

    sensor_type = "camera"  # 传感器类型

    # 如果使用多个传感器，需要为每个传感器指定放置位置
    # 可以在这里添加，但用户可以根据需要自行实现。

    height = 270  # 相机图像高度
    width = 480   # 相机图像宽度
    horizontal_fov_deg = 87.000  # 水平视场角（以度为单位）
    
    # 相机参数，垂直视场角通过纵横比和水平视场角计算得出
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    max_range = 15.0  # 最大测量范围（米）
    min_range = 0.2   # 最小测量范围（米）

    # 相机类型（深度、范围、点云、分割）
    # 可以组合: (depth+segmentation), (range+segmentation), (pointcloud+segmentation)
    # 其他组合是简单的，如果你想，可以在代码中添加支持。

    calculate_depth = (
        True  # 获取深度图像而不是范围图像。如果设置为False，将返回范围图像
    )
    return_pointcloud = False  # 返回点云而不是图像。如果设置为True，上面的深度选项将被忽略
    pointcloud_in_world_frame = False  # 点云是否在世界坐标系下
    segmentation_camera = True  # 是否启用分割相机

    # 从传感器元素坐标系到传感器基座链接坐标系的变换
    euler_frame_rot_deg = [-90.0, 0, -90.0]  # 欧拉角旋转（以度为单位）

    # 要从传感器返回的数据类型
    normalize_range = True  # 当点云在世界坐标系时将设置为false

    # 不要更改此项。
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # 除以最大范围。当点云在世界坐标系时被忽略

    # 对于超出范围值的处理
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，则值范围为[-1]U[0,1]，否则为用户替代-1.0的值
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，则值范围为[-1]U[0,1]，否则为用户替代-1.0的值

    # 随机化传感器的位置
    randomize_placement = True  # 是否随机化传感器放置
    min_translation = [0.07, -0.06, 0.01]  # 最小平移范围
    max_translation = [0.12, 0.03, 0.04]   # 最大平移范围
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]  # 最小欧拉角旋转范围
    max_euler_rotation_deg = [5.0, 5.0, 5.0]      # 最大欧拉角旋转范围

    # 标称位置和方向（仅适用于Isaac Gym相机传感器）
    # 如果选择使用Isaac Gym传感器，它们的位置和方向将不会被随机化
    nominal_position = [0.10, 0.0, 0.03]  # 标称位置
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]  # 标称方向（以度为单位）

    use_collision_geometry = False  # 是否使用碰撞几何体

    class sensor_noise:
        enable_sensor_noise = False  # 是否启用传感器噪声
        pixel_dropout_prob = 0.01  # 像素丢失概率
        pixel_std_dev_multiplier = 0.01  # 像素标准差乘数
