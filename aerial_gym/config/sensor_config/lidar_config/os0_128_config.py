from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig
import numpy as np


class OS_0_128_Config(BaseLidarConfig):
    # 标准的OS0-128配置
    height = 128  # 激光雷达图像的高度（以像素为单位）
    width = 512   # 激光雷达图像的宽度（以像素为单位）
    
    horizontal_fov_deg_min = -180  # 水平视场角最小值（度）
    horizontal_fov_deg_max = 180    # 水平视场角最大值（度）
    vertical_fov_deg_min = -45      # 垂直视场角最小值（度）
    vertical_fov_deg_max = +45      # 垂直视场角最大值（度）

    # 最小和最大范围与真实传感器不匹配，但这里为了方便进行限制
    max_range = 35.0  # 最大探测距离（米）
    min_range = 0.2   # 最小探测距离（米）

    # 激光雷达类型（范围、点云、分割）
    # 可以组合使用: (range+segmentation), (pointcloud+segmentation)
    return_pointcloud = (
        False  # 默认返回范围图像而不是点云
    )
    pointcloud_in_world_frame = False  # 点云是否在世界坐标系中
    segmentation_camera = True  # 设置为True将同时返回分割图像和范围图像或点云

    # 从传感器元素坐标框架到传感器基座链接框架的变换
    euler_frame_rot_deg = [0.0, 0.0, 0.0]  # 欧拉角旋转（度）

    # 要从传感器返回的数据类型
    normalize_range = True  # 当点云在世界框架中时，将设置为False

    # 不要更改此项。
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # 除以max_range。当点云在世界框架中时被忽略

    # 对于超出范围的值该如何处理
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，则将是[-1]U[0,1]，否则将是用户替代-1.0设置的值
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，则将是[-1]U[0,1]，否则将是用户替代-1.0设置的值

    class sensor_noise:
        enable_sensor_noise = False  # 是否启用传感器噪声
        pixel_dropout_prob = 0.01     # 像素丢失概率
        pixel_std_dev_multiplier = 0.01  # 像素标准差乘数
