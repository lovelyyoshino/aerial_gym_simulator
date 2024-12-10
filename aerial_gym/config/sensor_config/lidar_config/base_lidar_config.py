from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig
import numpy as np

# 基础激光雷达配置类，继承自基本传感器配置类
class BaseLidarConfig(BaseSensorConfig):
    num_sensors = 1  # 激光雷达的数量

    sensor_type = "lidar"  # 传感器类型为激光雷达

    # 如果使用多个传感器，需要为每个传感器指定放置位置
    # 可以在这里添加，但用户也可以根据需要自行实现。

    # 标准OS0-128配置
    height = 128  # 垂直分辨率
    width = 512   # 水平分辨率
    horizontal_fov_deg_min = -180  # 水平视场角最小值（度）
    horizontal_fov_deg_max = 180    # 水平视场角最大值（度）
    vertical_fov_deg_min = -45       # 垂直视场角最小值（度）
    vertical_fov_deg_max = +45       # 垂直视场角最大值（度）

    # 最小和最大范围不与真实传感器匹配，但这里为了方便进行限制
    max_range = 10.0  # 最大探测距离（米）
    min_range = 0.2   # 最小探测距离（米）

    # 激光雷达的数据类型（范围、点云、分割）
    # 可以组合: (range+segmentation), (pointcloud+segmentation)
    return_pointcloud = (
        False  # 默认返回范围图像而不是点云
    )
    pointcloud_in_world_frame = False  # 点云是否在世界坐标系中
    segmentation_camera = True  # 设置为真将返回分割图像以及范围图像或点云

    # 从传感器元素坐标框架到传感器基座链接框架的变换
    euler_frame_rot_deg = [0.0, 0.0, 0.0]  # 欧拉角旋转（度）

    # 要从传感器返回的数据类型
    normalize_range = True  # 当点云在世界框架中时，将设置为假

    # 不要更改此项。
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # 除以max_range。当点云在世界框架中时被忽略

    # 对于超出范围的值的处理方式
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为真，则会是[-1]U[0,1]，否则将由用户替代-1.0的值
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为真，则会是[-1]U[0,1]，否则将由用户替代-1.0的值

    # 随机化传感器的位置
    randomize_placement = True  # 是否随机放置传感器
    min_translation = [0.07, -0.06, 0.01]  # 最小平移量
    max_translation = [0.12, 0.03, 0.04]   # 最大平移量
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]  # 最小欧拉角旋转（度）
    max_euler_rotation_deg = [5.0, 5.0, 5.0]      # 最大欧拉角旋转（度）

    # 名义位置和方向（仅适用于Isaac Gym相机传感器）
    nominal_position = [0.10, 0.0, 0.03]  # 名义位置（米）
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]  # 名义方向（欧拉角，度）

    class sensor_noise:
        enable_sensor_noise = False  # 是否启用传感器噪声
        pixel_dropout_prob = 0.01     # 像素丢失概率
        pixel_std_dev_multiplier = 0.01  # 像素标准差乘数
