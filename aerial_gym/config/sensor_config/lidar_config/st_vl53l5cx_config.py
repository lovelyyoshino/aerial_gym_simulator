from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig

# ST_VL53L5CXConfig类继承自BaseLidarConfig，用于配置ST VL53L5CX激光雷达传感器的参数
class ST_VL53L5CXConfig(BaseLidarConfig):
    num_sensors = 1  # 该类型传感器的数量
    sensor_type = "lidar"  # 传感器类型

    # 如果使用多个传感器，需要为每个传感器指定放置位置
    # 可以在这里添加，但用户也可以根据需要自行实现。

    height = 8  # 传感器高度
    width = 8   # 传感器宽度
    horizontal_fov_deg_min = -45  # 水平视场角最小值（度）
    horizontal_fov_deg_max = 45    # 水平视场角最大值（度）
    vertical_fov_deg_min = -45      # 垂直视场角最小值（度）
    vertical_fov_deg_max = +45      # 垂直视场角最大值（度）

    # 最小和最大范围与实际传感器不匹配，但这里为了方便进行限制
    max_range = 4.0  # 最大探测距离（米）
    min_range = 0.2  # 最小探测距离（米）

    # 激光雷达的数据类型（范围、点云、分割）
    # 可以组合: (范围+分割), (点云+分割)
    # 其他组合是简单的，如果你想，可以在代码中添加支持。

    return_pointcloud = (
        False  # 默认返回范围图像，而不是点云。如果设置为True，则返回点云。
    )
    pointcloud_in_world_frame = False  # 点云是否在世界坐标系下
    segmentation_camera = False  # 设置为True将同时返回分割图像和范围图像或点云

    # 从传感器元素坐标系到传感器基座链接坐标系的变换
    euler_frame_rot_deg = [0.0, 0.0, 0.0]  # 欧拉角旋转（度）

    # 要从传感器返回的数据类型
    normalize_range = False  # 当点云在世界坐标系时，将被设置为False

    # 不要更改此项。
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )  # 除以max_range。当点云在世界坐标系时被忽略

    # 对超出范围值的处理
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，值将在[-1]U[0,1]之间，否则用-1.0替代
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # 如果normalize_range为True，值将在[-1]U[0,1]之间，否则用-1.0替代

    # 随机化传感器的位置
    randomize_placement = False  # 是否随机化传感器放置
    min_translation = [0.07, -0.06, 0.01]  # 最小平移量
    max_translation = [0.12, 0.03, 0.04]   # 最大平移量
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]  # 最小欧拉角旋转（度）
    max_euler_rotation_deg = [5.0, 5.0, 5.0]     # 最大欧拉角旋转（度）

    # 标称位置和方向（仅适用于Isaac Gym相机传感器）
    nominal_position = [0.10, 0.0, 0.03]  # 标称位置
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]  # 标称方向（欧拉角）

    class sensor_noise:
        enable_sensor_noise = True  # 启用传感器噪声
        pixel_dropout_prob = 0.01  # 像素丢失概率
        pixel_std_dev_multiplier = 0.02  # 像素标准差乘数
