from aerial_gym.config.sensor_config.lidar_config.os0_128_config import OS_0_128_Config

# 定义一个名为OS_1_64_Config的类，继承自OS_0_128_Config类
class OS_1_64_Config(OS_0_128_Config):
    # 将高度设置为64，表示传感器垂直方向上的光束数量
    height = 64

    # 设置水平视场角的最小值和最大值（单位：度）
    horizontal_fov_deg_min = -180  # 水平视场角最小值
    horizontal_fov_deg_max = 180    # 水平视场角最大值
    
    # 设置垂直视场角的最小值和最大值（单位：度）
    vertical_fov_deg_min = -22.5    # 垂直视场角最小值
    vertical_fov_deg_max = +22.5     # 垂直视场角最大值

    # 设置激光雷达的最大探测范围和最小探测范围（单位：米）
    max_range = 90.0                 # 最大探测范围
    min_range = 0.7                  # 最小探测范围

    # 定义一个内部类sensor_noise，用于配置传感器噪声参数
    class sensor_noise:
        enable_sensor_noise = False   # 是否启用传感器噪声
        pixel_dropout_prob = 0.01      # 像素丢失概率
        pixel_std_dev_multiplier = 0.01 # 像素标准差乘数
