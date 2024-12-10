from aerial_gym.config.sensor_config.lidar_config.os0_128_config import OS_0_128_Config

# 定义一个名为OS_0_64_Config的类，继承自OS_0_128_Config类
class OS_0_64_Config(OS_0_128_Config):
    # 将高度设置为64，保持其他配置与父类相同，只改变垂直光线的数量
    height = 64

    class sensor_noise:
        # 启用传感器噪声的标志，默认为False表示不启用
        enable_sensor_noise = False
        # 像素丢失概率，表示在每个像素中随机丢失的概率
        pixel_dropout_prob = 0.01
        # 像素标准差乘数，用于控制噪声强度
        pixel_std_dev_multiplier = 0.01
