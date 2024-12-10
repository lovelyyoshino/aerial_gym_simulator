from aerial_gym.config.sensor_config.camera_config.luxonis_oak_d_config import (
    LuxonisOakDConfig,
)

# 定义LuxonisOakDProWConfig类，继承自LuxonisOakDConfig类
class LuxonisOakDProWConfig(LuxonisOakDConfig):
    # 相机的高度设置为270像素
    height = 270
    # 相机的宽度设置为480像素
    width = 480
    # 水平视场角（HFOV）设置为127.0度
    horizontal_fov_deg = 127.0
    # 相机参数：垂直视场角（VFOV）通过纵横比和水平视场角计算得出
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    
    # 最大探测范围设置为12.0米
    max_range = 12.0
    # 最小探测范围设置为0.2米
    min_range = 0.2
