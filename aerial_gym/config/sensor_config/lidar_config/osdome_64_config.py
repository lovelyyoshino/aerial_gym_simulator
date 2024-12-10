from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig

# OSDome_64_Config类继承自BaseLidarConfig，用于配置一个特定的激光雷达传感器（Dome 64）
class OSDome_64_Config(BaseLidarConfig):
    # 激光雷达的高度，表示垂直方向上的光束数量
    height = 64  
    # 激光雷达的宽度，表示水平方向上采样点的数量
    width = 512  
    
    # 水平视场角最小值和最大值，以度为单位
    horizontal_fov_deg_min = -180  
    horizontal_fov_deg_max = 180   
    
    # 垂直视场角最小值和最大值，以度为单位
    vertical_fov_deg_min = 0       
    vertical_fov_deg_max = 90      
    
    # 激光雷达的最大探测范围（米）
    max_range = 20.0               
    # 激光雷达的最小探测范围（米）
    min_range = 0.5                

    # 是否返回点云数据
    return_pointcloud = False       
    # 是否启用分割相机
    segmentation_camera = True       

    # 随机化传感器的位置
    randomize_placement = False      
    # 传感器在x、y、z轴上的最小平移量
    min_translation = [0.0, 0.0, 0.0] 
    # 传感器在x、y、z轴上的最大平移量
    max_translation = [0.0, 0.0, 0.0] 
    
    # 前置安装的圆顶激光雷达的最小欧拉旋转角度
    min_euler_rotation_deg = [0.0, 0.0, 0.0]  
    # 前置安装的圆顶激光雷达的最大欧拉旋转角度
    max_euler_rotation_deg = [0.0, 0.0, 0.0]  

    class sensor_noise:
        # 是否启用传感器噪声
        enable_sensor_noise = False     
        # 像素丢失概率
        pixel_dropout_prob = 0.01        
        # 像素标准差乘数
        pixel_std_dev_multiplier = 0.01   
