from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，用于记录程序运行中的信息
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

from aerial_gym.sim import sim_config_registry
from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimConfig

from tqdm import tqdm
import time

if __name__ == "__main__":
    # 注册基础的无重力仿真配置
    sim_config_registry.register("base_sim_no_gravity", BaseSimConfig)

    # 构建环境管理器，设置仿真参数
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",  # 仿真名称
        env_name="empty_env",             # 环境名称
        robot_name="base_quadrotor",      # 机器人名称
        controller_name="lee_velocity_control",  # 控制器名称
        args=None,                        # 附加参数
        device="cuda:0",                 # 使用的设备（GPU）
        num_envs=2,                      # 环境数量
        headless=True,                   # 是否以无头模式运行
    )
    
    # 检查IMU传感器是否启用，如果未启用则记录错误并退出
    if env_manager.robot_manager.robot.cfg.sensor_config.enable_imu == False:
        logger.error(
            "The IMU is disabled for this environment. The IMU data collection will not work."
        )
        exit(1)
        
    # 获取全局张量字典
    tensor_dict = env_manager.global_tensor_dict
    
    # 初始化动作张量，大小为(num_envs, 4)，用于存储控制指令
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    
    # 重置环境
    env_manager.reset()
    
    start_time = time.time()  # 记录开始时间
    
    # f = open("simulated_imu_data.csv", "w")  # 打开文件以写入模拟的IMU数据（注释掉了）
    
    # 获取仿真的时间步长
    sim_dt = env_manager.sim_config.sim.dt
    
    # 循环进行仿真，持续3小时，每次步进0.005秒
    for i in range(int(3.0 * 3600 / 0.005)):
        # 执行一步仿真，传入动作
        env_manager.step(actions=actions)
        
        # 从全局张量字典中获取IMU测量值，并转换为numpy数组
        imu_measurement = tensor_dict["imu_measurement"][0].cpu().numpy()
        
        # 打印IMU测量值
        print(imu_measurement)
        
        # 将IMU测量值写入文件（注释掉了）
        # f.write(
        #     f"{i*sim_dt},{imu_measurement[0]},{imu_measurement[1]},{imu_measurement[2]},\
        #         {imu_measurement[3]},{imu_measurement[4]},{imu_measurement[5]}\n"
        # )
