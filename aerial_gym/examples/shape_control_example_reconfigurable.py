from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，使用当前模块的名称
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt

from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimNoGravityConfig
from aerial_gym.registry.sim_registry import sim_config_registry

if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    
    # 输出警告信息，说明示例的功能
    logger.warning(
        "\n\n\nThis example demonstrates shape control of a reconfigurable robot with joint angle setpoints. Motor control for this robot is not implemented.\n\n\n"
    )
    
    # 设置仿真时间步长为0.002秒
    BaseSimNoGravityConfig.sim.dt = 0.002
    
    # 打印重力设置（应为0，因为是无重力仿真）
    print(BaseSimNoGravityConfig.sim.gravity)
    
    # 注册仿真配置到注册表中
    sim_config_registry.register("base_sim_no_gravity_2ms", BaseSimNoGravityConfig)
    
    # 构建环境管理器，创建一个名为"empty_env_2ms"的空环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity_2ms",
        env_name="empty_env_2ms",
        robot_name="snakey",
        controller_name="no_control",
        args=None,
        device="cuda:0",  # 使用CUDA设备进行加速
        num_envs=16,      # 创建16个并行环境
        headless=args.headless,  # 是否以无头模式运行
        use_warp=args.use_warp,  # 是否使用warp技术
    )
    
    # 初始化动作张量，大小为(16, 4)，并将前3列设为0
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, [0, 1, 2]] = 0.0
    
    # 重置环境
    env_manager.reset()

    # 主循环，执行10000次迭代
    for i in range(10000):
        # 每200步改变目标形状
        if i % 200 == 0:
            logger.info(f"Step {i}, changing target shape.")
            env_manager.reset()  # 重置环境
            
            # 生成新的关节位置目标，范围在-1到1之间
            dof_pos = 2*(torch.ones((env_manager.num_envs, 6)).to("cuda:0") - 0.5)
            
            # 设置机器人的关节速度目标
            env_manager.robot_manager.robot.set_dof_velocity_targets((3.14159 / 5.0) * dof_pos)
        
        # 执行一步仿真，传入动作
        env_manager.step(actions=actions)
