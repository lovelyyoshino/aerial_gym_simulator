from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，用于记录程序的运行信息
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

from aerial_gym.utils.math import quat_from_euler_xyz_tensor

from aerial_gym.config.sim_config.base_sim_no_gravity_config import BaseSimNoGravityConfig
from aerial_gym.registry.sim_registry import sim_config_registry

if __name__ == "__main__":
    # 注册基础无重力仿真配置到仿真配置注册表中
    sim_config_registry.register("base_sim_no_gravity", BaseSimNoGravityConfig)
    
    # 记录警告信息，说明示例演示了用于ROV（遥控水下机器人）的几何控制器的使用
    logger.warning("This example demonstrates the use of geometric controllers for a rov.")
    
    # 构建环境管理器，设置仿真参数
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_no_gravity",  # 仿真名称
        env_name="empty_env",             # 环境名称
        robot_name="base_rov",            # 机器人名称
        controller_name="fully_actuated_control",  # 控制器名称
        args=None,                        # 附加参数
        device="cuda:0",                 # 使用的设备（GPU）
        num_envs=64,                     # 环境数量
        headless=False,                  # 是否以无头模式运行
        use_warp=False,                  # 不使用相机，因为这个示例不需要
    )
    
    # 初始化动作张量，大小为(num_envs, 7)，并将其移动到CUDA设备上
    actions = torch.zeros((env_manager.num_envs, 7)).to("cuda:0")
    actions[:, 6] = 1.0  # 设置最后一列的值为1.0
    
    # 重置环境
    env_manager.reset()
    
    # 记录信息，说明该脚本提供了一个自定义几何位置控制器的示例
    logger.info(
        "\n\n\n\n\n\n This script provides an example of a custom geometric position controller for a custom BlueROV robot. \n\n\n\n\n\n"
    )
    
    # 主循环，进行10000个步骤的仿真
    for i in range(10000):
        if i % 500 == 0:
            # 每500步记录一次信息，并改变目标设定点
            logger.info(f"Step {i}, changing target setpoint.")
            
            # 随机生成新的目标位置（x, y, z），范围在[-2.0, 2.0]
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            
            # 随机生成新的目标姿态（四元数），通过从欧拉角转换得到
            actions[:, 3:7] = quat_from_euler_xyz_tensor(
                torch.pi * (torch.rand_like(actions[:, 3:6]) * 2 - 1)  # 欧拉角范围在[-π, π]
            )
        
        # 执行一步仿真，传入当前的动作
        env_manager.step(actions=actions)
