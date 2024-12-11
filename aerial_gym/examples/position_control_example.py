from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，用于记录程序的运行信息
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    
    # 记录警告信息，说明该示例演示了四旋翼的几何控制器使用
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    
    # 构建环境管理器，设置仿真名称、环境名称、机器人名称和控制器名称等参数
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",                # 仿真名称
        env_name="empty_env",               # 环境名称
        robot_name="base_quadrotor",        # 机器人名称
        controller_name="lee_position_control",  # 控制器名称
        args=None,                          # 附加参数（此处为None）
        device="cuda:0",                   # 使用的设备（GPU）
        num_envs=args.num_envs,            # 环境数量，从命令行参数获取
        headless=args.headless,            # 是否无头模式，从命令行参数获取
        use_warp=args.use_warp,            # 是否使用warp功能，从命令行参数获取
    )
    
    # 初始化动作张量，形状为(num_envs, 4)，并将其移动到CUDA设备上
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    
    # 重置环境
    env_manager.reset()
    
    # 主循环，进行10000个时间步的仿真
    for i in range(10000):
        if i % 1000 == 0:
            # 每1000步记录一次信息，并改变目标设定点
            logger.info(f"Step {i}, changing target setpoint.")
            # 以下两行代码被注释掉，原本用于随机生成新的目标位置和角度
            # actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            # actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            
            # 重置环境以重新开始
            env_manager.reset()
        
        # 执行动作步骤，将当前动作传递给环境管理器
        env_manager.step(actions=actions)
