from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，使用当前模块的名称
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    # 警告信息：动态环境可能会显著降低仿真速度，请谨慎使用。
    logger.warning(
        "\n\n\nWhile possible, a dynamic environment will slow down the simulation by a lot. Use with caution. Native Isaac Gym cameras work faster than Warp in this case.\n\n\n"
    )
    
    # 获取命令行参数
    args = get_args()
    
    # 构建仿真环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",  # 仿真名称
        env_name="dynamic_env",  # 环境名称
        robot_name="lmf2",  # 机器人名称
        controller_name="lmf2_position_control",  # 控制器名称
        args=None,  # 附加参数（此处为None）
        device="cuda:0",  # 使用的设备（GPU）
        num_envs=args.num_envs,  # 环境数量
        headless=args.headless,  # 是否无头模式
        use_warp=args.use_warp,  # 是否使用Warp技术
    )
    
    # 初始化动作张量，形状为(num_envs, 4)，并将其移动到CUDA设备上
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    
    # 重置环境
    env_manager.reset()
    
    # 计算环境中的资产数量（减去1是因为机器人也是一个资产）
    num_assets_in_env = (
        env_manager.IGE_env.num_assets_per_env - 1
    )  
    print(f"Number of assets in the environment: {num_assets_in_env}")
    
    # 获取环境数量
    num_envs = env_manager.num_envs

    # 初始化资产运动状态张量，形状为(num_envs, num_assets_in_env, 6)
    asset_twist = torch.zeros(
        (num_envs, num_assets_in_env, 6), device="cuda:0", requires_grad=False
    )
    
    # 设置第一个维度的初始值为-1.0
    asset_twist[:, :, 0] = -1.0
    
    # 主循环，进行10000次迭代
    for i in range(10000):
        if i % 1000 == 0:
            # 每1000步打印一次信息，并随机改变目标设定点
            logger.info(f"Step {i}, changing target setpoint.")
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)  # 随机生成新的位置目标
            actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)  # 随机生成新的角度目标
            
            # env_manager.reset()  # 此行代码被注释掉，不重置环境
        
        # 更新资产的运动状态，设置sin和cos函数以产生周期性变化
        asset_twist[:, :, 0] = torch.sin(0.2 * i * torch.ones_like(asset_twist[:, :, 0]))  # x轴方向的速度
        asset_twist[:, :, 1] = torch.cos(0.2 * i * torch.ones_like(asset_twist[:, :, 1]))  # y轴方向的速度
        asset_twist[:, :, 2] = 0.0  # z轴方向的速度保持为0
        
        # 执行一步仿真，传入动作和资产运动状态
        env_manager.step(actions=actions, env_actions=asset_twist)
