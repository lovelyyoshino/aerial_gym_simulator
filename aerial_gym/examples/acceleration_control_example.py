from aerial_gym.utils.logging import CustomLogger

# 创建一个自定义日志记录器，使用当前模块的名称
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

if __name__ == "__main__":
    # 记录调试信息
    logger.debug("this is how a debug message looks like")
    # 记录一般信息
    logger.info("this is how an info message looks like")
    # 记录警告信息
    logger.warning("this is how a warning message looks like")
    # 记录错误信息
    logger.error("this is how an error message looks like")
    # 记录严重错误信息
    logger.critical("this is how a critical message looks like")
    
    # 提供脚本功能说明的信息
    logger.info(
        "\n\n\n\n\n\n This script provides an example of a robot with constant forward acceleration directly input to the environment. \n\n\n\n\n\n"
    )
    
    # 构建环境管理器，创建模拟环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",  # 模拟器名称
        env_name="env_with_obstacles",  # 环境名称（带障碍物）
        robot_name="base_quadrotor",  # 机器人名称（四旋翼）
        controller_name="lee_acceleration_control",  # 控制器名称
        args=None,  # 附加参数
        num_envs=16,  # 环境数量
        device="cuda:0",  # 使用的设备（GPU）
        headless=False,  # 是否以无头模式运行
        use_warp=True,  # 启用warp模式，以在没有对象时禁用相机
    )
    
    # 初始化动作张量，大小为(num_envs, 4)，并将其移动到CUDA设备上
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 0] = 0.25  # 设置常量前向加速度
    
    # 重置环境
    env_manager.reset()
    
    # 主循环，执行1000次步骤
    for i in range(1000):
        if i % 100 == 0:
            print("i", i)  # 每100步打印一次当前步数
            env_manager.reset()  # 重置环境
        
        # 执行一步操作
        env_manager.step(actions=actions)
        
        # 渲染当前环境状态
        env_manager.render()
        
        # 重置已终止和截断的环境
        env_manager.reset_terminated_and_truncated_envs()
