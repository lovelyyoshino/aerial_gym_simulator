import time
import numpy as np

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.sim.sim_builder import SimBuilder

import torch
import numpy as np

from PIL import Image
import torch

if __name__ == "__main__":
    start = time.time()  # 记录程序开始时间
    seed = 0  # 设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为PyTorch设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设备设置随机种子

    rendering_benchmark = False  # 是否进行渲染基准测试的标志
    logger.warning(
        "This script provides an example of a rendering benchmark for the environment. The rendering benchmark will measure the FPS and the real-time speedup of the environment."
    )
    logger.warning(
        "\n\n\nThe rendering benchmark will run by default. Please set rendering_benchmark = False to run the physics benchmark. \n\n\n"
    )

    if rendering_benchmark == True:  # 如果选择进行渲染基准测试
        env_manager = SimBuilder().build_env(  # 构建环境管理器
            sim_name="base_sim",  # 模拟器名称
            env_name="env_with_obstacles",  # 环境名称，包含障碍物
            robot_name="base_quadrotor",  # 机器人名称
            controller_name="lee_velocity_control",  # 控制器名称
            args=None,  # 额外参数
            device="cuda:0",  # 使用的设备
            num_envs=16,  # 环境数量
            headless=True,  # 无头模式，不显示图形界面
            use_warp=True,  # 使用warp技术加速
        )
        if env_manager.robot_manager.robot.cfg.sensor_config.enable_camera == False:  # 检查相机是否启用
            logger.error(
                "The camera is disabled for this environment. The rendering benchmark will not work."
            )
            exit(1)  # 如果相机未启用，则退出程序
    else:  # 否则构建一个空环境
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="empty_env",  # 空环境
            robot_name="base_quadrotor",
            controller_name="no_control",  # 不使用控制器
            args=None,
            device="cuda:0",
            num_envs=256,  # 增加环境数量以提高性能评估
            headless=True,
            use_warp=True,
        )
        if env_manager.robot_manager.robot.cfg.sensor_config.enable_camera == True:  # 检查相机是否启用
            logger.critical(
                "The camera is enabled for this environment. This will cause the benchmark to be slower than expected. Please disable the camera for a more accurate benchmark."
            )

    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")  # 初始化动作张量，大小为(num_envs, 4)
    env_manager.reset()  # 重置环境
    elapsed_steps = -100  # 已经过的步骤计数初始化
    with torch.no_grad():  # 在不计算梯度的情况下执行以下代码块
        for i in range(10000):  # 循环10,000次
            # 允许模拟器预热一段时间，然后再测量时间
            if i == 100:
                start = time.time()  # 开始计时
                elapsed_steps = 0  # 重置已过步数
            env_manager.step(actions=actions)  # 执行一步仿真
            if rendering_benchmark == True:  # 如果是渲染基准测试
                env_manager.render(render_components="sensor")  # 渲染传感器组件
            elapsed_steps += 1  # 增加已过步数
            if i % 50 == 0:  # 每50步输出一次日志
                if i < 0:
                    logger.warning("Warming up....")  # 预热阶段警告
                else:
                    logger.critical(
                        f"i {elapsed_steps}, Current time: {time.time() - start}, FPS: {elapsed_steps * env_manager.num_envs / (time.time() - start)}, Real Time Speedup: {elapsed_steps * env_manager.num_envs * env_manager.sim_config.sim.dt / (time.time() - start)}"
                    )  # 输出当前步数、时间、FPS和实时加速比
    end = time.time()  # 记录结束时间
