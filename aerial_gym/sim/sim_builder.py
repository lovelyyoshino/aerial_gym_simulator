from aerial_gym.env_manager.env_manager import EnvManager

import torch


class SimBuilder:
    def __init__(self):
        # 初始化模拟构建器，设置环境和机器人相关的属性
        self.sim_name = None  # 模拟名称
        self.env_name = None  # 环境名称
        self.robot_name = None  # 机器人名称
        self.env = None  # 环境实例
        pass

    def delete_env(self):
        # 删除环境的垃圾清理
        del self.env  # 删除环境实例
        # 确保所有CUDA内存被释放
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA
        self.env = None  # 将环境实例设置为None

    def build_env(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        # 构建环境的函数
        # sim_name: 模拟名称
        # env_name: 环境名称
        # robot_name: 机器人名称
        # controller_name: 控制器名称
        # device: 设备（如GPU或CPU）
        # args: 其他参数（可选）
        # num_envs: 环境数量（可选）
        # use_warp: 是否使用warp技术（可选）
        # headless: 是否以无头模式运行（可选）

        self.sim_name = sim_name  # 设置模拟名称
        self.env_name = env_name  # 设置环境名称
        self.robot_name = robot_name  # 设置机器人名称
        # 创建EnvManager实例，管理环境的创建和控制，这个会调用env_manager
        self.env = EnvManager(
            sim_name=sim_name,
            env_name=env_name,
            robot_name=robot_name,
            controller_name=controller_name,
            args=args,
            device=device,
            num_envs=num_envs,
            use_warp=use_warp,
            headless=headless,
        )
        return self.env  # 返回创建的环境实例
