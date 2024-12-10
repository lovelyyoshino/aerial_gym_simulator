from abc import ABC, abstractmethod

from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("BaseRobot")


class BaseRobot(ABC):
    """
    基础类，用于表示空中机器人。此类应被具体的机器人类继承。
    """

    def __init__(self, robot_config, controller_name, env_config, device):
        """
        初始化BaseRobot类的实例。

        参数:
        robot_config: 机器人配置，包含机器人相关的参数。
        controller_name: 控制器的名称，用于指定所使用的控制器。
        env_config: 环境配置，包含环境相关的参数。
        device: 设备信息，指定运行设备（如CPU或GPU）。
        """
        self.cfg = robot_config  # 保存机器人配置
        self.num_envs = env_config.env.num_envs  # 获取环境数量
        self.device = device  # 保存设备信息

        # 创建控制器并获取控制器配置
        self.controller, self.controller_config = controller_registry.make_controller(
            controller_name,
            self.num_envs,
            self.device,
        )
        logger.info("[DONE] Initializing controller")  # 日志记录控制器初始化完成

        # 初始化控制器
        logger.info(f"Initializing controller {controller_name}")  # 日志记录当前控制器名称
        self.controller_config = controller_registry.get_controller_config(controller_name)  # 获取控制器配置
        if controller_name == "no_control":
            # 如果控制器为无控制模式，则设置动作数量为电机数量
            self.controller_config.num_actions = self.cfg.control_allocator_config.num_motors

        self.num_actions = self.controller_config.num_actions  # 保存动作数量

    @abstractmethod
    def init_tensors(self, global_tensor_dict):
        """
        初始化张量，设置机器人状态和环境边界。

        参数:
        global_tensor_dict: 包含全局张量的字典，主要用于状态和环境边界的初始化。
        """
        self.dt = global_tensor_dict["dt"]  # 时间步长
        self.gravity = global_tensor_dict["gravity"]  # 重力加速度
        self.robot_state = global_tensor_dict["robot_state_tensor"]  # 机器人状态张量
        self.robot_position = global_tensor_dict["robot_position"]  # 机器人位置
        self.robot_orientation = global_tensor_dict["robot_orientation"]  # 机器人朝向
        self.robot_linvel = global_tensor_dict["robot_linvel"]  # 机器人线速度
        self.robot_angvel = global_tensor_dict["robot_angvel"]  # 机器人角速度

        # 用于机器人施加力和扭矩的张量
        self.robot_force_tensors = global_tensor_dict["robot_force_tensor"]
        self.robot_torque_tensors = global_tensor_dict["robot_torque_tensor"]

        self.env_bounds_min = global_tensor_dict["env_bounds_min"]  # 环境边界最小值
        self.env_bounds_max = global_tensor_dict["env_bounds_max"]  # 环境边界最大值

    @abstractmethod
    def reset(self):
        """
        重置机器人状态，具体实现由子类定义。
        """
        pass

    @abstractmethod
    def reset_idx(self, env_ids):
        """
        重置指定环境的索引，具体实现由子类定义。

        参数:
        env_ids: 需要重置的环境索引。
        """
        pass

    @abstractmethod
    def step(self):
        """
        执行一步操作，具体实现由子类定义。
        """
        pass

    # @abstractmethod
    # def apply_noise(self):
    #     """
    #     应用噪声，具体实现由子类定义。
    #     """
    #     pass

    # @abstractmethod
    # def get_state(self):
    #     """
    #     获取当前状态，具体实现由子类定义。
    #     """
    #     pass

    # @abstractmethod
    # def set_state(self, state):
    #     """
    #     设置当前状态，具体实现由子类定义。
    #     """
    #     pass
