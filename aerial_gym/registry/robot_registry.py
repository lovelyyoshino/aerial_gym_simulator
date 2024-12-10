from aerial_gym.registry.controller_registry import controller_registry


class RobotRegistry:
    """
    这个类用于跟踪创建的机器人。
    可以将新机器人添加到注册表中，并可以被其他类访问。
    这将允许环境管理器创建机器人，并使机器人管理器能够访问这些机器人。
    """

    def __init__(self) -> None:
        # 初始化一个空的字典用于存储机器人类和配置
        self.robot_classes = {}
        self.robot_configs = {}

    def register(self, robot_name, robot_class, robot_config):
        """
        将机器人添加到机器人字典中。
        
        参数:
        robot_name: 机器人的名称
        robot_class: 机器人的类
        robot_config: 机器人的配置
        """
        # 将机器人类和配置以名称为键存储
        self.robot_classes[robot_name] = robot_class
        self.robot_configs[robot_name] = robot_config

    def get_robot_class(self, robot_name):
        """
        从机器人字典中获取机器人的类。
        
        参数:
        robot_name: 机器人的名称
        
        返回:
        机器人的类
        """
        # 返回指定名称的机器人类
        return self.robot_classes[robot_name]

    def get_robot_config(self, robot_name):
        """
        从机器人字典中获取机器人的配置。
        
        参数:
        robot_name: 机器人的名称
        
        返回:
        机器人的配置
        """
        # 返回指定名称的机器人配置
        return self.robot_configs[robot_name]

    def get_robot_names(self):
        """
        从机器人字典中获取机器人的名称列表。
        
        返回:
        机器人的名称集合
        """
        # 返回所有注册的机器人名称
        return self.robot_classes.keys()

    def make_robot(self, robot_name, controller_name, env_config, device):
        """
        从机器人字典中创建一个机器人实例。
        
        参数:
        robot_name: 机器人的名称
        controller_name: 控制器的名称
        env_config: 环境配置
        device: 设备信息
        
        返回:
        一个元组，包含机器人实例和机器人的配置
        """
        # 检查指定的机器人名称是否在注册表中
        if robot_name not in self.robot_classes:
            raise ValueError(f"Robot {robot_name} not found in robot registry")
        
        # 创建并返回一个机器人实例及其配置
        return (
            self.robot_classes[robot_name](
                self.robot_configs[robot_name], controller_name, env_config, device
            ),
            self.robot_configs[robot_name],
        )


# 创建一个全局的机器人注册表
robot_registry = RobotRegistry()
