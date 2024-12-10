class ControllerRegistry:
    """
    这个类用于注册和获取环境中的控制器。
    """

    def __init__(self) -> None:
        # 初始化控制器类和配置的字典
        self.controller_classes = {}  # 存储控制器类的字典
        self.controller_configs = {}   # 存储控制器配置的字典

    def register_controller(self, controller_name, controller_class, controller_config):
        """
        将控制器添加到控制器字典中。
        
        参数:
        controller_name: 控制器的名称
        controller_class: 控制器的类
        controller_config: 控制器的配置
        """
        # 将控制器类和配置存储到字典中
        self.controller_classes[controller_name] = controller_class
        self.controller_configs[controller_name] = controller_config

    def get_controller_class(self, controller_name):
        """
        从控制器字典中获取控制器类。
        
        参数:
        controller_name: 控制器的名称
        
        返回:
        对应的控制器类
        """
        return self.controller_classes[controller_name]

    def get_controller_names(self):
        """
        从控制器字典中获取所有控制器的名称。
        
        返回:
        控制器名称的可迭代对象
        """
        return self.controller_classes.keys()

    def get_controller_config(self, controller_name):
        """
        从控制器字典中获取控制器的配置。
        
        参数:
        controller_name: 控制器的名称
        
        返回:
        对应的控制器配置
        """
        return self.controller_configs[controller_name]

    def make_controller(self, controller_name, num_envs, device, mode="robot"):
        """
        从控制器字典中创建一个控制器。
        
        参数:
        controller_name: 控制器的名称
        num_envs: 环境的数量
        device: 设备信息
        mode: 模式（默认为"robot"）
        
        返回:
        创建的控制器实例和对应的配置
        """
        # 检查控制器名称是否在注册表中
        if controller_name not in self.controller_classes:
            raise ValueError(
                f"控制器 {controller_name} 在控制器注册表中未找到。可用的控制器有 {self.controller_classes.keys()}"
            )
        # 创建并返回控制器实例及其配置
        return (
            self.controller_classes[controller_name](
                self.controller_configs[controller_name],
                num_envs,
                device,
            ),
            self.controller_configs[controller_name],
        )


controller_registry = ControllerRegistry()  # 创建控制器注册表实例
