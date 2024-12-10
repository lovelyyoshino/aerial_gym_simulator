class EnvConfigRegistry:
    """
    此类用于跟踪已注册的环境类。
    新的环境配置可以添加到注册表中，并可以被其他类访问。
    """

    def __init__(self) -> None:
        # 初始化一个空字典，用于存储环境配置
        self.env_configs = {}

    def register(self, env_name, env_config):
        """
        将环境配置添加到环境字典中。
        
        参数:
        env_name: str - 环境的名称
        env_config: any - 环境的配置，可以是任何类型
        """
        # 将环境名称和对应的环境配置添加到字典中
        self.env_configs[env_name] = env_config

    def get_env_config(self, env_name):
        """
        从环境字典中获取指定环境的配置。
        
        参数:
        env_name: str - 需要获取配置的环境名称
        
        返回:
        env_config: any - 对应环境的配置
        """
        # 返回指定环境名称的配置
        return self.env_configs[env_name]

    def get_env_names(self):
        """
        从环境字典中获取所有环境的名称。
        
        返回:
        keys: dict_keys - 所有注册环境的名称列表
        """
        # 返回所有环境名称的键
        return self.env_configs.keys()

    def make_env(self, env_name):
        """
        从环境字典中创建指定环境的实例。
        
        参数:
        env_name: str - 需要创建的环境名称
        
        返回:
        env_config: any - 返回对应环境的配置
        
        抛出:
        ValueError - 如果指定的环境名称没有在注册表中找到
        """
        # 检查环境名称是否在字典中
        if env_name not in self.env_configs:
            # 如果没有找到，抛出异常
            raise ValueError(f"env {env_name} not found in env registry")
        # 返回对应环境的配置
        return self.env_configs[env_name]


# 创建一个全局环境注册表
env_config_registry = EnvConfigRegistry()
