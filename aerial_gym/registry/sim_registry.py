class SimConfigRegistry:
    """
    此类用于跟踪已注册的仿真配置类。
    可以将新的仿真配置添加到注册表中，并可以被其他类访问。
    """

    def __init__(self) -> None:
        # 初始化一个空字典，用于存储仿真配置
        self.sim_configs = {}

    def register(self, sim_name, sim_config):
        """
        将仿真配置添加到仿真字典中。
        
        参数:
        sim_name: str - 仿真的名称
        sim_config: object - 仿真配置对象
        """
        # 将仿真名称与其配置关联并存入字典
        self.sim_configs[sim_name] = sim_config

    def get_sim_config(self, sim_name):
        """
        从仿真字典中获取指定的仿真配置。
        
        参数:
        sim_name: str - 仿真的名称
        
        返回:
        object - 对应的仿真配置对象
        """
        # 根据仿真名称返回其配置
        return self.sim_configs[sim_name]

    def get_sim_names(self):
        """
        从仿真字典中获取所有仿真名称。
        
        返回:
        dict_keys - 仿真名称的集合
        """
        # 返回所有注册的仿真名称
        return self.sim_configs.keys()

    def make_sim(self, sim_name):
        """
        根据仿真字典中的名称创建仿真。
        
        参数:
        sim_name: str - 仿真的名称
        
        返回:
        object - 对应的仿真配置对象
        
        异常:
        ValueError - 如果指定的仿真名称在注册表中未找到
        """
        # 检查仿真名称是否在注册表中
        if sim_name not in self.sim_configs:
            # 如果未找到，则抛出异常
            raise ValueError(f"sim {sim_name} not found in sim registry")
        # 返回对应的仿真配置对象
        return self.sim_configs[sim_name]


# 创建一个全局的仿真配置注册表
sim_config_registry = SimConfigRegistry()
