class TaskRegistry:
    def __init__(self):
        # 初始化任务注册表，包含任务类和任务配置的字典
        self.task_class_registry = {}  # 存储任务名称与任务类的映射
        self.task_config_registry = {}  # 存储任务名称与任务配置的映射

    def register_task(self, task_name, task_class, task_config):
        # 注册一个新任务，将任务类和任务配置存储到相应的字典中
        # task_name: 任务名称
        # task_class: 任务类
        # task_config: 任务配置
        self.task_class_registry[task_name] = task_class  # 将任务名称与任务类关联
        self.task_config_registry[task_name] = task_config  # 将任务名称与任务配置关联

    def get_task_class(self, task_name):
        # 根据任务名称获取对应的任务类
        # task_name: 任务名称
        # 返回: 对应的任务类
        return self.task_class_registry[task_name]

    def get_task_config(self, task_name):
        # 根据任务名称获取对应的任务配置
        # task_name: 任务名称
        # 返回: 对应的任务配置
        return self.task_config_registry[task_name]

    def get_task_names(self):
        # 获取所有已注册的任务名称
        # 返回: 任务名称列表
        return list(self.task_class_registry.keys())

    def get_task_classes(self):
        # 获取所有已注册的任务类
        # 返回: 任务类列表
        return list(self.task_class_registry.values())

    def get_task_configs(self):
        # 获取所有已注册的任务配置
        # 返回: 任务配置列表
        return list(self.task_config_registry.values())

    def make_task(self, task_name, seed=None, num_envs=None, headless=None, use_warp=None):
        # 创建一个任务实例
        # task_name: 任务名称
        # seed: 随机种子
        # num_envs: 环境数量
        # headless: 是否无头运行
        # use_warp: 是否使用变速
        task_class = self.get_task_class(task_name)  # 获取任务类
        task_config = self.get_task_config(task_name)  # 获取任务配置
        # 根据任务类和配置实例化任务对象，并传入其他参数
        return task_class(
            task_config, seed=seed, num_envs=num_envs, headless=headless, use_warp=use_warp
        )


task_registry = TaskRegistry()  # 创建任务注册表的实例
