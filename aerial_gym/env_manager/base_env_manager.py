from abc import ABC

# 基础管理器类，使用抽象基类（ABC）来定义接口
class BaseManager(ABC):
    # 初始化函数，接受配置和设备参数
    def __init__(self, config, device):
        self.cfg = config  # 存储配置参数
        self.device = device  # 存储设备信息

    # 重置函数，子类需要实现该方法
    def reset(self):
        raise NotImplementedError("reset not implemented")  # 抛出未实现异常

    # 根据环境ID重置函数，子类需要实现该方法
    def reset_idx(self, env_ids):
        raise NotImplementedError("reset_idx not implemented")  # 抛出未实现异常

    # 在物理步骤之前执行的函数，可以用于处理动作
    def pre_physics_step(self, actions):
        pass  # 该方法可以被子类重写，但默认不执行任何操作

    # 物理步骤，子类需要实现该方法
    def step(self):
        raise NotImplementedError("step not implemented")  # 抛出未实现异常

    # 在物理步骤之后执行的函数，可以用于后续处理
    def post_physics_step(self):
        pass  # 该方法可以被子类重写，但默认不执行任何操作

    # 初始化张量，接受一个全局张量字典
    def init_tensors(self, global_tensor_dict):
        pass  # 该方法可以被子类重写，但默认不执行任何操作
