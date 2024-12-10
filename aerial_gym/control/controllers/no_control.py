from aerial_gym.utils.logging import CustomLogger  # 导入自定义日志记录器
from aerial_gym.control.control_allocation import ControlAllocator  # 导入控制分配器

logger = CustomLogger("no_control")  # 创建一个名为"no_control"的日志记录实例


class NoControl:
    def __init__(self, config, num_envs, device):
        # 初始化NoControl类的实例
        # config: 配置参数
        # num_envs: 环境数量
        # device: 设备信息
        pass

    def init_tensors(self, global_tensor_dict=None):
        # 初始化张量
        # global_tensor_dict: 可选的全局张量字典
        pass

    def __call__(self, *args, **kwargs):
        # 将实例作为函数调用时，转发到update方法
        return self.update(*args, **kwargs)

    def reset_commands(self):
        # 重置控制命令
        pass

    def reset(self):
        # 重置NoControl实例
        # 调用reset_idx方法重置指定的环境索引
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        # 重置指定的环境索引
        # env_ids: 要重置的环境ID列表
        pass

    def randomize_params(self, env_ids):
        # 随机化参数
        # env_ids: 要随机化的环境ID列表
        pass

    def update(self, command_actions):
        # 更新控制命令
        # command_actions: 输入的控制命令
        # 直接返回输入的控制命令，这表示该类不会对命令进行任何修改
        return command_actions
