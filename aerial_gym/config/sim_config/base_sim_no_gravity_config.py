# 从基础模拟配置模块导入BaseSimConfig类
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig

# 定义一个名为BaseSimNoGravityConfig的类，继承自BaseSimConfig
class BaseSimNoGravityConfig(BaseSimConfig):
    # 在BaseSimNoGravityConfig中定义一个内部类sim，继承自BaseSimConfig.sim
    class sim(BaseSimConfig.sim):
        # 设置重力向量为零，即在此模拟中不考虑重力影响
        gravity = [0.0, 0.0, 0.0]
