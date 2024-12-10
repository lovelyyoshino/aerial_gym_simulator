# 从模拟器配置注册表中导入 sim_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry

# 导入基本模拟配置类
from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig

# 导入无头模式的基本模拟配置类
from aerial_gym.config.sim_config.base_sim_headless_config import (
    BaseSimHeadlessConfig,
)

# 导入 2ms 模式的模拟配置类
from aerial_gym.config.sim_config.sim_config_2ms import SimCfg2Ms

# 将基本模拟配置类注册到模拟器配置注册表中
sim_config_registry.register("base_sim", BaseSimConfig)
# 将无头模式的基本模拟配置类注册到模拟器配置注册表中
sim_config_registry.register("base_sim_headless", BaseSimHeadlessConfig)
# 将 2ms 模式的模拟配置类注册到模拟器配置注册表中
sim_config_registry.register("base_sim_2ms", SimCfg2Ms)

# 如果需要注册自定义的模拟配置，可以取消下面两行的注释
# 从自定义模拟配置中导入 CustomSimConfig 类
# from aerial_gym.config.sim_config.custom_sim_config import CustomSimConfig
# 将自定义模拟配置类注册到模拟器配置注册表中
# sim_config_registry.register("custom_sim", CustomSimConfig)
