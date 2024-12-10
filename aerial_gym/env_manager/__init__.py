import isaacgym  # 导入Isaac Gym库，用于仿真和训练
from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg  # 导入带障碍物环境的配置
from aerial_gym.config.env_config.empty_env import EmptyEnvCfg  # 导入空环境的配置
from aerial_gym.config.env_config.forest_env import ForestEnvCfg  # 导入森林环境的配置
from aerial_gym.config.env_config.env_config_2ms import EnvCfg2Ms  # 导入2ms环境的配置
from aerial_gym.config.env_config.dynamic_environment import DynamicEnvironmentCfg  # 导入动态环境的配置

from aerial_gym.registry.env_registry import env_config_registry  # 导入环境配置注册表

# 注册不同的环境配置到环境配置注册表中
env_config_registry.register("env_with_obstacles", EnvWithObstaclesCfg)  # 注册带障碍物环境配置
env_config_registry.register("empty_env", EmptyEnvCfg)  # 注册空环境配置
env_config_registry.register("forest_env", ForestEnvCfg)  # 注册森林环境配置
env_config_registry.register("empty_env_2ms", EnvCfg2Ms)  # 注册2ms空环境配置
env_config_registry.register("dynamic_env", DynamicEnvironmentCfg)  # 注册动态环境配置
