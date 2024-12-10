# 导入配置文件
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg  # 导入基础四旋翼配置
from aerial_gym.config.robot_config.base_octarotor_config import BaseOctarotorCfg  # 导入基础八旋翼配置
from aerial_gym.config.robot_config.base_random_config import BaseRandCfg  # 导入基础随机配置
from aerial_gym.config.robot_config.base_rov_config import BaseROVCfg  # 导入基础水下机器人配置
from aerial_gym.config.robot_config.base_quad_root_link_control_config import (
    BaseQuadRootLinkControlCfg,  # 导入基础四旋翼根链接控制配置
)

from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg  # 导入LMF2配置
from aerial_gym.config.robot_config.morphy_config import MorphyCfg  # 导入Morphy配置
from aerial_gym.config.robot_config.morphy_stiff_config import MorphyStiffCfg  # 导入刚性Morphy配置
from aerial_gym.config.robot_config.snakey_config import SnakeyCfg  # 导入Snakey配置
from aerial_gym.config.robot_config.snakey5_config import Snakey5Cfg  # 导入Snakey5配置
from aerial_gym.config.robot_config.snakey6_config import Snakey6Cfg  # 导入Snakey6配置

# 导入机器人类
from aerial_gym.robots.base_multirotor import BaseMultirotor  # 导入基础多旋翼类
from aerial_gym.robots.base_rov import BaseROV  # 导入基础水下机器人类
from aerial_gym.robots.base_reconfigurable import BaseReconfigurable  # 导入基础可重构机器人类
from aerial_gym.robots.morphy import Morphy  # 导入Morphy机器人类

# 获取机器人注册表
from aerial_gym.registry.robot_registry import robot_registry  # 导入机器人注册表

# 在此处注册机器人类
robot_registry.register("base_quadrotor", BaseMultirotor, BaseQuadCfg)  # 注册基础四旋翼机器人
robot_registry.register("base_octarotor", BaseMultirotor, BaseOctarotorCfg)  # 注册基础八旋翼机器人
robot_registry.register("base_random", BaseMultirotor, BaseRandCfg)  # 注册基础随机机器人
robot_registry.register("base_quad_root_link_control", BaseMultirotor, BaseQuadRootLinkControlCfg)  # 注册基础四旋翼根链接控制机器人
robot_registry.register("morphy_stiff", BaseMultirotor, MorphyStiffCfg)  # 注册刚性Morphy机器人
robot_registry.register("morphy", Morphy, MorphyCfg)  # 注册Morphy机器人

robot_registry.register("snakey", BaseReconfigurable, SnakeyCfg)  # 注册Snakey机器人
robot_registry.register("snakey5", BaseReconfigurable, Snakey5Cfg)  # 注册Snakey5机器人
robot_registry.register("snakey6", BaseReconfigurable, Snakey6Cfg)  # 注册Snakey6机器人
robot_registry.register("base_rov", BaseROV, BaseROVCfg)  # 注册基础水下机器人
robot_registry.register("lmf2", BaseMultirotor, LMF2Cfg)  # 注册LMF2机器人
