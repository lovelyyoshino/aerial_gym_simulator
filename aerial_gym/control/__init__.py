# 导入控制器类
from aerial_gym.control.controllers.acceleration_control import (
    LeeAccelerationController,
)
from aerial_gym.control.controllers.attitude_control import LeeAttitudeController
from aerial_gym.control.controllers.velocity_control import LeeVelocityController
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.controllers.velocity_steeing_angle_controller import (
    LeeVelocitySteeringAngleController,
)
from aerial_gym.control.controllers.rates_control import LeeRatesController
from aerial_gym.control.controllers.no_control import NoControl

# 导入控制器的配置
from aerial_gym.config.controller_config.lee_controller_config import (
    control as lee_controller_config,
)
from aerial_gym.config.controller_config.no_control_config import (
    control as no_control_config,
)

from aerial_gym.config.controller_config.lee_controller_config_octarotor import (
    control as lee_controller_config_octarotor,
)

from aerial_gym.control.controllers.fully_actuated_control import FullyActuatedController
from aerial_gym.config.controller_config.fully_actuated_controller_rov import (
    control as fully_actuated_controller_config,
)

from aerial_gym.config.controller_config.lmf2_controller_config import (
    control as lmf2_controller_config,
)

# 注册控制器到控制器注册表
from aerial_gym.registry.controller_registry import controller_registry

# 注册无控制器
controller_registry.register_controller("no_control", NoControl, no_control_config)

# 注册Lee控制器及其配置
controller_registry.register_controller(
    "lee_acceleration_control", LeeAccelerationController, lee_controller_config
)
controller_registry.register_controller(
    "lee_attitude_control", LeeAttitudeController, lee_controller_config
)
controller_registry.register_controller(
    "lee_velocity_control", LeeVelocityController, lee_controller_config
)
controller_registry.register_controller(
    "lee_position_control", LeePositionController, lee_controller_config
)
controller_registry.register_controller(
    "lee_rates_control", LeeRatesController, lee_controller_config
)

# 注册八旋翼的Lee控制器及其配置
controller_registry.register_controller(
    "lee_acceleration_control_octarotor", LeeAccelerationController, lee_controller_config_octarotor
)
controller_registry.register_controller(
    "lee_attitude_control_octarotor", LeeAttitudeController, lee_controller_config_octarotor
)
controller_registry.register_controller(
    "lee_velocity_control_octarotor", LeeVelocityController, lee_controller_config_octarotor
)
controller_registry.register_controller(
    "lee_position_control_octarotor", LeePositionController, lee_controller_config_octarotor
)
controller_registry.register_controller(
    "lee_rates_control_octarotor", LeeRatesController, lee_controller_config_octarotor
)

# 注册速度转向角控制器
controller_registry.register_controller(
    "lee_velocity_steering_angle_control",
    LeeVelocitySteeringAngleController,
    lee_controller_config,
)

# 注册完全驱动控制器
controller_registry.register_controller(
    "fully_actuated_control", FullyActuatedController, fully_actuated_controller_config
)

# 注册LMF2控制器及其配置
controller_registry.register_controller(
    "lmf2_position_control", LeePositionController, lmf2_controller_config
)

controller_registry.register_controller(
    "lmf2_velocity_control", LeeVelocityController, lmf2_controller_config
)

controller_registry.register_controller(
    "lmf2_attitude_control", LeeAttitudeController, lmf2_controller_config
)

controller_registry.register_controller(
    "lmf2_rates_control", LeeRatesController, lmf2_controller_config
)

controller_registry.register_controller(
    "lmf2_acceleration_control", LeeAccelerationController, lmf2_controller_config
)
