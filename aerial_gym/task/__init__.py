# 从aerial_gym.task.position_setpoint_task模块导入PositionSetpointTask类
from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
    PositionSetpointTask,
)

# 从aerial_gym.task.position_setpoint_task_sim2real模块导入PositionSetpointTaskSim2Real类
from aerial_gym.task.position_setpoint_task_sim2real.position_setpoint_task_sim2real import (
    PositionSetpointTaskSim2Real,
)

# 从aerial_gym.task.position_setpoint_task_acceleration_sim2real模块导入PositionSetpointTaskAccelerationSim2Real类
from aerial_gym.task.position_setpoint_task_acceleration_sim2real.position_setpoint_task_acceleration_sim2real import (
    PositionSetpointTaskAccelerationSim2Real,
)

# 从aerial_gym.task.navigation_task模块导入NavigationTask类
from aerial_gym.task.navigation_task.navigation_task import NavigationTask

# 从配置文件中导入位置设定任务的相关配置
from aerial_gym.config.task_config.position_setpoint_task_config import (
    task_config as position_setpoint_task_config,
)

# 从配置文件中导入位置设定任务（仿真到真实）的相关配置
from aerial_gym.config.task_config.position_setpoint_task_sim2real_config import (
    task_config as position_setpoint_task_sim2real_config,
)

# 从配置文件中导入加速度位置设定任务（仿真到真实）的相关配置
from aerial_gym.config.task_config.position_setpoint_task_acceleration_sim2real_config import (
    task_config as position_setpoint_task_acceleration_sim2real_config,
)

# 从配置文件中导入导航任务的相关配置
from aerial_gym.config.task_config.navigation_task_config import (
    task_config as navigation_task_config,
)

# 导入任务注册表，用于注册不同类型的任务
from aerial_gym.registry.task_registry import task_registry


# 注册位置设定任务及其配置
task_registry.register_task(
    "position_setpoint_task", PositionSetpointTask, position_setpoint_task_config
)
# 注册位置设定任务（仿真到真实）及其配置
task_registry.register_task(
    "position_setpoint_task_sim2real",
    PositionSetpointTaskSim2Real,
    position_setpoint_task_sim2real_config,
)

# 注册加速度位置设定任务（仿真到真实）及其配置
task_registry.register_task(
    "position_setpoint_task_acceleration_sim2real",
    PositionSetpointTaskAccelerationSim2Real,
    position_setpoint_task_acceleration_sim2real_config,
)

# 注册导航任务及其配置
task_registry.register_task("navigation_task", NavigationTask, navigation_task_config)


# 从aerial_gym.task.position_setpoint_task_reconfigurable模块导入可重构的位置设定任务类
from aerial_gym.task.position_setpoint_task_reconfigurable.position_setpoint_task_reconfigurable import (
    PositionSetpointTaskReconfigurable,
)

# 从配置文件中导入可重构位置设定任务的相关配置
from aerial_gym.config.task_config.position_setpoint_task_config_reconfigurable import (
    task_config as position_setpoint_task_config_reconfigurable,
)

# 从aerial_gym.task.position_setpoint_task_morphy模块导入Morphy位置设定任务类
from aerial_gym.task.position_setpoint_task_morphy.position_setpoint_task_morphy import (
    PositionSetpointTaskMorphy,
)

# 从配置文件中导入Morphy位置设定任务的相关配置
from aerial_gym.config.task_config.position_setpoint_task_morphy_config import (
    task_config as position_setpoint_task_config_morphy,
)


# 注册可重构的位置设定任务及其配置
task_registry.register_task(
    "position_setpoint_task_reconfigurable",
    PositionSetpointTaskReconfigurable,
    position_setpoint_task_config_reconfigurable,
)

# 注册Morphy位置设定任务及其配置
task_registry.register_task(
    "position_setpoint_task_morphy",
    PositionSetpointTaskMorphy,
    position_setpoint_task_config_morphy,
)


## 如果需要使用自定义任务，请取消注释以下内容

# 从aerial_gym.task.custom_task模块导入CustomTask类
# task_registry.register_task("custom_task", CustomTask, custom_task.task_config)
