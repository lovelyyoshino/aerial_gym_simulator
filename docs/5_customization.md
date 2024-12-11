## 自定义物理参数

在 `sim` 类中，可以使用另一个类名设置不同的物理引擎和查看窗口组合参数。此配置可以在相关文件夹的 `__init__.py` 文件中注册，以便在仿真中使用，或者在运行时通过以下命令在代码中使用：

??? 示例 "自定义仿真参数"
```python
from aerial_gym.registry.sim_registry import sim_registry
from aerial_gym.simulation.sim_params import BaseSimParams

class CustomSimParamsFallingForwards(BaseSimParams):
    class sim(BaseSimConfig.sim):
        dt = 0.01  # 自定义参数
        gravity = [+1.0, 0.0, 0.0]
        

# 在此处注册您的自定义类
sim_registry.register_sim_params("falling_forwards", CustomSimParamsFallingForwards)

### 在代码中进一步使用注册的类以生成仿真 ###
```

然后，可以创建一个配置文件，指定仿真应使用自定义参数。仿真器（代码中的 Isaac Gym 实例）必须重新启动以使更改生效。

!!! 注意 "物理仿真参数可以通过 sim_registry 注册"
- 在 `sim/__init__.py` 文件中，以便在整个仿真运行中被命名和识别，或者
- 在运行时通过以下命令在代码中注册：
    ```python
    from aerial_gym.registry.sim_registry import sim_registry
    sim_registry.register_sim_params("falling_forwards_sim_params", CustomSimParamsFallingForwards)
    ```
    !!! 警告 "仿真实例需要重新启动以使参数生效"

## 自定义环境

我们提供了一个示例环境，其中包含多个链接的参数树对象，以模拟森林。

可以通过设置其资产属性并为该环境创建一个环境文件，指定要包含在环境中的资产，从而向环境中添加更多对象。

通过这种方式，可以：

1. 在不同环境中重用相同的资产集
2. 轻松组合多个具有相同或不同参数的资产的环境，以实现随机化
3. 包含您自己的资产并为特定任务创建自定义环境

??? 示例 "森林环境的配置文件"
```python
from aerial_gym.config.env_config.env_object_config import EnvObjectConfig
import numpy as np

class ForestEnvCfg(EnvObjectConfig):
    class env:
        num_envs = 64
        num_env_actions = 4
        env_spacing = 5.0  # 在高度场/三角网格中未使用
        num_physics_steps_per_env_step_mean = 10  # 相机渲染之间的步骤数均值
        num_physics_steps_per_env_step_std = 0  # 相机渲染之间的步骤数标准差
        render_viewer_every_n_steps = 1  # 每 n 步渲染查看器
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        collision_force_threshold = 0.005  # 碰撞力阈值 [N]
        create_ground_plane = False  # 创建地面平面
        sample_timestep_for_latency = True  # 采样时间步以获取延迟噪声
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        use_warp = True
        lower_bound_min = [-5.0, -5.0, -1.0]  # 环境空间的下界
        lower_bound_max = [-5.0, -5.0, -1.0]  # 环境空间的下界
        upper_bound_min = [5.0, 5.0, 3.0]  # 环境空间的上界
        upper_bound_max = [5.0, 5.0, 3.0]  # 环境空间的上界

    class env_config:
        include_asset_type = {
            "trees": True,
            "objects": True,
            "bottom_wall": True,
        }

        # 将上述名称映射到定义资产的类。它们可以在 include_asset_type 中启用和禁用
        asset_type_to_dict_map = {
            "trees": EnvObjectConfig.tree_asset_params,
            "objects": EnvObjectConfig.object_asset_params,
            "bottom_wall": EnvObjectConfig.bottom_wall,
        }
```
环境如下所示：

![森林环境](./images/forest_environment.png)

## 自定义控制器

可以根据用户对其首选机器人配置的需求添加额外的控制器。我们提供了一个非标准控制器的示例，该控制器在 `controller` 文件夹中跟踪速度和转向角命令。车辆速度以车辆框架表示，转向角相对于世界框架测量。控制器可以在 `controller` 文件夹的 `__init__.py` 文件中注册，或在运行时在代码中注册。为了更好地展示与我们现有代码的集成，我们利用了 `base_lee_controller.py` 类提供的功能，但用户不必遵循此结构，可以根据自己的需求编写自己的控制器结构。我们还修改了 `base_lee_controller.py` 文件中的控制器，以展示如何控制一个具有 8 个电机的完全驱动平台，并使用它来控制水下车辆的模型。我们还提供了一个示例文件，以模拟该控制器与水下机器人模型的配合。

??? 示例 "`FullyActuatedController` 代码"
```python
class FullyActuatedController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        
    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)

    def update(self, command_actions):
        """
        完全驱动控制器。输入为期望的位置和方向。
        command_actions = [p_x, p_y, p_z, qx, qy, qz, qw]
        位置设定点在世界框架中
        方向参考相对于世界框架
        """
        self.reset_commands()
        command_actions[:, 3:7] = normalize(command_actions[:, 3:7])
        self.accel[:] = self.compute_acceleration(command_actions[:, 0:3], torch.zeros_like(command_actions[:, 0:3]))
        forces = self.mass * (self.accel - self.gravity)
        self.wrench_command[:, 0:3] = quat_rotate_inverse(
            self.robot_orientation, forces
        )
        self.desired_quat[:] = command_actions[:, 3:]
        self.wrench_command[:, 3:6] = self.compute_body_torque(
            self.desired_quat, torch.zeros_like(command_actions[:, 0:3])
        )
        return self.wrench_command
```

!!! 注意 "控制器可以通过 controller_registry 注册"
- 在 `controller/__init__.py` 文件中，以便在整个仿真运行中被命名和识别，或者
- 在运行时通过以下命令在代码中注册：
    ```python
    from aerial_gym.registry.controller_registry import controller_registry
    controller_registry.register_controller(
        "fully_actuated_control", FullyActuatedController, fully_actuated_controller_config
    )
    ```

## 自定义机器人

我们支持在仿真器中添加自定义机器人配置和自定义控制方法。仿真器提供了一个具有 8 个电机的任意机器人配置示例。如果机器人配置有显著差异，可以为其创建自己的 Python 类，以控制机器人链接、控制器，并利用机器人上的传感器。机器人配置可以在 `robot` 文件夹的 `__init__.py` 文件中注册，或在运行时在代码中注册。此外，当前的文件结构允许我们重用相同的机器人类，并使用适当的配置文件。

在我们的案例中，我们使用 `base_quadrotor.py` 类以及适当的机器人配置文件。例如，对于任意机器人模型，我们使用以下配置文件：

??? 示例 "任意机器人模型的配置文件"
```python
# 上述仿真器的资产参数

class control_allocator_config:
    num_motors = 8
    force_application_level = "motor_link"
    # "motor_link" 或 "root_link" 以在根链接或单个电机链接上施加力
    
    motor_mask = [1 + 8 + i for i in range(0, 8)]
    motor_directions = [1, -1, 1, -1, 1, -1, 1, -1]
    
    allocation_matrix = [[ 5.55111512e-17, -3.21393805e-01, -4.54519478e-01, -3.42020143e-01,
                        9.69846310e-01,  3.42020143e-01,  8.66025404e-01, -7.54406507e-01],
                        [ 1.00000000e+00, -3.42020143e-01, -7.07106781e-01,  0.00000000e+00,
                        -1.73648178e-01,  9.39692621e-01,  5.00000000e-01, -1.73648178e-01],
                        [ 1.66533454e-16, -8.83022222e-01,  5.41675220e-01,  9.39692621e-01,
                        1.71010072e-01,  1.11022302e-16,  1.11022302e-16,  6.33022222e-01],
                        [ 1.75000000e-01,  1.23788742e-01, -5.69783368e-02,  1.34977168e-01,
                        3.36959042e-02, -2.66534135e-01, -7.88397460e-02, -2.06893989e-02],
                        [ 1.00000000e-02,  2.78845133e-01, -4.32852308e-02, -2.72061766e-01,
                        -1.97793856e-01,  8.63687139e-02,  1.56554446e-01, -1.71261290e-01],
                        [ 2.82487373e-01, -1.41735490e-01, -8.58541103e-02,  3.84858939e-02,
                        -3.33468026e-01,  8.36741468e-02,  8.46777988e-03, -8.74336259e-02]]
    
    # 在此处，分配矩阵是由用户根据机器人的 URDF 文件计算得出的
    # 以映射电机力对作用在机器人上的净力和扭矩的影响。

class motor_model_config:
    motor_time_constant_min = 0.01
    motor_time_constant_max = 0.03
    max_thrust = 5.0
    min_thrust = -5.0
    max_thrust_rate = 100.0
    thrust_to_torque_ratio = 0.01 # 推力与扭矩比与惯性矩阵相关，请勿更改

# 机器人其他参数如下
```

此外，我们还提供了一个控制水下 [BlueROV](https://bluerobotics.com/store/rov/bluerov2/) 机器人模型的示例，具有 8 个电机和一个用于完全驱动平台的自定义控制器。我们提供了一个示例文件，展示了使用控制器对机器人进行全状态跟踪。

![BlueROV 位置跟踪](./gifs/BlueROVExample.gif)

## 自定义任务

您可以参考 `tasks/custom_task` 中的示例文件，并按照此处所示实现自己的任务规范：

??? 示例 "自定义任务类定义"
```python
class CustomTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.device = self.task_config.device
        # 在此处编写您自己的实现

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            args=self.task_config.args,
            device=self.device,
        )

        # 在此处实现与您的任务相关的内容

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.sim_env.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        # 在此处编写您的实现
        return None

    def reset_idx(self, env_ids):
        # 在此处编写您的实现
        return

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # 使用动作，获取观察
        # 计算奖励，返回元组
        # 在这种情况下，终止的回合需要
        # 首先重置，并返回新回合的第一个观察。

        # 用与您的任务相关的内容替换此内容
        self.sim_env.step(actions=actions)
        
        return None # 用与您的任务相关的内容替换此内容

@torch.jit.script
def compute_reward(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # 在此处实现内容
    return 0
```

!!! 注意 "任务必须通过任务注册表注册"
- 在 `task/__init__.py` 文件中，以便在整个仿真运行中被命名和识别，或者
- 在运行时通过以下命令在代码中注册：
    ```python
    from aerial_gym.registry.task_registry import task_registry
    task_registry.register_task("custom_task", CustomTask)
    ```

## 自定义传感器

暴露传感器参数以进行光线投射传感器，使用户能够单独自定义，以模拟外部感知传感器。我们提供了一个基于 Ouster OSDome LiDAR 的半球形 LiDAR 传感器示例。

??? 示例 "参数可以如下配置"
```python
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig

class OSDome_64_Config(BaseLidarConfig):
    # 保持一切基本相同并更改垂直光线的数量
    height = 64
    width = 512
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180
    vertical_fov_deg_min = 0
    vertical_fov_deg_max = 90
    max_range = 20.0
    min_range = 0.5

    return_pointcloud= False
    segmentation_camera = True

    # 随机化传感器的放置
    randomize_placement = False
    min_translation = [0.0, 0.0, 0.0]
    max_translation = [0.0, 0.0, 0.0]
    # 前置圆顶 LiDAR 的示例
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]
    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
```

来自半球形 LiDAR 的数据被投影到范围和分割图像中，如下所示：

![OSDome 深度数据](./gifs/dome_lidar_depth_frames_200.gif)
![OSDome 分割数据](./gifs/dome_lidar_seg_frames_200.gif)