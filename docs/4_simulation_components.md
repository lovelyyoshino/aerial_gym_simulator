# 模拟组件

模拟器由可组合的实体构成，允许更改物理引擎参数，添加各种环境处理功能，以便在运行时操控环境，加载资源，选择机器人和控制器，并设计自定义传感器。

## 注册表

代码中包含注册表，用于对各种配置文件和类进行命名映射。注册表通过参数简化配置，并允许在模拟中混合搭配各种设置、机器人、环境和传感器。新的配置可以动态创建并注册，以便通过活动代码以编程方式使用，而无需停止手动配置模拟。

代码中包含以下组件的注册表：

[目录]

要注册您自己的自定义组件，请查看[自定义](./5_customization.md)页面。

## 模拟参数

我们使用NVIDIA的Isaac Gym作为模拟引擎。模拟器允许选择不同的物理后端，如PhysX和Flex。模拟器提供了一组API与环境进行交互，例如设置重力、时间步长和渲染选项。我们提供了一组基于PhysX的物理引擎默认配置，可以根据用户的需求进行设置。模拟参数在此设置：

??? 示例 "默认模拟参数"
```python
class BaseSimConfig:
    # 观察者相机：
    class viewer:
        headless = False
        ref_env = 0
        camera_position = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]
        camera_orientation_euler_deg = [0, 0, 0]  # [deg]
        camera_follow_type = "FOLLOW_TRANSFORM"
        width = 1280
        height = 720
        max_range = 100.0  # [m]
        min_range = 0.1
        horizontal_fov_deg = 90
        use_collision_geometry = False
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.2]  # m
        camera_follow_position_global_offset = [-1.0, 0.0, 0.4]  # m

    class sim:
        dt = 0.01
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0是y，1是z
        use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 2
            contact_offset = 0.002  # [m]
            rest_offset = 0.001  # [m]
            bounce_threshold_velocity = 0.1  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> 需要用于8000个环境及以上
            default_buffer_size_multiplier = 10
            contact_collection = 1  # 0: 从不，1: 最后一个子步骤，2: 所有子步骤（默认=2）
```

虽然该模拟器的物理引擎默认设置为PhysX，但可以通过少量修改更改为Flex。尚未对此目的进行测试。

## 资源

Aerial Gym中的模拟资源通常是URDF文件，每个文件可以具有自己的模拟参数。资源的配置文件存储在`config/asset_config`中。在我们的实现中，我们根据资源的类型定义了资源类，并将其属性整合在一起。每种类型的资源可以通过不同的URDF文件表示，从而在环境中实现随机化，而无需进一步指定要加载的资源。只需将额外的URDF文件添加到适当的文件夹路径，即可将其添加到模拟选择池中。

每个资源的参数源自`BaseAssetParams`类，该类包括每种类型在每个环境中加载的资源数量，指定根资源文件夹，指定资源的位置、方向比例以及物理属性，如阻尼系数、密度等。可以使用其他参数来指定资源上力传感器的存在、每个链接或整个身体的分割标签等。

`BaseAssetParams`文件如下：

??? 示例 "可配置的`BaseAssetParams`类示例"
```python
class BaseAssetParams:
    num_assets = 1  # 每个环境中包含的资源数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file = None  # 如果file=None，将随机选择资源。如果不为None，则将使用该文件

    min_position_ratio = [0.5, 0.5, 0.5]  # 最小位置作为边界的比例
    max_position_ratio = [0.5, 0.5, 0.5]  # 最大位置作为边界的比例

    collision_mask = 1

    disable_gravity = False
    replace_cylinder_with_capsule = True  # 用胶囊替换碰撞圆柱，导致更快/更稳定的模拟
    flip_visual_attachments = (
        True  # 一些.obj网格必须从y-up翻转为z-up
    )
    density = 0.000001
    angular_damping = 0.0001
    linear_damping = 0.0001
    max_angular_velocity = 100.0
    max_linear_velocity = 100.0
    armature = 0.001

    collapse_fixed_joints = True
    fix_base_link = True
    color = None
    keep_in_env = False

    body_semantic_label = 0
    link_semantic_label = 0
    per_link_semantic = False
    semantic_masked_links = {}
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # 位置，四元数x，y，z，w
    use_collision_mesh_instead_of_visual = False
```

## 环境

环境规范决定了模拟环境中的组件。可以使用配置文件选择特定的机器人（及其传感器）、为机器人指定的控制器以及环境中存在的障碍物，并选择相对于环境边界生成和随机化其位置的策略。环境配置文件存储在`config/env_config`文件夹中。环境管理器调用每个环境实体，以允许在每个时间步执行特定的用户编码行为或在环境重置时执行某些操作。环境管理器负责在环境中生成资源、机器人和障碍物，并管理它们之间的交互。

??? 示例 "带有机器人的空环境配置示例"
```python
class EmptyEnvCfg:
    class env:
        num_envs = 3  # 环境数量
        num_env_actions = 0  # 这是环境处理的动作数量
        # 这些是发送到环境实体的动作
        # 其中一些可能用于控制环境中的各种实体
        # 例如，障碍物的运动等。
        env_spacing = 1.0  # 不适用于高度场/三角网
        num_physics_steps_per_env_step_mean = 1  # 相机渲染之间的步骤数量均值
        num_physics_steps_per_env_step_std = 0  # 相机渲染之间的步骤数量标准差
        render_viewer_every_n_steps = 10  # 每n步渲染观察者
        collision_force_threshold = 0.010  # 碰撞力阈值
        manual_camera_trigger = False  # 手动触发相机捕获
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        create_ground_plane = False  # 创建地面
        sample_timestep_for_latency = True  # 采样时间步长以获取延迟噪声
        perturb_observations = True
        keep_same_env_for_num_episodes = 1

        use_warp = False
        e_s = env_spacing
        lower_bound_min = [-e_s, -e_s, -e_s]  # 环境空间的下界
        lower_bound_max = [-e_s, -e_s, -e_s]  # 环境空间的下界
        upper_bound_min = [e_s, e_s, e_s]  # 环境空间的上界
        upper_bound_max = [e_s, e_s, e_s]  # 环境空间的上界

    class env_config:
        include_asset_type = {}

        asset_type_to_dict_map = {}
```

为了将资源添加到环境中，可以使用`include_asset_type`字典来指定要包含在环境中的资源。`asset_type_to_dict_map`字典将资源类型映射到定义资源参数的类。

??? 示例 "带有障碍物的环境配置文件示例"
```python
from aerial_gym.config.env_config.env_object_config import EnvObjectConfig

import numpy as np


class EnvWithObstaclesCfg(EnvObjectConfig):
    class env:
        num_envs = 64
        num_env_actions = 4  # 这是环境处理的动作数量
        # 其中一些可以是来自RL代理的输入，用于机器人
        # 其中一些可以用于控制环境中的各种实体
        # 例如，障碍物的运动等。
        env_spacing = 5.0  # 不适用于高度场/三角网

        num_physics_steps_per_env_step_mean = 10  # 相机渲染之间的步骤数量均值
        num_physics_steps_per_env_step_std = 0  # 相机渲染之间的步骤数量标准差

        render_viewer_every_n_steps = 1  # 每n步渲染观察者
        reset_on_collision = (
            True  # 当四旋翼上的接触力超过阈值时重置环境
        )
        collision_force_threshold = 0.05  # 碰撞力阈值 [N]
        create_ground_plane = False  # 创建地面
        sample_timestep_for_latency = True  # 采样时间步长以获取延迟噪声
        perturb_observations = True
        keep_same_env_for_num_episodes = 1

        use_warp = True
        lower_bound_min = [-2.0, -4.0, -3.0]  # 环境空间的下界
        lower_bound_max = [-1.0, -2.5, -2.0]  # 环境空间的下界
        upper_bound_min = [9.0, 2.5, 2.0]  # 环境空间的上界
        upper_bound_max = [10.0, 4.0, 3.0]  # 环境空间的上界

    class env_config:
        include_asset_type = {
            "panels": True,
            "thin": False,
            "trees": False,
            "objects": True,
            "left_wall": True,
            "right_wall": True,
            "back_wall": True,
            "front_wall": True,
            "top_wall": True,
            "bottom_wall": True,
        }

        # 将上述名称映射到定义资源的类。它们可以在include_asset_type中启用和禁用
        asset_type_to_dict_map = {
            "panels": EnvObjectConfig.panel_asset_params,
            "thin": EnvObjectConfig.thin_asset_params,
            "trees": EnvObjectConfig.tree_asset_params,
            "objects": EnvObjectConfig.object_asset_params,
            "left_wall": EnvObjectConfig.left_wall,
            "right_wall": EnvObjectConfig.right_wall,
            "back_wall": EnvObjectConfig.back_wall,
            "front_wall": EnvObjectConfig.front_wall,
            "bottom_wall": EnvObjectConfig.bottom_wall,
            "top_wall": EnvObjectConfig.top_wall,
        }
```

## 任务

环境规范决定了在独立的模拟实例中填充的内容，以及基于命令动作的集体模拟如何随时间推移。这里的任务略有不同。我们打算使用这个术语来解释从环境中获取特定任务信息。任务类实例化整个模拟及其所有并行机器人和资源，因此可以访问所有模拟信息。我们打算使用这个类来确定环境如何被解释为RL任务。例如，给定的模拟实例具有模拟参数、环境和资源规范、机器人、传感器和控制器规范，可以用于训练一个策略，以执行完全不同的任务。这些任务的示例包括通过杂物的设定点导航、观察模拟中的特定资源、在模拟中栖息在特定资源上等等。所有这些任务都可以在环境中使用相同的对象集执行，但需要对环境数据进行不同的解释，以便适当地训练RL算法。这可以在任务类中完成。任务可以在`config/task_config`文件中指定，如下所示：

??? 示例 "任务配置示例"
```python
class task_config:
    seed = 10
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    args = {}
    num_envs = 2
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 1000  # 模拟的真实物理时间为此值乘以sim.dt
    return_state_before_reset = False
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1/3.5, 1/3.5, 1/3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }

    # a + bx for action scaling
    consant_for_action = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    scale_for_action = torch.tensor([3.0, 3.0, 3.0, 1.50], dtype=torch.float32, device=device)


    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        return task_config.consant_for_action + task_config.scale_for_action * clamped_action     
```

一个用于位置设定点导航（没有传感器或障碍物）的示例任务如下所示：

??? 示例 "`PositionSetpointTask`类定义示例"
```python
class PositionSetpointTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.device = self.task_config.device
        # 将奖励参数的每个元素设置为torch张量
        # 这里的常见样板代码

        # 目前只有“观察”被发送到演员和评论家。
        # “privileged_obs”尚未在sample-factory中处理

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

    # 常见样板代码

    def step(self, actions):
        # 这使用动作，获取观察
        # 计算奖励，返回元组
        # 在这种情况下，需要首先重置已终止的回合，并返回新回合的第一个观察。

        transformed_action = self.action_transformation_function(actions)
        self.sim_env.step(actions=transformed_action)
        
        # 由于在计算奖励后进行重置，因此必须执行此步骤。
        # 这使得机器人能够返回更新的状态，并在重置后向RL代理发送更新的观察。
        # 这对于RL代理在重置后获取正确的状态至关重要。
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0)
        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple
        
    ...

    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]  # 环境/世界坐标系中的位置
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]  # 环境/世界坐标系中的方向
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]  # 在身体/IMU坐标系中的线速度
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]  # 在身体/IMU坐标系中的角速度
        self.task_obs["rewards"] = self.rewards  # 在计算后为时间步长计算的奖励
        self.task_obs["terminations"] = self.terminations  # 终止/碰撞 
        self.task_obs["truncations"] = self.truncations  # 截断或提前重置（用于多样化数据和回合）


@torch.jit.script
def exp_func(x, gain, exp):
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def compute_reward(
    pos_error, crashes, action, prev_action, curriculum_level_multiplier, parameter_dict
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)

    pos_reward = 2.0 / (1.0 + dist * dist)

    dist_reward = (20 - dist) / 20.0

    total_reward = (
        pos_reward + dist_reward  # + up_reward + action_diff_reward + absolute_action_reward
    )
    total_reward[:] = curriculum_level_multiplier * total_reward
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(
        crashes > 0.0, -2 * torch.ones_like(total_reward), total_reward
    )
    return total_reward, crashes
```

任务类最终旨在与RL框架一起使用，因此符合[Gymnasium API](https://gymnasium.farama.org)规范。在上述类中，`step(...)`函数首先将RL代理的命令转换为特定机器人平台的控制命令，通过转换动作输入。随后，这个命令被发送到机器人，环境被推进。最后，为新状态计算奖励，并确定[截断和终止](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)，并返回最终元组供RL框架使用。同样，对于轨迹跟踪，只需更改奖励函数和观察即可训练RL算法，而无需对资源、机器人或环境进行任何更改。

要添加您自己的[自定义任务](./5_customization.md/#custom-tasks)，请参考[自定义模拟器](./5_customization.md)部分。

??? 问题 "**环境与任务的区别**"
许多不同的模拟器实现互换术语。在我们的案例中，我们将环境视为定义机器人及其物理环境的组件，即机器人附近的资源、决定模拟世界中各种实体如何相互作用的物理引擎参数，以及传感器通过传感器参数感知数据的方式。

另一方面，任务是对模拟世界及其提供的信息的解释，以达到用户所期望的特定目标。同一环境可以用于训练多个任务，而任务可以在不更改环境定义的情况下进行更改。

例如，带有四旋翼的空环境可以用于训练位置设定点任务或轨迹跟踪任务。带有一组障碍物的环境可以用于训练一个策略，该策略可以在障碍物之间导航或在环境中的特定资源上栖息。任务是对环境数据的解释，以便RL算法学习所需的行为。

将其与OpenAI Gym任务套件中的熟悉环境联系起来，在我们的案例中，“环境”可以指代具有其相关动态的CartPole世界，而“任务”则允许相同的CartPole被控制以保持杆子直立，或以给定的角速度摆动杆子，或使杆子的端点位于环境中的给定位置。所有这些都需要为RL算法学习所需行为而制定不同的奖励和观察公式。

## 机器人

机器人可以独立于传感器和环境进行指定和配置，使用机器人[注册表](#registries)。有关机器人的更多信息，请参见[机器人和控制器](./3_robots_and_controllers.md/#robots)页面。

## 控制器

控制器可以独立于机器人平台进行指定和选择。然而，请注意，并非所有控制器的组合都能在所有平台上产生最佳结果。控制器可以从控制器注册表中注册和选择。有关控制器的更多信息，请参见[机器人和控制器](./3_robots_and_controllers.md/#controllers)页面。

## 传感器

与上述类似，传感器可以独立指定和选择。然而，由于传感器安装在机器人平台上，我们选择在机器人配置文件中为机器人选择传感器，而不是直接作为注册表（您可以通过非常小的代码更改自行做到这一点）。有关传感器功能的更多信息，请参见[传感器和渲染](./8_sensors_and_rendering.md)页面。