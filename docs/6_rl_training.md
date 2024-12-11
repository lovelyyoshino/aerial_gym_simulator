# 强化学习

## 基于深度图像的导航任务强化学习

我们提供了一个现成的策略，该策略用于[利用深度碰撞编码实现无碰撞飞行的强化学习](https://arxiv.org/abs/2402.03947)的研究工作。观察空间已重新定义，以匹配论文中所示的示例，并提供了一个脚本以在训练好的策略上运行推理。若要亲自检查该策略的性能，请按照以下步骤操作：

```bash
cd examples/dce_rl_navigation
bash run_trained_navigation_policy.sh
```

现在您应该能够看到训练好的策略在实际应用中的表现：
![导航的强化学习](./gifs/rl_for_navigation_example.gif)

在此任务中，默认使用Warp进行渲染，机器人的深度相机如下面所示观察环境：

![深度流 RL 1](./gifs/depth_frames_example_1.gif) ![深度流 RL 2](./gifs/depth_frames_example_2.gif) 

如果您使用此工作，请引用以下论文：

```bibtex
@misc{kulkarni2024reinforcementlearningcollisionfreeflight,
      title={Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding}, 
      author={Mihir Kulkarni and Kostas Alexis},
      year={2024},
      eprint={2402.03947},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2402.03947}, 
}
```

## 使用Aerial Gym模拟器为您的机器人训练策略

我们提供了与[rl-games](#rl-games)、[sample-factory](#sample-factory)和[CleanRL](#cleanrl)框架的现成示例，并附有相关脚本。模拟器的[`Task`](./4_simulation_components.md/#tasks)定义允许与模拟器进行最简化的集成，使开发者/用户能够专注于设计合适的模拟环境，而不是花时间将环境与RL训练框架集成。

### 训练您自己的导航策略

类似的任务配置可以用于启用导航控制策略，但机器人需要使用机载相机传感器模拟外部感知传感器数据。此更改发生在机器人端，并需要启用机器人的相机/LiDAR传感器。

`config/robot_config/base_quad_config.py`或相应的机器人配置文件需要修改以启用相机传感器。相机传感器可以如下启用：
```python
class BaseQuadCfg:
    ...
    class sensor_config:
        enable_camera = True  # 当不需要相机传感器时为False
        camera_config = BaseDepthCameraConfig

        enable_lidar = False
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig
    ...
```

随后，可以使用`rl_training`文件夹中提供的训练脚本训练各种算法。

使用`rl_games`的示例：
```bash
### 为四旋翼训练带有速度控制的导航策略
python3 runner.py --file=./ppo_aerial_quad_navigation.yaml --num_envs=1024 --headless=True

### 在模拟中重放训练好的策略
python3 runner.py --file=./ppo_aerial_quad_navigation.yaml --num_envs=16 --play --checkpoint=./runs/<weights_file_path>/<weights_filename>.pth
```

在单个NVIDIA RTX 3090 GPU上，训练大约需要一个小时。`navigation_task`默认使用深度碰撞编码器来表示考虑到障碍物的潜在表示，这些障碍物是由机器人尺寸“膨胀”而成的。

### 训练您自己的位置控制策略

我们提供了现成的任务定义，以训练各种机器人的位置控制策略。任务定义保持不变——无论配置如何，奖励函数可以根据用户在性能方面的特定需求（如控制的平滑性、能效等）进行修改。提供的RL算法将学习为机载控制器生成控制命令或直接向机器人提供电机命令。

要为各种机器人使用各种机载控制器训练位置设定策略，请按如下方式配置`config/task_config/position_setpoint_task_config.py`：
```python
class task_config:
    seed = -1  # 在此设置您的种子。-1将随机化种子
    sim_name = "base_sim"

    env_name = "empty_env"
    # env_name = "env_with_obstacles"
    # env_name = "forest_env"

    robot_name = "base_quadrotor"
    # robot_name = "base_fully_actuated"
    # robot_name = "base_random"

    controller_name = "lee_attitude_control"
    # controller_name = "lee_acceleration_control"
    # controller_name = "no-control"
    ...
```

要使用`rl_games`训练策略，请运行以下命令：
```bash
# 训练策略
python3 runner.py --task=position_setpoint_task --num_envs=8192 --headless=True --use_warp=True

# 重放训练好的策略
python3 runner.py --task=position_setpoint_task --num_envs=16 --headless=False --use_warp=True --checkpoint=<path_to_checkpoint_weights>.pth --play
```

该策略在单个NVIDIA RTX 3090 GPU上训练时间不到一分钟。训练得到的策略如下所示：

![位置控制的强化学习](./gifs//rl_for_position.gif)

## rl-games

模拟的[`Task`](./4_simulation_components.md/#tasks)实例需要用框架特定的环境包装器进行包装，以支持并行环境。

??? 示例 "rl-games包装器示例"
```python
class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(terminated | truncated, torch.ones_like(terminated), torch.zeros_like(terminated))

        return (
            observations["observations"],
            rewards,
            dones,
            infos,
        )


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print("AERIALRLGPUEnv", config_name, num_actors, kwargs)
        print(env_configurations.configurations)
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            np.ones(self.env.task_config.action_space_dim) * -1.0, np.ones(self.env.task_config.action_space_dim) * 1.0
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf, np.ones(self.env.task_config.observation_space_dim) * np.Inf
        )

        print(info["action_space"], info["observation_space"])

        return info
```

在这里，环境被包装在`AERIALRLGPUEnv`实例中。`ExtractObsWrapper`是一个[gym](https://gymnasium.farama.org/)包装器，允许从环境中提取观察值。虽然这并不是必需的，因为我们的[`Task`](./4_simulation_components.md/#tasks)允许这种灵活性，但我们保留了这种结构以保持与我们之前发布和其他实现的一致性。

以下是使用姿态控制的四旋翼位置设定任务的rl-games训练包装器示例：

![](./gifs/rl-games-training-v2.gif)

同样，使用电机命令的完全驱动八旋翼和随机配置（8个电机）的位置信息设定任务示例如下：

![](./gifs/rl-games-fully-actuated-training-v2.gif)

![](./gifs/rl-games-random-configuration-training-v2.gif)

## Sample Factory

与上述描述类似，[Sample Factory](https://github.com/alex-petrenko/sample-factory)的集成需要一个[gym](https://gymnasium.farama.org/)包装器。实现如下：

??? 示例 "Sample Factory包装器示例"
```python
class AerialGymVecEnv(gym.Env):
    '''
    Aerial Gym环境的包装器，使其与Sample Factory兼容。
    '''
    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.action_space = convert_space(self.env.action_space)

        # Aerial Gym示例环境实际上返回字典
        if obs_key == "obs":
            self.observation_space = gym.spaces.Dict(convert_space(self.env.observation_space))
        else:
            raise ValueError(f"未知的观察键: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # 一些IGE环境在第一时间步返回全零，但这可能是可以接受的
        obs, rew, terminated, truncated, infos = self.env.reset()
        return obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        return obs, rew, terminated, truncated, infos

    def render(self):
        pass


def make_aerialgym_env(full_task_name: str, cfg: Config, _env_config=None, render_mode: Optional[str] = None) -> Env:
    return AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs")
```

以下是使用姿态控制的四旋翼位置设定任务的示例：

![](./gifs/sample-factory-training-v2.gif)

## CleanRL

任务定义为RL提供的灵活性使其可以直接与CleanRL框架一起使用，无需更改：
??? 示例 "CleanRL包装器示例"
```python
# 环境设置
envs = task_registry.make_task(task_name=args.task)

envs = RecordEpisodeStatisticsTorch(envs, device)

print("动作数量: ", envs.task_config.action_space_dim)
print("观察数量: ", envs.task_config.observation_space_dim)
```

以下是使用姿态控制的四旋翼位置设定任务的示例：

![](./gifs/cleanrl-training-v2.gif)

## 添加您自己的RL框架

请参考现有的[sample-factory](#sample-factory)、[rl-games](#rl-games)和[CleanRL](#cleanrl)实现。我们非常乐意将您的实现纳入模拟器，使其对不同需求/偏好的用户更具可访问性。如果您希望为此存储库做出贡献，请在[GitHub上提交拉取请求](https://github.com/ntnu-arl/aerial_gym_simulator/compare)。