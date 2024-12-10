# Isaac Gym Environments for Legged Robots #
This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.  

**Maintainer**: Nikita Rudin  
**Affiliation**: Robotic Systems Lab, ETH Zurich  
**Contact**: rudinn@ethz.ch  

---

### :bell: Announcement (09.01.2024) ###

With the shift from Isaac Gym to Isaac Sim at NVIDIA, we have migrated all the environments from this work to [Isaac Lab](https://github.com/isaac-sim/IsaacLab). Following this migration, this repository will receive limited updates and support. We encourage all users to migrate to the new framework for their applications.

Information about this work's locomotion-related tasks in Isaac Lab is available [here](https://isaac-sim.github.io/IsaacLab/source/features/environments.html#locomotion).

---

### Useful Links ###

Project website: https://leggedrobotics.github.io/legged_gym/   
Paper: https://arxiv.org/abs/2109.11978

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one containing  all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=anymal_c_flat```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=anymal_c_flat```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and has no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resources/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`. It is also possible that you need to do `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(for conda user. Replace /path/to/ to the corresponding path.).

### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesirable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from the reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```


```bash
├── aerial_gym
│   ├── assets
│   │   ├── base_asset.py               # 定义基本的资产类
│   │   ├── isaacgym_asset.py           # 处理与Isaac Gym相关的资产
│   │   └── warp_asset.py               # 处理Warp相关的资产
│   ├── config
│   │   ├── asset_config                 # 资产配置文件(主要是URDF模型)
│   │   ├── controller_config            # 控制器配置文件
│   │   ├── env_config                   # 环境配置文件(会调用asset_config当中的配置来构成环境)
│   │   ├── robot_config                 # 机器人配置文件，会调用sensor_config这个传感器配置
│   │   ├── sensor_config                # 传感器配置文件（包括相机、IMU、雷达的传感器）
│   │   ├── sim_config                   # 仿真配置文件
│   │   └── task_config                  # 任务配置文件
│   ├── control
│   │   ├── control_allocation.py        # 控制分配算法实现
│   │   ├── controllers                  # 各种控制器的实现
│   │   ├── __init__.py                  # 初始化控制模块
│   │   └── motor_model.py               # 电机模型的实现
│   ├── env_manager
│   │   ├── asset_loader.py              # 资产加载器
│   │   ├── asset_manager.py             # 资产管理器
│   │   ├── base_env_manager.py          # 基础环境管理器
│   │   ├── env_manager.py               # 环境管理主逻辑
│   │   ├── IGE_env_manager.py           # IGE环境管理器
│   │   ├── IGE_viewer_control.py        # IGE查看器控制
│   │   ├── __init__.py                  # 初始化环境管理模块
│   │   ├── obstacle_manager.py          # 障碍物管理器
│   │   └── warp_env_manager.py          # Warp环境管理器
│   ├── examples
│   │   ├── acceleration_control_example.py # 加速度控制示例
│   │   ├── benchmark.py                 # 性能基准测试
│   │   ├── dce_rl_navigation            # DCE强化学习导航示例
│   │   ├── dynamic_env_example.py       # 动态环境示例
│   │   ├── imu_data_collection.py       # IMU数据收集示例
│   │   ├── inference_example.py         # 推断示例
│   │   ├── morphy_soft_arm_example.py   # Morphy软臂示例
│   │   ├── navigation_task_example.py    # 导航任务示例
│   │   ├── position_control_example_morphy.py # Morphy位置控制示例
│   │   ├── position_control_example.py   # 位置控制示例
│   │   ├── position_control_example_rov.py # ROV位置控制示例
│   │   ├── rl_env_example.py            # 强化学习环境示例
│   │   ├── rl_games_example             # 强化学习游戏示例
│   │   ├── save_camera_stream_normal_faceID.py # 保存正常人脸ID的摄像头流
│   │   ├── save_camera_stream.py        # 保存摄像头流
│   │   ├── shape_control_example_reconfigurable.py # 可重构形状控制示例
│   │   ├── stored_data                  # 存储的数据目录
│   │   └── sys_id.py                    # 系统识别示例
│   ├── __init__.py                      # 初始化包
│   ├── pyproject.toml                   # 项目依赖和配置信息
│   ├── registry
│   │   ├── controller_registry.py       # 控制器注册表
│   │   ├── env_registry.py              # 环境注册表
│   │   ├── robot_registry.py            # 机器人注册表
│   │   ├── sim_registry.py              # 仿真注册表
│   │   └── task_registry.py             # 任务注册表
│   ├── rl_training
│   │   ├── cleanrl                      # CleanRL库实现
│   │   ├── rl_games                     # RL Games实现
│   │   └── sample_factory               # Sample Factory实现
│   ├── robots
│   │   ├── base_multirotor.py           # 基础多旋翼机器人类
│   │   ├── base_reconfigurable.py       # 基础可重构机器人类
│   │   ├── base_robot.py                 # 基础机器人类
│   │   ├── base_rov.py                  # 基础ROV类
│   │   ├── __init__.py                  # 初始化机器人模块
│   │   ├── morphy.py                    # Morphy机器人类
│   │   └── robot_manager.py             # 机器人管理器
│   ├── sensors
│   │   ├── base_sensor.py               # 基础传感器类
│   │   ├── imu_sensor.py                # IMU传感器类
│   │   ├── isaacgym_camera_sensor.py    # Isaac Gym摄像头传感器类
│   │   └── warp                          # Warp相关传感器
│   ├── sim
│   │   ├── __init__.py                  # 初始化仿真模块
│   │   └── sim_builder.py               # 仿真构建器
│   ├── sim2real
│   │   ├── config.py                    # Sim2Real配置
│   │   ├── nn_inference_class.py        # 神经网络推断类
│   │   ├── sample_factory_inference.py   # Sample Factory推断实现
│   │   ├── sample_factory_ros_node.py   # Sample Factory ROS节点
│   │   ├── vae_image_encoder.py         # VAE图像编码器
│   │   ├── vae.py                       # VAE相关实现
│   │   └── weights                      # 权重文件
│   ├── task
│   │   ├── base_task.py                 # 基础任务类
│   │   ├── custom_task                  # 自定义任务
│   │   ├── __init__.py                  # 初始化任务模块
│   │   ├── navigation_task              # 导航任务
│   │   ├── position_setpoint_task       # 位置设定点任务
│   │   ├── position_setpoint_task_acceleration_sim2real # 加速度设定点任务
│   │   ├── position_setpoint_task_morphy # Morphy位置设定点任务
│   │   ├── position_setpoint_task_reconfigurable # 可重构位置设定点任务
│   │   └── position_setpoint_task_sim2real # Sim2Real位置设定点任务
│   └── utils
│       ├── calculate_mixing_matrix      # 计算混合矩阵的工具
│       ├── curriculum_manager.py         # 课程管理器
│       ├── helpers.py                   # 辅助函数
│       ├── imu_to_rosbag.py             # IMU数据转换为ROS包
│       ├── __init__.py                  # 初始化工具模块
│       ├── logging.py                   # 日志记录工具
│       ├── math.py                      # 数学相关工具
│       ├── real_robot_sysid.py          # 真实机器人系统识别
│       ├── tensor_pid.py                # 张量PID控制器
│       └── vae                          # VAE相关工具
├── LICENSE                               # 项目的许可证文件
├── mkdocs.yml                            # MkDocs配置文件
├── pyproject.toml                       # 项目依赖和配置信息
├── README.md                             # 项目的自述文件
├── requirements.txt                     # 项目依赖的Python包
├── resources
│   ├── models
│   │   └── environment_assets           # 环境资产模型
│   └── robots
│       ├── BlueROV                      # BlueROV机器人资源
│       ├── lmf2                         # LMF2机器人资源
│       ├── morphy                       # Morphy机器人资源
│       ├── octarotor                    # Octarotor机器人资源
│       ├── quad                         # 四旋翼机器人资源
│       ├── random                       # 随机机器人资源
│       ├── snakey                       # Snakey机器人资源
│       ├── snakey5                      # Snakey5机器人资源
│       └── snakey6                      # Snakey6机器人资源
└── setup.py                             # 安装脚本
```