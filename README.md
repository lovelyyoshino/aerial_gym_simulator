# [Aerial Gym Simulator](index.md)

欢迎来到[Aerial Gym Simulator](https://www.github.com/ntnu-arl/aerial_gym_simulator)的仓库。请参考我们的[文档](https://ntnu-arl.github.io/aerial_gym_simulator/)以获取有关如何开始使用模拟器以及如何将其应用于您的研究的详细信息。

Aerial Gym Simulator是一个高保真、基于物理的模拟器，旨在训练微型无人机（MAV）平台，如多旋翼飞行器，利用基于学习的方法学习飞行和在复杂环境中导航。该环境基于底层的[NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)模拟器构建。我们提供标准平面四旋翼平台的空中机器人模型，以及具有任意配置的全驱动平台和多旋翼飞行器。这些配置支持低级和高级几何控制器，这些控制器驻留在GPU上，并为数千个多旋翼飞行器的同时控制提供并行化。

这是模拟器的*第二个版本*，包括多种新功能和改进。任务定义和环境配置允许对所有环境实体进行细粒度的定制，而无需处理大型单一环境文件。自定义渲染框架允许以高速获取深度和分割图像，并可用于模拟具有不同属性的自定义传感器，如激光雷达。该模拟器是开源的，并根据[BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause)发布。

Aerial Gym Simulator允许您在不到一分钟的时间内训练基于状态的控制策略：

![Aerial Gym Simulator](./docs/gifs/Aerial%20Gym%20Position%20Control.gif)

并在不到一小时的时间内训练基于视觉的导航策略：

![RL for Navigation](./docs/gifs/rl_for_navigation_example.gif)

配备了GPU加速和可定制的基于光线投射的激光雷达和相机传感器，具有深度和分割能力：

![Depth Frames 1](./docs/gifs/camera_depth_frames.gif) ![Lidar Depth Frames 1](./docs/gifs/lidar_depth_frames.gif)

![Seg Frames 1](./docs/gifs/camera_seg_frames.gif) ![Lidar Seg Frames 1](./docs/gifs/lidar_seg_frames.gif)

## 特性

- **模块化和可扩展设计**，允许用户轻松创建自定义环境、机器人、传感器、任务和控制器，并通过修改[模拟组件](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components)以编程方式动态更改参数。
- **从头重写**，提供对每个模拟组件的高度控制，并能够广泛[定制](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization)模拟器以满足您的需求。
- **高保真物理引擎**，利用[NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download)，为模拟多旋翼平台提供高保真物理引擎，并可以添加对自定义物理引擎后端和渲染管道的支持。
- **并行几何控制器**，驻留在GPU上，为[数千个多旋翼](https://ntnu-arl.github.io/aerial_gym_simulator/3_robots_and_controllers/#controllers)飞行器的同时控制提供并行化。
- **自定义渲染框架**（基于[NVIDIA Warp](https://nvidia.github.io/warp/)）用于设计[自定义传感器](https://ntnu-arl.github.io/aerial_gym_simulator/8_sensors_and_rendering/#warp-sensors)并执行并行化的基于内核的操作。
- **模块化和可扩展**，允许用户轻松创建[自定义环境](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-environments)、[机器人](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-robots)、[传感器](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-sensors)、[任务](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-tasks)和[控制器](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-controllers)。
- **可添加基于RL的控制和导航策略**，用于机器人学习任务。[包括用于开始训练您自己的机器人的脚本。](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training)

> [!重要]
> 对[**Isaac Lab**](https://isaac-sim.github.io/IsaacLab/)和[**Isaac Sim**](https://developer.nvidia.com/isaac/sim)的支持目前正在开发中。我们预计将在不久的将来发布此功能。

请参考详细介绍我们模拟器先前版本的论文，以获取有关创建Aerial Gym Simulator的动机和设计原则的见解：[https://arxiv.org/abs/2305.16510](https://arxiv.org/abs/2305.16510)（链接将很快更新以反映新版本！）。

## 为什么选择Aerial Gym Simulator？

Aerial Gym Simulator旨在同时模拟数千个MAV，并配备了在现实世界系统中使用的低级和高级控制器。此外，新的自定义光线投射允许以超快的速度渲染环境，以便使用来自环境的深度和分割进行任务。

此新版本中的优化代码允许在不到一分钟的时间内训练用于机器人控制的电机命令策略，并在不到一小时的时间内训练基于视觉的导航策略。提供了大量示例，以便用户快速开始训练自定义机器人的策略。

## 引用
在您的研究中引用Aerial Gym Simulator时，请引用以下论文：

```bibtex
@misc{kulkarni2023aerialgymisaac,
      title={Aerial Gym -- Isaac Gym Simulator for Aerial Robots}, 
      author={Mihir Kulkarni and Theodor J. L. Forgaard and Kostas Alexis},
      year={2023},
      eprint={2305.16510},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2305.16510}, 
}
```

如果您在导航任务中使用了与此模拟器一起提供的强化学习策略，请引用以下论文：

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

## 快速链接
为了方便您，这里有一些指向文档中最重要部分的快速链接：

- [安装](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/#installation)
- [机器人和控制器](https://ntnu-arl.github.io/aerial_gym_simulator/3_robots_and_controllers)
- [传感器和渲染能力](https://ntnu-arl.github.io/aerial_gym_simulator/8_sensors_and_rendering)
- [强化学习训练](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training)
- [模拟组件](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components)
- [定制化](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization)
- [常见问题和故障排除](https://ntnu-arl.github.io/aerial_gym_simulator/7_FAQ_and_troubleshooting)

## 联系方式

Mihir Kulkarni  &nbsp;&nbsp;&nbsp; [电子邮件](mailto:mihirk284@gmail.com) &nbsp; [GitHub](https://github.com/mihirk284) &nbsp; [领英](https://www.linkedin.com/in/mihir-kulkarni-6070b6135/) &nbsp; [X（前身为Twitter）](https://twitter.com/mihirk284)

Welf Rehberg &nbsp;&nbsp;&nbsp;&nbsp; [电子邮件](mailto:welf.rehberg@ntnu.no) &nbsp; [GitHub](https://github.com/Zwoelf12) &nbsp; [领英](https://www.linkedin.com/in/welfrehberg/)

Theodor J. L. Forgaard &nbsp;&nbsp;&nbsp; [电子邮件](mailto:tjforgaa@stud.ntnu.no) &nbsp; [GitHub](https://github.com/tforgaard) &nbsp; [领英](https://www.linkedin.com/in/theodor-johannes-line-forgaard-665b5311a/)

Kostas Alexis &nbsp;&nbsp;&nbsp;&nbsp; [电子邮件](mailto:konstantinos.alexis@ntnu.no) &nbsp;  [GitHub](https://github.com/kostas-alexis) &nbsp; 
 [领英](https://www.linkedin.com/in/kostas-alexis-67713918/) &nbsp; [X（前身为Twitter）](https://twitter.com/arlteam)

该工作在[挪威科技大学（NTNU）](https://www.ntnu.no)的[自主机器人实验室](https://www.autonomousrobotslab.com)进行。有关更多信息，请访问我们的[网站](https://www.autonomousrobotslab.com/)。

## 致谢
本材料得到了RESNAV（AFOSR奖号：FA8655-21-1-7033）和SPEAR（地平线欧洲资助协议号：101119774）的支持。

本仓库利用了一些来自[https://github.com/leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)和[IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)的代码和辅助脚本。

## 常见问题和故障排除 

请参考我们的[网站](https://ntnu-arl.github.io/aerial_gym_simulator/7_FAQ_and_troubleshooting/)或GitHub仓库中的[问题](https://github.com/ntnu-arl/aerial_gym_simulator/issues)部分以获取更多信息。

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
│   │   └── task_config                  # 任务配置文件(会调用上面的机器人、环境等配置，并加入强化学习的一些策略)
│   ├── control
│   │   ├── control_allocation.py        # 控制分配算法实现
│   │   ├── controllers                  # 各种控制器的实现（主要是给仿真使用计算）
│   │   ├── __init__.py                  # 初始化控制模块
│   │   └── motor_model.py               # 电机模型的实现
│   ├── env_manager     # env_manager->robots->sensors
│   │   ├── asset_loader.py              # 资产加载器
│   │   ├── asset_manager.py             # 资产管理器
│   │   ├── base_env_manager.py          # 基础环境管理器
│   │   ├── env_manager.py               # 环境管理主逻辑
│   │   ├── IGE_env_manager.py           # IGE环境管理器
│   │   ├── IGE_viewer_control.py        # IGE查看器控制
│   │   ├── __init__.py                  # 初始化环境管理模块
│   │   ├── obstacle_manager.py          # 障碍物管理器
│   │   └── warp_env_manager.py          # Warp环境管理器
│   ├── examples      # examples->task->sim->env_manager->robots->sensors(再细化)
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
│   ├── rl_training     # rl_training->task->sim->env_manager->robots->sensors(训练好给example用)
│   │   ├── cleanrl                      # CleanRL库实现
│   │   ├── rl_games                     # RL Games实现
│   │   └── sample_factory               # Sample Factory实现
│   ├── robots     # robots->sensors
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
│   ├── sim    # sim->env_manager->robots->sensors
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
│   ├── task    # task->sim->env_manager->robots->sensors
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
基本流程就是rl会强化学习在场景中的策略，然后再task中为策略对应的任务。在学习完毕后会在example中调用策略完成实际使用