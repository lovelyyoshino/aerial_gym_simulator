# 常见问题与故障排除

## 常见问题解答

!!! question "预计何时支持 Isaac Lab？"
    我们正在努力在不久的将来支持 Isaac Lab。请关注相关更新。

!!! question "如何将 Isaac Gym 模拟器与我的自定义机器人一起使用？"
    Isaac Gym 模拟器设计为模块化和灵活，允许用户轻松集成自定义机器人。您可以参考文档中的 [自定义机器人集成](./5_customization.md/#custom-robots) 部分，以获取有关如何将自定义机器人与 Isaac Gym 模拟器集成的详细说明。

!!! question "如何随机化机器人上传感器的姿态？"
    通过在传感器配置文件中启用 `randomize_placement` 标志，可以随机化传感器的姿态。然而，这仅适用于 Warp 渲染管道，并且在 Isaac Gym 的本地渲染管道中速度较慢，因为它要求用户循环遍历每个传感器实例。默认情况下，传感器位置在每次环境重置时随机化，但如果您愿意，也可以在每个时间步进行随机化，且开销很小。

!!! question "如何更改机器人生成时的随机姿态？"
    这可以通过在机器人配置文件中设置 `min_init_state` 和 `max_init_state` 参数来实现。默认情况下，机器人的起始姿态在每次环境重置时随机化。根据当前结构，位置是环境边界的比例，方向可以通过最小和最大滚转、俯仰和偏航值来定义。

!!! question "**环境与任务的区别**"
    许多不同的模拟器实现互换了这些术语。在我们的情况下，我们将环境视为定义机器人及其物理环境的组件，即机器人附近的资产、物理引擎的参数，这些参数决定了模拟世界中各种实体如何相互作用，以及传感器如何通过传感器参数感知数据。

    另一方面，任务是对模拟世界及其提供/收集的信息的解释，以达到用户所期望的特定目标。同一环境可以用于训练多个任务，而任务可以在不改变环境定义的情况下进行更改。

    例如，一个空的环境与四旋翼可以用于训练位置设定任务或轨迹跟踪任务。一个带有障碍物的环境可以用于训练一种策略，该策略可以在障碍物之间导航或在环境中的特定资产上栖息。任务是对环境数据的解释，以便强化学习算法学习所需的行为。

    为了与 OpenAI Gym 任务套件中的熟悉环境相关联，在我们的情况下，“环境”可以指代具有 CartPole 动力学的 CartPole 世界，而“任务”则允许同一 CartPole 被控制以保持杆子直立，或以给定的角速度摆动杆子，或将杆子的端点放置在环境中的特定位置。所有这些都需要为强化学习算法制定不同的奖励和观察公式，以学习所需的行为。

## 故障排除

!!! danger "我的 Isaac Gym 观察窗口没有显示任何内容或是空白的"
    这可能是由于您的 NVIDIA 驱动程序版本不一致所致。请确保您的系统上安装了符合 Isaac Gym 文档要求的适当 NVIDIA 驱动程序，并且 Isaac Gym 示例（如 `1080_balls_of_solitude.py` 和 `joint_monkey.py`）能够正常工作。请确保环境变量 `LD_LIBRARY_PATH` 和 `VK_ICD_FILENAMES` 设置正确。

!!! danger "rgbImage 缓冲区错误 999"
    ```bash
    [Error] [carb.gym.plugin] cudaImportExternalMemory failed on rgbImage buffer with error 999
    ```

    这很可能是由于 Vulkan 配置不当所致。请参考 Isaac Gym 文档的故障排除部分，检查 `VK_ICD_FILENAMES` 环境变量是否已设置，并确保该文件存在。

!!! danger "我的模拟资产相互穿透，没有发生碰撞"
    这发生在模拟资产的四元数设置不当的情况下。请检查四元数是否已按 Isaac Gym 模拟器的要求进行归一化，并且格式为 `[q_x, q_y, q_z, q_w]`。

!!! danger "我的模拟资产在每次重置时旋转得非常快"
    您的实现中似乎存在某处的 `nan` 值。这可能是由于多种原因造成的，例如四元数归一化不当或传感器测量不当。请检查您的代码，确保所有测量值有效且在预期范围内。

!!! warning "我看到一个以 `if len(self._meshes) == 0:` 结尾的错误"
    您很可能是通过 pip 安装了 urdfpy 包。此版本存在一个错误，该错误已在 URDFPY 项目存储库的 `master` 分支中解决。请按照 [安装页面](./2_getting_started.md/#installation) 从源代码安装该包。