# 开始使用

## 安装

1. 下载 [Isaac Gym Preview 4 Release](https://developer.nvidia.com/isaac-gym/download)
2. 使用以下说明安装 Isaac Gym 模拟器：
    1. 安装一个新的 conda 环境并激活它
        ```bash
        conda create -n aerialgym python=3.8
        conda activate aerialgym
        ```
    2. 安装依赖项
        ```bash
        conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
         pytorch-cuda=11.7 -c pytorch -c conda-forge -c defaults
        conda install pyyaml==6.0 tensorboard==2.13.0 -c conda-forge -c pytorch -c defaults -c nvidia
        # 或者安装最新版本的 PyTorch，使用与您的驱动程序兼容的 CUDA 版本
        # conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
        conda install -c fvcore -c iopath -c conda-forge fvcore iopath
        conda install -c pytorch3d pytorch3d
        ```
    3. 安装 Isaac Gym 及其依赖项
        ```bash
        cd <isaacgym_folder>/python
        pip3 install -e .
        # 设置 Isaac Gym 的环境变量
        export LD_LIBRARY_PATH=~/miniconda3/envs/aerialgym/lib
        # 或者
        export LD_LIBRARY_PATH=~/anaconda3/envs/aerialgym/lib

        # 如果您收到错误消息 "rgbImage buffer error 999"
        # 请设置此环境变量
        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

        # 请将此添加到您的 .bashrc 或 .zshrc 文件中，以避免每次运行模拟器时都要设置环境变量
        ```
    4. 测试安装
        ```bash
        cd <isaacgym_folder>/python/examples
        python3 1080_balls_of_solitude.py
        ```
    如果上述示例运行没有任何错误，则表示 Isaac Gym 安装成功。

    **注意** "在 Isaac Gym 的 `gymutil.py` 中更改参数解析器"
    在安装 Aerial Gym 模拟器之前，需要对 Isaac Gym 安装进行更改。
    Isaac Gym 中的参数解析器会干扰其他学习框架可能需要的其余参数。可以通过将 `isaacgym` 文件夹中 `gymutil.py` 的第 337 行从
    ```python
    args = parser.parse_args()
    ```
    更改为
    ```python
    args, _ = parser.parse_known_args()
    ```
3. 创建工作空间目录并在 conda 环境中安装依赖项
    ```bash
    mkdir -p ~/workspaces/aerial_gym_ws/src && cd ~/workspaces/aerial_gym_ws/src
    # 由于 pip 分发版本中未解决的圆柱网格表示问题，需要从源代码安装 urdfpy 包。
    # 更多信息请参见：https://github.com/mmatl/urdfpy/issues/20
    git clone git@github.com:mmatl/urdfpy.git
    cd urdfpy
    pip3 install -e .
    ```

4. 下载并安装 Aerial Gym 模拟器
    ```bash
    cd ~/workspaces/aerial_gym_ws/src
    git clone git@github.com:ntnu-arl/aerial_gym_simulator.git
    # 或者通过 HTTPS
    # git clone https://github.com/ntnu-arl/aerial_gym_simulator.git

    cd aerial_gym
    pip3 install -e .
    ```
5. 测试示例环境
    ```bash
    cd ~/workspaces/aerial_gym_ws/src/aerial_gym/aerial_gym/examples
    python3 position_control_example.py
    ```

## 运行示例

### 基本环境示例

### 位置控制任务示例

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator/examples
python3 position_control_example.py
```

??? 示例 "位置控制示例代码"
```python
from aerial_gym.utils.logging import CustomLogger
logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch

if __name__ == "__main__":
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=64,
        headless=False,
        use_warp=False # 因为在这个示例中，机器人不应该有摄像头。
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()
    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"步骤 {i}, 更改目标设定点。")
            actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
        env_manager.step(actions=actions)
```

![位置控制示例](./gifs/position_control_example.gif)

上述示例演示了如何创建一个空的模拟环境，选择一个四旋翼机器人，并使用几何位置控制器控制该机器人。在此示例中，`action` 变量被发送到机器人，作为命令的位置和偏航设定点进行跟踪。每 100 次迭代更改为随机位置和偏航设定点。每次迭代都会渲染模拟。

### 渲染和保存图像

```bash
cd <path_to_aerial_gym_simulator>/examples
python3 save_camera_stream.py
```

??? 示例 "渲染和保存图像的代码"
```python
import numpy as np
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch

if __name__ == "__main__":
    logger.debug("这就是调试消息的样子")
    logger.info("这就是信息消息的样子")
    logger.warning("这就是警告消息的样子")
    logger.error("这就是错误消息的样子")
    logger.critical("这就是严重消息的样子")

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",  # "forest_env", #"empty_env", # empty_env
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=True,
    )
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")

    env_manager.reset()
    seg_frames = []
    depth_frames = []
    merged_image_frames = []
    for i in range(10000):
        if i % 100 == 0 and i > 0:
            print("i", i)
            env_manager.reset()
            # 将帧保存为 gif：
            seg_frames[0].save(
                f"seg_frames_{i}.gif",
                save_all=True,
                append_images=seg_frames[1:],
                duration=100,
                loop=0,
            )
            depth_frames[0].save(
                f"depth_frames_{i}.gif",
                save_all=True,
                append_images=depth_frames[1:],
                duration=100,
                loop=0,
            )
            merged_image_frames[0].save(
                f"merged_image_frames_{i}.gif",
                save_all=True,
                append_images=merged_image_frames[1:],
                duration=100,
                loop=0,
            )
            seg_frames = []
            depth_frames = []
            merged_image_frames = []
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        # 重置已崩溃的环境
        env_manager.reset_terminated_and_truncated_envs()
        try:
            image1 = (
                255.0 * env_manager.global_tensor_dict["depth_range_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)
            seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        except Exception as e:
            logger.error("获取图像时出错")
            logger.error("似乎图像张量尚未创建。")
            logger.error("这可能是由于环境中缺少功能性摄像头。")
            raise e
        seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
        seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
            seg_image1.max() - seg_image1.min()
        )

        # 在 matplotlib 中设置色图为 plasma
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
        seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))

        depth_image1 = Image.fromarray(image1)
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1
        image_4d[:, :, 1] = image1
        image_4d[:, :, 2] = image1
        image_4d[:, :, 3] = 255.0
        merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
        # 将帧保存到数组：
        seg_frames.append(seg_image1)
        depth_frames.append(depth_image1)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))

```

机器人传感器可以通过 `env_manager` 的 `global_tensor_dict` 属性访问。在此示例中，深度/范围和分割图像每 100 次迭代保存为 gif。深度图像以灰度图像保存，而分割图像则使用 matplotlib 的 plasma 色图以彩色图像保存。合并图像以 gif 格式保存，深度图像位于上半部分，分割图像位于下半部分。

从摄像头和 LiDAR 传感器的传感器流创建的 gif 如下所示：

![范围和分割图像](./gifs/camera_depth_frames.gif) ![LiDAR 和分割图像](./gifs/lidar_depth_frames.gif)

![范围和分割图像](./gifs/camera_seg_frames.gif) ![LiDAR 和分割图像](./gifs/lidar_seg_frames.gif)

### 强化学习环境示例

```bash
cd <path_to_aerial_gym_simulator>/examples
python3 rl_env_example.py
```

??? 示例 "强化学习接口示例代码"
```python
import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task(
        "position_setpoint_task",
        # 其他参数未在此处设置，使用任务配置文件中的默认值
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,
        )
    ).to("cuda:0")
    actions[:] = 0.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()
```