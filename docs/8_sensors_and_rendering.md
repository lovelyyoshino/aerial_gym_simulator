# 传感器与渲染

## 在仿真中启用传感器

可以通过编辑机器人配置文件来选择并启用或禁用每个机器人的传感器：
```python
class sensor_config:
    enable_camera = True
    camera_config = BaseDepthCameraConfig

    enable_lidar = False
    lidar_config = BaseLidarConfig  # OSDome_64_Config

    enable_imu = False
    imu_config = BaseImuConfig
```

## 扭曲传感器

我们使用 NVIDIA Warp 开发了一套传感器。我们创建了基于光线投射的传感器的自定义实现，以从环境中获取深度、范围、点云、分割和表面法线信息。以下是开箱即用的各种传感器及其可用的自定义选项。

一些常见的可配置参数如下所述：

```python
height = 128 # 图像高度或 LiDAR 传感器中的扫描线数量
width = 512 # 图像宽度或 LiDAR 传感器中每条扫描线的点数
horizontal_fov_deg_min = -180 # 最小水平视场角（FoV），单位为度
horizontal_fov_deg_max = 180 # 最大水平视场角（FoV），单位为度
vertical_fov_deg_min = -45 # 最小垂直视场角（FoV），单位为度
vertical_fov_deg_max = +45 # 最大垂直视场角（FoV），单位为度
max_range = 10.0 # 最大范围
min_range = 0.2 # 最小范围
calculate_depth = True # 计算深度图像 / 设置为 False 将返回范围（仅适用于相机传感器）
return_pointcloud = False  # 返回点云 [x,y,z] 而不是图像
pointcloud_in_world_frame = False # 点云以机器人或世界坐标系表示
segmentation_camera = True  # 也返回分割图像
normalize_range = True  # 根据传感器范围限制归一化值 0 和 1。
```

### 深度相机

深度相机传感器提供有关环境的深度信息。该传感器可以配置不同的视场（FOV）和分辨率设置。深度相机可用于生成深度图像、点云和表面法线。

对于该传感器模型，垂直和水平视场角之间的关系为

$$\textrm{vfov} = 2 \arctan(\tan(\textrm{hfov}/2) * \textrm{height}/\textrm{width}).$$

深度图像随后作为一个张量 `[num_envs, num_sensor_per_robot, vertical_pixels, horizontal_pixels]` 返回，其中最后两个维度是深度图像的垂直和水平像素。Aerial Gym Simulator 还可以在传感器配置文件中设置 `calculate_pointcloud = True` 时返回环境的点云。这将作为一个 4D 张量 `[num_envs, num_sensor_per_robot, vertical_pixels, horizontal_pixels, 3]` 返回，其中最后一个维度是点的 3D 坐标。

!!!注意
    当返回范围/深度图像时，信息以维度 `[num_envs, num_sensor_per_env, vertical_pixels, horizontal_pixels]` 返回，然而如果选择了 [分割相机](#segmentation-camera) 选项，则会初始化另一个具有相同维度的张量以存储此信息。如果选择以点云形式返回数据，则数据将作为 4D 张量 `[num_envs, num_sensor_per_env, vertical_pixels, horizontal_pixels, 3]` 返回，其中最后一个维度是世界坐标系中点的 3D 坐标。

以下是使用该传感器获得的深度图像的一些示例：

![Depth Image D455 1](./gifs/d455_camera_depth_frames_100.gif) ![Depth Image D455 2](./gifs/d455_camera_depth_frames_200.gif)

### LiDAR

LiDAR 传感器是根据 Ouster OS0-128 传感器的配置建模的。该传感器可以配置不同的视场、分辨率、范围和噪声设置。该传感器允许返回范围图像和环境的点云，以及分割图像和表面法线。与许多流行的现有实现类似，该实现会在物理步骤执行后对所有光线进行光线投射。这并未考虑 LiDAR 的旋转和机器人在 LiDAR 一次扫描过程中的位置变化。

使用理想化的 LiDAR 传感器进行光线投射。每条光线的方向是使用传感器的水平和垂直视场角计算的。光线方向的计算如下：

$$ \textrm{ray}[i, j] = \begin{bmatrix} \cos(\phi) \times \cos(\theta) \\ \sin(\phi) \times \cos(\theta) \\ \sin(\theta) \end{bmatrix}, $$

其中 $\phi$ 是方位角，$\theta$ 是仰角。方位角和仰角在传感器的水平和垂直方向上分别从最大值到最小值变化。

??? 示例 "初始化 LiDAR 传感器的光线方向的代码"
```python
for i in range(self.num_scan_lines):
    for j in range(self.num_points_per_line):
        # 光线从 +HFoV/2 到 -HFoV/2 和 +VFoV/2 到 -VFoV/2
        azimuth_angle = self.horizontal_fov_max - (
            self.horizontal_fov_max - self.horizontal_fov_min
        ) * (j / (self.num_points_per_line - 1))
        elevation_angle = self.vertical_fov_max - (
            self.vertical_fov_max - self.vertical_fov_min
        ) * (i / (self.num_scan_lines - 1))
        ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)
        ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)
        ray_vectors[i, j, 2] = math.sin(elevation_angle)
# 归一化 ray_vectors
ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)
```

对于 [深度相机](#depth-camera) 和 [LiDAR](#lidar) 传感器，测量可以作为范围和/或深度图像以及点云返回。通过在传感器配置文件中设置参数 `pointcloud_in_world_frame = True`，点云可以在世界坐标系或传感器坐标系中返回。

以下是 LiDAR 渲染的示例：

![Ouster OS0 1](./gifs/ouster_lidar_depth_frames_100.gif) ![Ouster OS0 2](./gifs/ouster_lidar_depth_frames_200.gif)

### 分割相机

分割相机与深度或 LiDAR 传感器结合提供。分割相机提供环境的分割图像。在 Isaac Gym 渲染框架中，分割信息可以嵌入环境中每个资产的每个链接中，但为了实现更快的渲染和更大的灵活性，我们允许我们的 Warp 环境表示包含每个网格顶点的分割信息。实际上，这种方法劫持了网格中每个顶点的速度场，以编码分割信息。然后，从与传感器原点发出的光线相交的每个面的特定顶点查询分割信息。虽然我们提供了与 Isaac Gym 渲染框架中提供的功能相当的能力，但这可以轻松扩展以包括更多信息，或者具有更复杂的分割信息，可以分配给与网格中特定面（三角形）相关的顶点。

分割相机与深度/范围或 LiDAR 传感器相关联，只有通过在传感器配置文件中设置标志 `segmentation_camera = True` 才能启用。此外，分割相机直接查询网格以读取顶点数据的 `velocities` 字段，并使用面的第一个顶点的第一个元素（x 速度场）来编码和读取来自网格面的分割信息。

??? 示例 "分割相机内核代码"
```python
if wp.mesh_query_ray(
    mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f
):
    dist = t
    mesh_obj = wp.mesh_get(mesh)
    face_index = mesh_obj.indices[f * 3]
    segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
if pointcloud_in_world_frame:
    pixels[env_id, cam_id, scan_line, point_index] = (
        ray_origin + dist * ray_direction_world
    )
else:
    pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir
segmentation_pixels[env_id, cam_id, scan_line, point_index] = segmentation_value
```

分割相机对应的深度图像输出如下所示：

![Depth Image D455 1](./gifs/d455_camera_depth_frames_100.gif) ![Depth Image D455 2](./gifs/d455_camera_depth_frames_200.gif)

![segmentation Image D455 1](./gifs/d455_camera_seg_frames_100.gif) ![segmentation Image D455 2](./gifs/d455_camera_seg_frames_200.gif)

同样，对于 LiDAR 传感器，范围和分割图像输出如下所示：

![Ouster OS0 1](./gifs/ouster_lidar_depth_frames_100.gif) ![Ouster OS0 2](./gifs/ouster_lidar_depth_frames_200.gif)

![segmentation Image D455 1](./gifs/ouster_lidar_seg_frames_100.gif) ![segmentation Image D455 2](./gifs/ouster_lidar_seg_frames_200.gif)

### 自定义传感器
虽然提供了实现深度相机和 LiDAR 传感器的框架，但可以根据用户的需求进行修改以模拟自定义传感器。例如，可以设计并在仿真中使用 ToF 相机和单光束范围传感器。或者，可以修改提供的实现以专门获取分割信息或场景流信息。

### 随机化传感器位置

此外，传感器可以随机放置和定向。默认情况下，这在每次 `<sensor>.reset()` 时发生，然而，与 Isaac Gym 传感器不同，用户也可以在每个时间步进行此操作，而不会引入任何延迟。随机化传感器位置和方向的参数如下所示，可以在相应的配置文件中更改：

??? 示例 "随机化传感器位置和方向"
```python
# 随机化传感器的位置
randomize_placement = True
min_translation = [0.07, -0.06, 0.01] # [m] # 传感器在 [x, y, z] 轴上的最大平移限制
max_translation = [0.12, 0.03, 0.04] # [m] # 传感器在 [x, y, z] 轴上的最小平移限制
min_euler_rotation_deg = [-5.0, -5.0, -5.0] # [deg] # 传感器在 [x, y, z] 轴上的最小旋转限制
max_euler_rotation_deg = [5.0, 5.0, 5.0] # [deg] # 传感器在 [x, y, z] 轴上的最大旋转限制

# 标称位置和方向（仅适用于 Isaac Gym 相机传感器）
# 这是因为在每次重置时随机化相机位置是缓慢的
nominal_position = [0.10, 0.0, 0.03] # [m] # 传感器在机器人坐标系中的标称位置
nominal_orientation_euler_deg = [0.0, 0.0, 0.0] # [deg] # 传感器在机器人坐标系中的标称方向
```

### 模拟传感器噪声

我们提供了两种传感器噪声模型。第一种是像素掉落模型，其中图像中的一个像素以概率 `pixel_dropout_prob` 随机设置为零。第二种是像素噪声模型，其中像素值受到高斯噪声的扰动，标准差为 `pixel_std_dev_multiplier` 倍的像素值。传感器噪声模型的参数如下所示：

```python
class sensor_noise:
    enable_sensor_noise = False
    pixel_dropout_prob = 0.01
    pixel_std_dev_multiplier = 0.01
```

!!! 问题 "我应该使用 Warp 还是 Isaac Gym 传感器？"
    这取决于您的使用案例。如果您正在模拟动态环境，则需要使用 Isaac Gym 传感器。如果您使用的是两个框架都提供的实现，并且只想要更快的仿真速度，强烈建议使用 Warp。一般来说，Warp 通常运行得更快，尤其是在模拟每个机器人多个低维传感器（例如 8x8 ToF 传感器）时，我们观察到速度提升的数量级，而在处理环境中许多复杂网格时，Isaac Gym 传感器的表现略好。

    然而，Isaac Gym 传感器无法模拟 LiDAR 或任何自定义传感器模型。如果您需要更好的可定制性和随机化传感器参数（如姿态或传感器投影模型）的能力，那么我们的 Warp 实现是推荐的选择。在这里，位置、方向、相机矩阵、光线投射方向等可以在每个时间步进行更改或随机化（哇哦！），而不会引入太多延迟。如果您希望在环境中嵌入额外的信息，以便传感器可以查询，那么我们的 Warp 实现是一个自然的选择，因为这允许一个极其强大的渲染框架。

### Isaac Gym 相机传感器

!!! 警告 "同时使用 Isaac Gym 和 Warp 渲染"
    同时启用 Warp 和 Isaac Gym 渲染管道可能会导致仿真速度下降，并且尚未经过广泛测试。建议一次只使用一个渲染管道。

我们提供了包装器，以便在仿真器中与机器人一起使用 Isaac Gym 相机。这些相机传感器是直接从 Isaac Gym 接口提供的。提供的传感器包括 RGB、深度、分割和光流相机。Isaac Gym Simulator 并未提供我们与基于 Warp 的传感器所提供的所有自定义选项。然而，我们提供了一个标准化的接口，以便在仿真器中与机器人一起使用这些传感器。

### IMU 传感器

实现了一个 IMU 传感器，可以计算机器人的加速度和角速度。该传感器默认配置为安装在机器人基链接的原点，方向是可配置的。这是因为 IMU 是使用 Isaac Gym 中的力传感器实现的，无法考虑向心力和陀螺效应。IMU 测量的获取方式如下：

$$ a_{\textrm{meas}} = a_{\textrm{true}} + b_a + n_a, $$

$$ \omega_{\textrm{meas}} = \omega_{\textrm{true}} + b_{\omega} + n_{\omega}, $$

其中，$a_{\textrm{meas}}$ 和 $\omega_{\textrm{meas}}$ 是测量的加速度和角速度，$a_{\textrm{true}}$ 和 $\omega_{\textrm{true}}$ 是真实的加速度和角速度，$b_a$ 和 $b_{\omega}$ 是偏差，$n_a$ 和 $n_{\omega}$ 是噪声项。噪声项被建模为具有标准差 $\sigma_{n_a}$ 和 $\sigma_{n_{\omega}}$ 的高斯噪声。偏差被建模为随机游走过程，参数为 $\sigma_{a}$ 和 $\sigma_{\omega}$。

??? 注释 "IMU 的偏差模型"
    是随机游走模型，提供的传感器的参数来自 VN-100 IMU。使用 IMU 需要在机器人资产配置中启用力传感器。

$$ b_{a,k} = b_{a,k-1} + \sigma_{a} \cdot \mathcal{N}(0,1) / \sqrt{\Delta t}, $$

$$ b_{\omega,k} = b_{\omega,k-1} + \sigma_{\omega} \cdot \mathcal{N}(0,1) \cdot \sqrt{\Delta t}, $$

其中 $b_{a,k}$ 和 $b_{\omega,k}$ 是时间 $k$ 的偏差，$\sigma_{a}$ 和 $\sigma_{\omega}$ 是偏差随机游走参数，$\Delta t$ 是时间步长，$a_{\textrm{meas}}$ 和 $\omega_{\textrm{meas}}$ 是测量的加速度和角速度，$a_{\textrm{true}}$ 和 $\omega_{\textrm{true}}$ 是真实的加速度和角速度，$b_a$ 和 $b_{\omega}$ 是偏差。噪声项 $n_a$ 和 $n_{\omega}$ 被建模为：

$$ n_a = \sigma_{n_a} \cdot \mathcal{N}(0,1) / \sqrt{\Delta t}, \textrm{和} $$

$$ n_{\omega} = \sigma_{n_{\omega}} \cdot \mathcal{N}(0,1) / \sqrt{\Delta t}. $$