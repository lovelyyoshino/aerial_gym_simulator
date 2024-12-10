import torch
import math

# import nvtx
import warp as wp

from aerial_gym.sensors.warp.warp_kernels.warp_lidar_kernels import LidarWarpKernels

class WarpNormalFaceIDLidar:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        # 初始化Lidar的参数和配置
        self.cfg = config  # 配置参数
        self.num_envs = num_envs  # 环境数量
        self.num_sensors = self.cfg.num_sensors  # 传感器数量
        self.mesh_ids_array = mesh_ids_array  # 网格ID数组
        self.num_scan_lines = self.cfg.height  # 扫描行数
        self.num_points_per_line = self.cfg.width  # 每行点数
        self.horizontal_fov_min = math.radians(self.cfg.horizontal_fov_deg_min)  # 水平视场最小角度（弧度）
        self.horizontal_fov_max = math.radians(self.cfg.horizontal_fov_deg_max)  # 水平视场最大角度（弧度）
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min  # 水平视场范围
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2  # 水平视场均值
        if self.horizontal_fov > 2 * math.pi:  # 检查视场范围是否合理
            raise ValueError("Horizontal FOV must be less than 2pi")

        self.vertical_fov_min = math.radians(self.cfg.vertical_fov_deg_min)  # 垂直视场最小角度（弧度）
        self.vertical_fov_max = math.radians(self.cfg.vertical_fov_deg_max)  # 垂直视场最大角度（弧度）
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min  # 垂直视场范围
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2  # 垂直视场均值
        if self.vertical_fov > math.pi:  # 检查视场范围是否合理
            raise ValueError("Vertical FOV must be less than pi")
        self.far_plane = self.cfg.max_range  # 最大探测距离
        self.device = device  # 设备配置（默认使用cuda:0）

        self.lidar_position_array = None  # 激光雷达位置数组
        self.lidar_quat_array = None  # 激光雷达四元数数组
        self.graph = None  # 渲染图

        self.pixels = None  # 像素数据
        self.face_pixels = None  # 面部像素数据

        self.normal_in_world_frame = self.cfg.normal_in_world_frame  # 世界坐标系中的法向量

        self.initialize_ray_vectors()  # 初始化光线向量

    def initialize_ray_vectors(self):
        # 初始化光线向量，生成一个2D的torch数组，存储光线向量（wp.vec3类型）
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # 光线从 +HFoV/2 到 -HFoV/2 和 +VFoV/2 到 -VFoV/2 进行划分
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))  # 计算方位角
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))  # 计算仰角
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)  # 计算光线向量x分量
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)  # 计算光线向量y分量
                ray_vectors[i, j, 2] = math.sin(elevation_angle)  # 计算光线向量z分量
        # 归一化光线向量
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # 转换为2D的warp数组（vec3类型）
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph_pointcloud(self, debug=False):
        # 创建渲染图的点云
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)  # 开始捕捉渲染图
        wp.launch(
            kernel=LidarWarpKernels.draw_optimized_kernel_normal_faceID,  # 调用绘制优化内核
            dim=(
                self.num_envs,
                self.num_sensors,
                self.num_scan_lines,
                self.num_points_per_line,
            ),  # 指定维度
            inputs=[
                self.mesh_ids_array,
                self.lidar_position_array,
                self.lidar_quat_array,
                self.ray_vectors,
                self.far_plane,
                self.pixels,
                self.face_pixels,
                self.normal_in_world_frame,
            ],
            device=self.device,
        )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)  # 结束捕捉渲染图并保存图形

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # 设置图像张量，初始化像素缓冲区
        self.pixels = wp.from_torch(pixels, dtype=wp.vec3)  # 将像素数据转换为warp格式
        self.face_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)  # 将分割像素数据转换为warp格式（可选）

    def set_pose_tensor(self, positions, orientations):
        # 设置姿态张量，初始化激光雷达的位置和方向
        self.lidar_position_array = wp.from_torch(positions, dtype=wp.vec3)  # 将位置数据转换为warp格式
        self.lidar_quat_array = wp.from_torch(orientations, dtype=wp.quat)  # 将方向数据转换为warp格式

    # @nvtx.annotate()
    def capture(self, debug=False):
        # 捕获渲染图像
        if self.graph is None:
            self.create_render_graph_pointcloud(debug)  # 如果图形未创建，则创建渲染图
        if self.graph is not None:
            wp.capture_launch(self.graph)  # 启动捕捉图形

        return wp.to_torch(self.pixels)  # 返回像素数据
