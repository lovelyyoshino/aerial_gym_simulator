import torch
import math

# import nvtx
import warp as wp

from aerial_gym.sensors.warp.warp_kernels.warp_lidar_kernels import LidarWarpKernels


class WarpLidar:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        # 初始化WarpLidar类
        # num_envs: 环境数量
        # config: 配置对象，包含传感器参数
        # mesh_ids_array: 网格ID数组
        # device: 计算设备，默认为"cuda:0"
        
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = mesh_ids_array
        self.num_scan_lines = self.cfg.height  # 激光雷达扫描行数
        self.num_points_per_line = self.cfg.width  # 每行的点数
        self.horizontal_fov_min = math.radians(self.cfg.horizontal_fov_deg_min)  # 水平视场角最小值（弧度）
        self.horizontal_fov_max = math.radians(self.cfg.horizontal_fov_deg_max)  # 水平视场角最大值（弧度）
        self.horizontal_fov = self.horizontal_fov_max - self.horizontal_fov_min  # 水平视场角范围
        self.horizontal_fov_mean = (self.horizontal_fov_max + self.horizontal_fov_min) / 2  # 水平视场角均值
        if self.horizontal_fov > 2 * math.pi:
            raise ValueError("Horizontal FOV must be less than 2pi")  # 检查视场角范围

        self.vertical_fov_min = math.radians(self.cfg.vertical_fov_deg_min)  # 垂直视场角最小值（弧度）
        self.vertical_fov_max = math.radians(self.cfg.vertical_fov_deg_max)  # 垂直视场角最大值（弧度）
        self.vertical_fov = self.vertical_fov_max - self.vertical_fov_min  # 垂直视场角范围
        self.vertical_fov_mean = (self.vertical_fov_max + self.vertical_fov_min) / 2  # 垂直视场角均值
        if self.vertical_fov > math.pi:
            raise ValueError("Vertical FOV must be less than pi")  # 检查视场角范围
        self.far_plane = self.cfg.max_range  # 最大探测范围
        self.device = device  # 计算设备

        self.lidar_position_array = None  # 激光雷达位置数组
        self.lidar_quat_array = None  # 激光雷达四元数数组
        self.graph = None  # 渲染图

        self.initialize_ray_vectors()  # 初始化射线向量

    def initialize_ray_vectors(self):
        # 初始化射线向量
        # 创建一个二维torch数组，包含射线向量（wp.vec3格式）
        
        ray_vectors = torch.zeros(
            (self.num_scan_lines, self.num_points_per_line, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # 从 +HFoV/2 到 -HFoV/2 和 +VFoV/2 到 -VFoV/2 计算射线方向
                azimuth_angle = self.horizontal_fov_max - (
                    self.horizontal_fov_max - self.horizontal_fov_min
                ) * (j / (self.num_points_per_line - 1))  # 计算方位角
                elevation_angle = self.vertical_fov_max - (
                    self.vertical_fov_max - self.vertical_fov_min
                ) * (i / (self.num_scan_lines - 1))  # 计算仰角
                ray_vectors[i, j, 0] = math.cos(azimuth_angle) * math.cos(elevation_angle)  # x分量
                ray_vectors[i, j, 1] = math.sin(azimuth_angle) * math.cos(elevation_angle)  # y分量
                ray_vectors[i, j, 2] = math.sin(elevation_angle)  # z分量
        # 归一化射线向量
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # 转换为2D warp数组格式的vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph_pointcloud(self, debug=False):
        # 创建点云渲染图
        # debug: 是否在调试模式下运行
        
        if not debug:
            print(f"creating render graph")  # 输出创建渲染图信息
            wp.capture_begin(device=self.device)  # 开始捕获图形
        if self.cfg.segmentation_camera == True:
            # 使用分割相机的情况
            wp.launch(
                kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud_segmentation,  # 使用分割点云核
                dim=(
                    self.num_envs,
                    self.num_sensors,
                    self.num_scan_lines,
                    self.num_points_per_line,
                ),
                inputs=[
                    self.mesh_ids_array,
                    self.lidar_position_array,
                    self.lidar_quat_array,
                    self.ray_vectors,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )

        else:
            # 不使用分割相机的情况
            wp.launch(
                kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,  # 使用普通点云核
                dim=(
                    self.num_envs,
                    self.num_sensors,
                    self.num_scan_lines,
                    self.num_points_per_line,
                ),
                inputs=[
                    self.mesh_ids_array,
                    self.lidar_position_array,
                    self.lidar_quat_array,
                    self.ray_vectors,
                    self.far_plane,
                    self.pixels,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")  # 输出结束捕获信息
            self.graph = wp.capture_end(device=self.device)  # 结束捕获并保存图形

    def create_render_graph_range(self, debug=False):
        # 创建距离渲染图
        # debug: 是否在调试模式下运行
        
        if not debug:
            print(f"creating render graph")  # 输出创建渲染图信息
            wp.capture_begin(device=self.device)  # 开始捕获图形
        if self.cfg.segmentation_camera == True:
            # 使用分割相机的情况
            wp.launch(
                kernel=LidarWarpKernels.draw_optimized_kernel_range_segmentation,  # 使用分割范围核
                dim=(
                    self.num_envs,
                    self.num_sensors,
                    self.num_scan_lines,
                    self.num_points_per_line,
                ),
                inputs=[
                    self.mesh_ids_array,
                    self.lidar_position_array,
                    self.lidar_quat_array,
                    self.ray_vectors,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                ],
                device=self.device,
            )
        else:
            # 不使用分割相机的情况
            wp.launch(
                kernel=LidarWarpKernels.draw_optimized_kernel_range,  # 使用普通范围核
                dim=(
                    self.num_envs,
                    self.num_sensors,
                    self.num_scan_lines,
                    self.num_points_per_line,
                ),
                inputs=[
                    self.mesh_ids_array,
                    self.lidar_position_array,
                    self.lidar_quat_array,
                    self.ray_vectors,
                    self.far_plane,
                    self.pixels,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")  # 输出结束捕获信息
            self.graph = wp.capture_end(device=self.device)  # 结束捕获并保存图形

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # 设置图像张量
        # pixels: 像素数据
        # segmentation_pixels: 可选的分割像素数据
        
        # 初始化缓冲区。未初始化时为None
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)  # 将像素数据转换为vec3格式
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame  # 设置点云在世界坐标系中的位置
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)  # 将像素数据转换为float32格式

        if self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)  # 分割像素数据转换为int32格式
        else:
            self.segmentation_pixels = segmentation_pixels  # 如果没有分割相机，则保持为None

    def set_pose_tensor(self, positions, orientations):
        # 设置位姿张量
        # positions: 激光雷达位置
        # orientations: 激光雷达方向（四元数）
        
        self.lidar_position_array = wp.from_torch(positions, dtype=wp.vec3)  # 将位置数据转换为vec3格式
        self.lidar_quat_array = wp.from_torch(orientations, dtype=wp.quat)  # 将方向数据转换为四元数格式

    # @nvtx.annotate()
    def capture(self, debug=False):
        # 捕获数据
        # debug: 是否在调试模式下运行
        
        if self.graph is None:  # 如果图形尚未创建
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug)  # 创建点云渲染图
            else:
                self.create_render_graph_range(debug)  # 创建距离渲染图

        if self.graph is not None:  # 如果图形已创建
            wp.capture_launch(self.graph)  # 启动图形捕获

        return wp.to_torch(self.pixels)  # 返回像素数据（torch格式）
