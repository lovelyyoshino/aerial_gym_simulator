# import nvtx
import warp as wp
import math

from aerial_gym.sensors.warp.warp_kernels.warp_camera_kernels import (
    DepthCameraWarpKernels,
)


class WarpCam:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        # 初始化WarpCam类
        self.cfg = config  # 配置参数
        self.num_envs = num_envs  # 环境数量
        self.num_sensors = self.cfg.num_sensors  # 传感器数量
        self.mesh_ids_array = mesh_ids_array  # 网格ID数组

        self.width = self.cfg.width  # 图像宽度
        self.height = self.cfg.height  # 图像高度

        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)  # 水平视场角（弧度）
        self.far_plane = self.cfg.max_range  # 远平面距离
        self.calculate_depth = self.cfg.calculate_depth  # 是否计算深度
        self.device = device  # 设备（默认为cuda:0）

        self.camera_position_array = None  # 相机位置数组
        self.camera_orientation_array = None  # 相机朝向数组
        self.graph = None  # 渲染图

        self.initialize_camera_matrices()  # 初始化相机矩阵

    def initialize_camera_matrices(self):
        # 计算相机参数
        W = self.width  # 图像宽度
        H = self.height  # 图像高度
        (u_0, v_0) = (W / 2, H / 2)  # 主点坐标
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)  # 焦距

        vertical_fov = 2 * math.atan(H / (2 * f))  # 垂直视场角
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)  # 水平缩放因子
        alpha_v = v_0 / math.tan(vertical_fov / 2)  # 垂直缩放因子

        # 简单的针孔模型
        self.K = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.K_inv = wp.inverse(self.K)  # 相机内参矩阵的逆

        self.c_x = int(u_0)  # 主点的x坐标
        self.c_y = int(v_0)  # 主点的y坐标

    def create_render_graph_pointcloud(self, debug=False):
        # 创建点云渲染图
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)  # 开始捕获渲染图
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:  # 如果启用分割相机
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.pointcloud_in_world_frame,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)  # 结束捕获渲染图

    def create_render_graph_depth_range(self, debug=False):
        # 创建深度范围渲染图
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)  # 开始捕获渲染图
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:  # 如果启用分割相机
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range_segmentation,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.segmentation_pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.num_sensors, self.width, self.height),
                inputs=[
                    self.mesh_ids_array,
                    self.camera_position_array,
                    self.camera_orientation_array,
                    self.K_inv,
                    self.far_plane,
                    self.pixels,
                    self.c_x,
                    self.c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)  # 结束捕获渲染图

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # 设置图像张量
        if self.cfg.return_pointcloud:  # 如果返回点云
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)  # 将像素转换为vec3类型
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame  # 点云在世界坐标系中的位置
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)  # 将像素转换为float32类型

        if self.cfg.segmentation_camera == True:  # 如果启用分割相机
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)  # 将分割像素转换为int32类型
        else:
            self.segmentation_pixels = segmentation_pixels  # 不处理分割像素

    def set_pose_tensor(self, positions, orientations):
        # 设置相机的位姿张量
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)  # 将位置转换为vec3类型
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)  # 将朝向转换为四元数

    # @nvtx.annotate()
    def capture(self, debug=False):
        # 捕获图像
        if self.graph is None:
            if self.cfg.return_pointcloud:  # 如果返回点云
                self.create_render_graph_pointcloud(debug=debug)  # 创建点云渲染图
            else:
                self.create_render_graph_depth_range(debug=debug)  # 创建深度范围渲染图

        if self.graph is not None:
            wp.capture_launch(self.graph)  # 启动渲染图

        return wp.to_torch(self.pixels)  # 返回转换为Torch张量的像素
