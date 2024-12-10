# 导入所需库
import nvtx
import warp as wp
import math

from aerial_gym.sensors.warp.warp_kernels.warp_camera_kernels import (
    DepthCameraWarpKernels,
)

# 定义WarpNormalFaceIDCam类
class WarpNormalFaceIDCam:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        # 初始化参数
        self.cfg = config  # 配置参数
        self.num_envs = num_envs  # 环境数量
        self.num_sensors = self.cfg.num_sensors  # 传感器数量
        self.mesh_ids_array = mesh_ids_array  # 网格ID数组

        self.width = self.cfg.width  # 图像宽度
        self.height = self.cfg.height  # 图像高度

        # 将水平视场角度转换为弧度
        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)
        self.far_plane = self.cfg.max_range  # 最大范围
        self.device = device  # 设备类型

        # 初始化相机相关变量
        self.camera_position_array = None  # 相机位置数组
        self.camera_orientation_array = None  # 相机方向数组
        self.graph = None  # 渲染图

        self.pixels = None  # 像素数据
        self.face_pixels = None  # 面部像素数据
        self.K_inv = None  # 相机内参的逆矩阵
        self.c_x = 0.0  # 中心x坐标
        self.c_y = 0.0  # 中心y坐标
        self.normal_in_world_frame = self.cfg.normal_in_world_frame  # 世界坐标系中的法向量

        # 初始化相机矩阵
        self.initialize_camera_matrices()

    def initialize_camera_matrices(self):
        # 计算相机参数
        W = self.width  # 图像宽度
        H = self.height  # 图像高度
        (u_0, v_0) = (W / 2, H / 2)  # 图像中心点坐标
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)  # 焦距计算

        vertical_fov = 2 * math.atan(H / (2 * f))  # 垂直视场角
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)  # 水平缩放因子
        alpha_v = v_0 / math.tan(vertical_fov / 2)  # 垂直缩放因子

        # 简单针孔模型
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
        self.K_inv = wp.inverse(self.K)  # 计算相机内参的逆矩阵

        self.c_x = int(u_0)  # 记录中心点x坐标
        self.c_y = int(v_0)  # 记录中心点y坐标

    def create_render_graph(self, debug=False):
        # 创建渲染图
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)  # 开始捕获渲染图
        # with wp.ScopedTimer("render"):
        wp.launch(
            kernel=DepthCameraWarpKernels.draw_optimized_kernel_normal_faceID,  # 启动深度相机内核
            dim=(self.num_envs, self.num_sensors, self.width, self.height),  # 设置维度
            inputs=[
                self.mesh_ids_array,  # 网格ID数组
                self.camera_position_array,  # 相机位置数组
                self.camera_orientation_array,  # 相机方向数组
                self.K_inv,  # 相机内参逆矩阵
                self.far_plane,  # 最大范围
                self.pixels,  # 像素数据
                self.face_pixels,  # 面部像素数据
                self.c_x,  # 中心x坐标
                self.c_y,  # 中心y坐标
                self.normal_in_world_frame,  # 世界坐标系中的法向量
            ],
            device=self.device,  # 设备类型
        )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)  # 结束捕获渲染图并保存图形

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # 初始化图像缓冲区
        self.pixels = wp.from_torch(pixels, dtype=wp.vec3)  # 将像素数据转换为warp格式
        self.face_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)  # 将分割像素数据转换为warp格式

    def set_pose_tensor(self, positions, orientations):
        # 设置相机的位置和方向
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)  # 设置相机位置
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)  # 设置相机方向

    # @nvtx.annotate()
    def capture(self, debug=False):
        # 捕获图像
        if self.graph is None:  # 如果图形未创建，则创建图形
            self.create_render_graph(debug=debug)
        if self.graph is not None:  # 如果图形已创建，则启动图形
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)  # 将像素数据转换回torch格式并返回
