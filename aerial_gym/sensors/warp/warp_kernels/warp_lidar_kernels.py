import warp as wp

# 定义常量，表示未命中的光线和分割值
NO_HIT_RAY_VAL = wp.constant(1000.0)  # 未命中的光线距离值
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))  # 未命中的分割值

class LidarWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),  # 激光雷达位置数组
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),  # 激光雷达方向四元数数组
        ray_vectors: wp.array2d(dtype=wp.vec3),  # 光线方向数组
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出点云像素数组
        pointcloud_in_world_frame: bool,  # 是否以世界坐标系输出点云
    ):
        # 获取当前线程的ID
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格
        lidar_position = lidar_pos_array[env_id, cam_id]  # 获取激光雷达位置
        lidar_quaternion = lidar_quat_array[env_id, cam_id]  # 获取激光雷达方向
        ray_origin = lidar_position  # 光线起点为激光雷达位置

        # 计算光线方向并进行归一化
        ray_dir = ray_vectors[scan_line, point_index]  # 获取当前光线方向
        ray_dir = wp.normalize(ray_dir)  # 归一化光线方向
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))  # 将光线方向转换到世界坐标系
        
        # 初始化变量
        t = float(0.0)  # 光线与物体的交点参数
        u = float(0.0)  # UV坐标
        v = float(0.0)  # UV坐标
        sign = float(0.0)  # 符号
        n = wp.vec3()  # 法向量
        f = int(0)  # 面的索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为未命中值
        
        # 查询光线与网格的交点
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t  # 如果有交点，更新距离
        
        # 根据是否以世界坐标系输出点云更新像素值
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        else:
            pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),  # 激光雷达位置数组
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),  # 激光雷达方向四元数数组
        ray_vectors: wp.array2d(dtype=wp.vec3),  # 光线方向数组
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出点云像素数组
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出分割像素数组
        pointcloud_in_world_frame: bool,  # 是否以世界坐标系输出点云
    ):
        # 获取当前线程的ID
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格
        lidar_position = lidar_pos_array[env_id, cam_id]  # 获取激光雷达位置
        lidar_quaternion = lidar_quat_array[env_id, cam_id]  # 获取激光雷达方向
        ray_origin = lidar_position  # 光线起点为激光雷达位置

        # 计算光线方向并进行归一化
        ray_dir = ray_vectors[scan_line, point_index]  # 获取当前光线方向
        ray_dir = wp.normalize(ray_dir)  # 归一化光线方向
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))  # 将光线方向转换到世界坐标系
        
        # 初始化变量
        t = float(0.0)  # 光线与物体的交点参数
        u = float(0.0)  # UV坐标
        v = float(0.0)  # UV坐标
        sign = float(0.0)  # 符号
        n = wp.vec3()  # 法向量
        f = int(0)  # 面的索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为未命中值
        segmentation_value = NO_HIT_SEGMENTATION_VAL  # 初始化分割值为未命中值

        # 查询光线与网格的交点
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t  # 如果有交点，更新距离
            mesh_obj = wp.mesh_get(mesh)  # 获取网格对象
            face_index = mesh_obj.indices[f * 3]  # 获取面的索引
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])  # 获取分割值
        
        # 根据是否以世界坐标系输出点云更新像素值
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
        else:
            pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir
        
        segmentation_pixels[env_id, cam_id, scan_line, point_index] = segmentation_value  # 更新分割像素

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_normal_faceID(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),  # 激光雷达位置数组
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),  # 激光雷达方向四元数数组
        ray_vectors: wp.array2d(dtype=wp.vec3),  # 光线方向数组
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出法线像素数组
        face_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出面ID像素数组
        pointcloud_in_world_frame: bool,  # 是否以世界坐标系输出点云
    ):
        # 获取当前线程的ID
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格
        lidar_position = lidar_pos_array[env_id, cam_id]  # 获取激光雷达位置
        lidar_quaternion = lidar_quat_array[env_id, cam_id]  # 获取激光雷达方向
        ray_origin = lidar_position  # 光线起点为激光雷达位置

        # 计算光线方向并进行归一化
        ray_dir = ray_vectors[scan_line, point_index]  # 获取当前光线方向
        ray_dir = wp.normalize(ray_dir)  # 归一化光线方向
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))  # 将光线方向转换到世界坐标系
        
        # 初始化变量
        t = float(0.0)  # 光线与物体的交点参数
        u = float(0.0)  # UV坐标
        v = float(0.0)  # UV坐标
        sign = float(0.0)  # 符号
        n = wp.vec3()  # 法向量
        f = int(-1)  # 面的索引，初始化为-1
        pixels[env_id, cam_id, scan_line, point_index] = n * NO_HIT_RAY_VAL  # 初始化像素为未命中值
        
        # 查询光线与网格的交点
        wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f)  # 不检查是否命中，直接获取法向量和面ID
        
        # 根据是否以世界坐标系输出点云更新像素值
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, scan_line, point_index] = n  # 输出法向量
        else:
            # 将法向量旋转到传感器坐标系
            pixels[env_id, cam_id, scan_line, point_index] = wp.normalize(
                wp.quat_rotate(wp.quat_inverse(lidar_quaternion), n)
            )
        face_pixels[env_id, cam_id, scan_line, point_index] = f  # 更新面ID像素

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),  # 激光雷达位置数组
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),  # 激光雷达方向四元数数组
        ray_vectors: wp.array2d(dtype=wp.vec3),  # 光线方向数组
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=float, ndim=4),  # 输出距离像素数组
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出分割像素数组
    ):
        # 获取当前线程的ID
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格
        lidar_position = lidar_pos_array[env_id, cam_id]  # 获取激光雷达位置
        lidar_quaternion = lidar_quat_array[env_id, cam_id]  # 获取激光雷达方向
        ray_origin = lidar_position  # 光线起点为激光雷达位置

        # 计算光线方向并进行归一化
        ray_dir = ray_vectors[scan_line, point_index]  # 获取当前光线方向
        ray_dir = wp.normalize(ray_dir)  # 归一化光线方向
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))  # 将光线方向转换到世界坐标系
        
        # 初始化变量
        t = float(0.0)  # 光线与物体的交点参数
        u = float(0.0)  # UV坐标
        v = float(0.0)  # UV坐标
        sign = float(0.0)  # 符号
        n = wp.vec3()  # 法向量
        f = int(0)  # 面的索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为未命中值
        segmentation_value = NO_HIT_SEGMENTATION_VAL  # 初始化分割值为未命中值
        
        # 查询光线与网格的交点
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t  # 如果有交点，更新距离
            mesh_obj = wp.mesh_get(mesh)  # 获取网格对象
            face_index = mesh_obj.indices[f * 3]  # 获取面的索引
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])  # 获取分割值
        
        pixels[env_id, cam_id, scan_line, point_index] = dist  # 更新距离像素
        segmentation_pixels[env_id, cam_id, scan_line, point_index] = segmentation_value  # 更新分割像素

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_range(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),  # 激光雷达位置数组
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),  # 激光雷达方向四元数数组
        ray_vectors: wp.array2d(dtype=wp.vec3),  # 光线方向数组
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=float, ndim=4),  # 输出距离像素数组
    ):
        # 获取当前线程的ID
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格
        lidar_position = lidar_pos_array[env_id, cam_id]  # 获取激光雷达位置
        lidar_quaternion = lidar_quat_array[env_id, cam_id]  # 获取激光雷达方向
        ray_origin = lidar_position  # 光线起点为激光雷达位置

        # 计算光线方向并进行归一化
        ray_dir = ray_vectors[scan_line, point_index]  # 获取当前光线方向
        ray_dir = wp.normalize(ray_dir)  # 归一化光线方向
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))  # 将光线方向转换到世界坐标系
        
        # 初始化变量
        t = float(0.0)  # 光线与物体的交点参数
        u = float(0.0)  # UV坐标
        v = float(0.0)  # UV坐标
        sign = float(0.0)  # 符号
        n = wp.vec3()  # 法向量
        f = int(0)  # 面的索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为未命中值
        
        # 查询光线与网格的交点
        if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
            dist = t  # 如果有交点，更新距离
        
        pixels[env_id, cam_id, scan_line, point_index] = dist  # 更新距离像素
