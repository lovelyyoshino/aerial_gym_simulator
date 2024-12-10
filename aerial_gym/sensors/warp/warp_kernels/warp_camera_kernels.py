import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)  # 没有击中物体的射线值
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))  # 没有击中物体的分割值

class DepthCameraWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # 相机位置数组
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # 相机四元数数组
        K_inv: wp.mat44,  # 相机内参的逆矩阵
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出的点云像素
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出的分割像素
        c_x: int,  # 主轴x坐标
        c_y: int,  # 主轴y坐标
        pointcloud_in_world_frame: bool,  # 点云是否在世界坐标系中
    ):
        # 获取当前线程的环境ID、相机ID、x和y坐标
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格ID
        cam_pos = cam_poss[env_id, cam_id]  # 获取当前相机的位置
        cam_quat = cam_quats[env_id, cam_id]  # 获取当前相机的四元数
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # 将坐标转换到Warp的坐标系
        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # 获取主轴的坐标
        # 将坐标转换到[-1, 1]范围
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(wp.transform_vector(K_inv, cam_coords_principal))  # 主轴的uv坐标
        # 计算相机射线
        ro = cam_pos  # 相机在世界坐标中的原点
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))  # 从相机到世界空间的方向并归一化
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # 主轴的射线方向
        t = float(0.0)  # 射线与物体的交点距离
        u = float(0.0)  # 交点u坐标
        v = float(0.0)  # 交点v坐标
        sign = float(0.0)  # 射线法向量的符号
        n = wp.vec3()  # 射线法向量
        f = int(0)  # 面索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为没有击中的值
        segmentation_value = NO_HIT_SEGMENTATION_VAL  # 初始化分割值为没有击中的值

        # 查询射线与网格的交点
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t  # 更新距离
            mesh_obj = wp.mesh_get(mesh)  # 获取网格对象
            face_index = mesh_obj.indices[f * 3]  # 获取面索引
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])  # 获取分割值
        # 根据是否在世界坐标系中选择不同的输出方式
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd  # 世界坐标系中的点云
        else:
            pixels[env_id, cam_id, y, x] = dist * uv  # 相机坐标系中的点云
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value  # 设置分割像素值

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_normal_faceID(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # 相机位置数组
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # 相机四元数数组
        K_inv: wp.mat44,  # 相机内参的逆矩阵
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出的法向量像素
        face_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出的面ID像素
        c_x: int,  # 主轴x坐标
        c_y: int,  # 主轴y坐标
        normal_in_world_frame: bool,  # 法向量是否在世界坐标系中
    ):
        # 获取当前线程的环境ID、相机ID、x和y坐标
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格ID
        cam_pos = cam_poss[env_id, cam_id]  # 获取当前相机的位置
        cam_quat = cam_quats[env_id, cam_id]  # 获取当前相机的四元数
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # 将坐标转换到Warp的坐标系
        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # 获取主轴的坐标
        # 将坐标转换到[-1, 1]范围
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(wp.transform_vector(K_inv, cam_coords_principal))  # 主轴的uv坐标
        # 计算相机射线
        ro = cam_pos  # 相机在世界坐标中的原点
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))  # 从相机到世界空间的方向并归一化
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # 主轴的射线方向
        t = float(0.0)  # 射线与物体的交点距离
        u = float(0.0)  # 交点u坐标
        v = float(0.0)  # 交点v坐标
        sign = float(0.0)  # 射线法向量的符号
        n = wp.vec3()  # 射线法向量
        f = int(-1)  # 初始化面索引为-1
        pixels[env_id, cam_id, y, x] = n * NO_HIT_RAY_VAL  # 初始化像素值为没有击中的法向量
        # 查询射线与网格的交点
        wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f)
        # 根据法向量是否在世界坐标系中选择不同的输出方式
        if normal_in_world_frame:
            pixels[env_id, cam_id, y, x] = n  # 世界坐标系中的法向量
        else:
            # 将法向量转换到相机坐标系中
            pixels[env_id, cam_id, y, x] = wp.vec3(
                wp.dot(n, rd_principal),  # 法向量在主轴方向的投影
                wp.dot(n, wp.cross(rd_principal, wp.vec3(0.0, 0.0, 1.0))),  # 法向量在相机Z轴的投影
                wp.dot(n, wp.cross(rd_principal, wp.vec3(0.0, 1.0, 0.0)))  # 法向量在相机Y轴的投影
            )
        face_pixels[env_id, cam_id, y, x] = f  # 设置面ID像素值

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # 相机位置数组
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # 相机四元数数组
        K_inv: wp.mat44,  # 相机内参的逆矩阵
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # 输出的点云像素
        c_x: int,  # 主轴x坐标
        c_y: int,  # 主轴y坐标
        pointcloud_in_world_frame: bool,  # 点云是否在世界坐标系中
    ):
        # 获取当前线程的环境ID、相机ID、x和y坐标
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格ID
        cam_pos = cam_poss[env_id, cam_id]  # 获取当前相机的位置
        cam_quat = cam_quats[env_id, cam_id]  # 获取当前相机的四元数
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # 将坐标转换到Warp的坐标系
        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # 获取主轴的坐标
        # 将坐标转换到[-1, 1]范围
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(wp.transform_vector(K_inv, cam_coords_principal))  # 主轴的uv坐标
        # 计算相机射线
        ro = cam_pos  # 相机在世界坐标中的原点
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))  # 从相机到世界空间的方向并归一化
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # 主轴的射线方向
        t = float(0.0)  # 射线与物体的交点距离
        u = float(0.0)  # 交点u坐标
        v = float(0.0)  # 交点v坐标
        sign = float(0.0)  # 射线法向量的符号
        n = wp.vec3()  # 射线法向量
        f = int(0)  # 面索引
        dist = NO_HIT_RAY_VAL  # 初始化距离为没有击中的值
        # 查询射线与网格的交点
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t  # 更新距离
        # 根据是否在世界坐标系中选择不同的输出方式
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd  # 世界坐标系中的点云
        else:
            pixels[env_id, cam_id, y, x] = dist * uv  # 相机坐标系中的点云

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # 相机位置数组
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # 相机四元数数组
        K_inv: wp.mat44,  # 相机内参的逆矩阵
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=float, ndim=4),  # 输出的深度范围像素
        c_x: int,  # 主轴x坐标
        c_y: int,  # 主轴y坐标
        calculate_depth: bool,  # 是否计算深度
    ):
        # 获取当前线程的环境ID、相机ID、x和y坐标
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格ID
        cam_pos = cam_poss[env_id, cam_id]  # 获取当前相机的位置
        cam_quat = cam_quats[env_id, cam_id]  # 获取当前相机的四元数
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # 将坐标转换到Warp的坐标系
        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # 获取主轴的坐标
        # 将坐标转换到[-1, 1]范围
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # 主轴的uv坐标
        # 计算相机射线
        ro = cam_pos  # 相机在世界坐标中的原点
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))  # 从相机到世界空间的方向并归一化
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # 主轴的射线方向
        t = float(0.0)  # 射线与物体的交点距离
        u = float(0.0)  # 交点u坐标
        v = float(0.0)  # 交点v坐标
        sign = float(0.0)  # 射线法向量的符号
        n = wp.vec3()  # 射线法向量
        f = int(0)  # 面索引
        multiplier = 1.0  # 乘数初始化为1.0
        if calculate_depth:
            multiplier = wp.dot(rd, rd_principal)  # 计算深度时的乘数
        dist = NO_HIT_RAY_VAL  # 初始化距离为没有击中的值
        # 查询射线与网格的交点
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t  # 更新距离

        pixels[env_id, cam_id, y, x] = dist  # 设置深度范围像素值

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_segmentation(
        mesh_ids: wp.array(dtype=wp.uint64),  # 网格ID数组
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # 相机位置数组
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # 相机四元数数组
        K_inv: wp.mat44,  # 相机内参的逆矩阵
        far_plane: float,  # 远平面距离
        pixels: wp.array(dtype=float, ndim=4),  # 输出的深度范围像素
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=4),  # 输出的分割像素
        c_x: int,  # 主轴x坐标
        c_y: int,  # 主轴y坐标
        calculate_depth: bool,  # 是否计算深度
    ):
        # 获取当前线程的环境ID、相机ID、x和y坐标
        env_id, cam_id, x, y = wp.tid()
        mesh = mesh_ids[env_id]  # 获取当前环境的网格ID
        cam_pos = cam_poss[env_id, cam_id]  # 获取当前相机的位置
        cam_quat = cam_quats[env_id, cam_id]  # 获取当前相机的四元数
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # 将坐标转换到Warp的坐标系
        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)  # 获取主轴的坐标
        # 将坐标转换到[-1, 1]范围
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # 主轴的uv坐标
        # 计算相机射线
        ro = cam_pos  # 相机在世界坐标中的原点
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))  # 从相机到世界空间的方向并归一化
        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))  # 主轴的射线方向
        t = float(0.0)  # 射线与物体的交点距离
        u = float(0.0)  # 交点u坐标
        v = float(0.0)  # 交点v坐标
        sign = float(0.0)  # 射线法向量的符号
        n = wp.vec3()  # 射线法向量
        f = int(0)  # 面索引
        multiplier = 1.0  # 乘数初始化为1.0
        if calculate_depth:
            multiplier = wp.dot(rd, rd_principal)  # 计算深度时的乘数
        dist = NO_HIT_RAY_VAL  # 初始化距离为没有击中的值
        segmentation_value = NO_HIT_SEGMENTATION_VAL  # 初始化分割值为没有击中的值
        # 查询射线与网格的交点
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t  # 更新距离
            mesh_obj = wp.mesh_get(mesh)  # 获取网格对象
            face_index = mesh_obj.indices[f * 3]  # 获取面索引
            segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])  # 获取分割值

        pixels[env_id, cam_id, y, x] = dist  # 设置深度范围像素值
        segmentation_pixels[env_id, cam_id, y, x] = segmentation_value  # 设置分割像素值
