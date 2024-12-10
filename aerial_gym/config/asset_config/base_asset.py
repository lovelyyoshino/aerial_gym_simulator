from aerial_gym import AERIAL_GYM_DIRECTORY

class BaseAssetParams:
    num_assets = 1  # 包含的资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"  # 资产文件夹路径
    file = None  # 如果file=None，将随机选择资产。如果不为None，则使用该文件

    min_position_ratio = [0.5, 0.5, 0.5]  # 最小位置比例，相对于边界
    max_position_ratio = [0.5, 0.5, 0.5]  # 最大位置比例，相对于边界

    collision_mask = 1  # 碰撞掩码，用于定义碰撞检测的对象

    disable_gravity = False  # 是否禁用重力
    replace_cylinder_with_capsule = (
        True  # 将碰撞圆柱体替换为胶囊体，以实现更快/更稳定的仿真
    )
    flip_visual_attachments = True  # 一些 .obj 网格必须从 y-up 翻转到 z-up
    density = 0.000001  # 物体密度
    angular_damping = 0.0001  # 角阻尼系数
    linear_damping = 0.0001  # 线性阻尼系数
    max_angular_velocity = 100.0  # 最大角速度
    max_linear_velocity = 100.0  # 最大线速度
    armature = 0.001  # 骨架参数

    collapse_fixed_joints = True  # 是否合并固定关节
    fix_base_link = True  # 是否固定基础链接
    color = None  # 颜色，默认为None
    keep_in_env = False  # 是否保持在环境中

    body_semantic_label = 0  # 身体语义标签
    link_semantic_label = 0  # 链接语义标签
    per_link_semantic = False  # 是否每个链接都有独立的语义
    semantic_masked_links = {}  # 语义遮罩链接字典
    place_force_sensor = False  # 是否放置力传感器
    force_sensor_parent_link = "base_link"  # 力传感器的父链接
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # 力传感器的位置和四元数（x, y, z, w）
    use_collision_mesh_instead_of_visual = False  # 是否使用碰撞网格而不是视觉网格
