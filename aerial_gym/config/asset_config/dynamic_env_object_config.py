from aerial_gym import AERIAL_GYM_DIRECTORY

import numpy as np

# 定义语义ID，用于标识不同的环境资产类型
THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
PANEL_SEMANTIC_ID = 20
FRONT_WALL_SEMANTIC_ID = 9
BACK_WALL_SEMANTIC_ID = 10
LEFT_WALL_SEMANTIC_ID = 11
RIGHT_WALL_SEMANTIC_ID = 12
BOTTOM_WALL_SEMANTIC_ID = 13
TOP_WALL_SEMANTIC_ID = 14


class asset_state_params:
    # 环境资产参数基类，定义了一些通用属性
    num_assets = 1  # 包含的资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"  # 资产文件夹路径
    file = None  # 如果file=None，将随机选择资产。如果不为None，则使用该文件

    min_position_ratio = [0.5, 0.5, 0.5]  # 最小位置比例
    max_position_ratio = [0.5, 0.5, 0.5]  # 最大位置比例

    collision_mask = 1  # 碰撞掩码

    disable_gravity = True  # 是否禁用重力
    replace_cylinder_with_capsule = (
        True  # 将碰撞圆柱体替换为胶囊体，以提高模拟速度和稳定性
    )
    flip_visual_attachments = True  # 一些.obj网格需要从y-up翻转到z-up
    density = 0.001  # 密度
    angular_damping = 0.1  # 角阻尼
    linear_damping = 0.1  # 线性阻尼
    max_angular_velocity = 100.0  # 最大角速度
    max_linear_velocity = 100.0  # 最大线速度
    armature = 0.001  # 骨架质量

    collapse_fixed_joints = True  # 是否合并固定关节
    fix_base_link = False  # 是否固定基础链接
    specific_filepath = None  # 如果不为None，则使用此文件夹而不是随机化
    color = None  # 颜色
    keep_in_env = False  # 是否保持在环境中

    body_semantic_label = 0  # 身体语义标签
    link_semantic_label = 0  # 链接语义标签
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_masked_links = {}  # 语义屏蔽链接
    place_force_sensor = False  # 是否放置力传感器
    force_sensor_parent_link = "base_link"  # 力传感器父链接
    force_sensor_transform = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]  # 力传感器的位置和四元数（x, y, z, w）

    use_collision_mesh_instead_of_visual = False  # 是否使用碰撞网格代替视觉效果


class panel_asset_params(asset_state_params):
    # 面板资产参数类，继承自asset_state_params
    num_assets = 3  # 包含的面板资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"  # 面板资产文件夹路径

    collision_mask = 1  # 碰撞掩码

    min_position_ratio = [0.3, 0.05, 0.05]  # 最小位置比例
    max_position_ratio = [0.85, 0.95, 0.95]  # 最大位置比例

    specified_position = [
        -1000.0,
        -1000.0,
        -1000.0,
    ]  # 如果大于-900，则使用此值而不是随机化比例

    min_euler_angles = [0.0, 0.0, -np.pi / 3.0]  # 最小欧拉角
    max_euler_angles = [0.0, 0.0, np.pi / 3.0]  # 最大欧拉角

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        0.0,
        0.0,
        -np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        0.0,
        0.0,
        np.pi / 3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = -1  # 实例将逐步分配的语义ID
    color = [170, 66, 66]  # 颜色


class thin_asset_params(asset_state_params):
    # 薄资产参数类，继承自asset_state_params
    num_assets = 0  # 不包含薄资产

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"  # 薄资产文件夹路径

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.3,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.95,
        0.95,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True  # 是否合并固定关节
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = -1  # 实例将逐步分配的语义ID
    color = [170, 66, 66]  # 颜色


class tree_asset_params(asset_state_params):
    # 树资产参数类，继承自asset_state_params
    num_assets = 6  # 包含的树资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"  # 树资产文件夹路径

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.1,
        0.1,
        0.0,
        0,
        -np.pi / 6.0,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.9,
        0.9,
        0.0,
        0,
        np.pi / 6.0,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    collapse_fixed_joints = True  # 是否合并固定关节
    per_link_semantic = True  # 每个链接是否有独立的语义
    keep_in_env = True  # 是否保持在环境中

    semantic_id = -1  # TREE_SEMANTIC_ID
    color = [70, 200, 100]  # 颜色

    semantic_masked_links = {}  # 语义屏蔽链接


class object_asset_params(asset_state_params):
    # 对象资产参数类，继承自asset_state_params
    num_assets = 40  # 包含的对象资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"  # 对象资产文件夹路径

    min_state_ratio = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False  # 是否保持在环境中
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = -1  # 实例将逐步分配的语义ID


class left_wall(asset_state_params):
    # 左墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的左墙资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 左墙资产文件夹路径
    file = "left_wall.urdf"  # 左墙模型文件

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    specific_filepath = "cube.urdf"  # 特定文件路径
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = LEFT_WALL_SEMANTIC_ID  # 左墙的语义ID
    color = [100, 200, 210]  # 颜色


class right_wall(asset_state_params):
    # 右墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的右墙资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 右墙资产文件夹路径
    file = "right_wall.urdf"  # 右墙模型文件

    min_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    per_link_semantic = False  # 每个链接是否有独立的语义
    specific_filepath = "cube.urdf"  # 特定文件路径
    semantic_id = RIGHT_WALL_SEMANTIC_ID  # 右墙的语义ID
    color = [100, 200, 210]  # 颜色


class top_wall(asset_state_params):
    # 顶墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的顶墙资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 顶墙资产文件夹路径
    file = "top_wall.urdf"  # 顶墙模型文件

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    specific_filepath = "cube.urdf"  # 特定文件路径
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = TOP_WALL_SEMANTIC_ID  # 顶墙的语义ID
    color = [100, 200, 210]  # 颜色


class bottom_wall(asset_state_params):
    # 底墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的底墙资产数量
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 底墙资产文件夹路径
    file = "bottom_wall.urdf"  # 底墙模型文件

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    specific_filepath = "cube.urdf"  # 特定文件路径
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = BOTTOM_WALL_SEMANTIC_ID  # 底墙的语义ID
    color = [100, 150, 150]  # 颜色


class front_wall(asset_state_params):
    # 前墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的前墙资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 前墙资产文件夹路径
    file = "front_wall.urdf"  # 前墙模型文件

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    specific_filepath = "cube.urdf"  # 特定文件路径
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = FRONT_WALL_SEMANTIC_ID  # 前墙的语义ID
    color = [100, 200, 210]  # 颜色


class back_wall(asset_state_params):
    # 后墙资产参数类，继承自asset_state_params
    num_assets = 1  # 包含的后墙资产数量

    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 后墙资产文件夹路径
    file = "back_wall.urdf"  # 后墙模型文件

    collision_mask = 1  # 碰撞掩码

    min_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = True  # 是否保持在环境中

    collapse_fixed_joints = True  # 是否合并固定关节
    specific_filepath = "cube.urdf"  # 特定文件路径
    per_link_semantic = False  # 每个链接是否有独立的语义
    semantic_id = BACK_WALL_SEMANTIC_ID  # 后墙的语义ID
    color = [100, 200, 210]  # 颜色
