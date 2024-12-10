from aerial_gym.config.asset_config.base_asset import *

import numpy as np

# 定义语义ID常量，用于区分不同类型的环境对象
THIN_SEMANTIC_ID = 1  # 薄物体的语义ID
TREE_SEMANTIC_ID = 2  # 树木的语义ID
OBJECT_SEMANTIC_ID = 3  # 一般物体的语义ID
WALL_SEMANTIC_ID = 8  # 墙壁的语义ID


class EnvObjectConfig:
    # 面板资产参数配置类
    class panel_asset_params(BaseAssetParams):
        num_assets = 6  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/panels"  # 资产文件夹路径

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

        min_position_ratio = [0.3, 0.05, 0.05]  # 最大位置比例（相对于边界）
        max_position_ratio = [0.85, 0.95, 0.95]  # 最小位置比例（相对于边界）

        specified_position = [
            -1000.0,
            -1000.0,
            -1000.0,
        ]  # 如果值大于-900，则使用此值而不是随机化比例

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
        per_link_semantic = True  # 每个链接是否有独立的语义
        semantic_id = -1  # 语义ID，默认为-1
        color = [170, 66, 66]  # 颜色设置

    # 薄物体资产参数配置类
    class thin_asset_params(base_asset_params):
        num_assets = 0  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/thin"  # 资产文件夹路径

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        semantic_id = THIN_SEMANTIC_ID  # 语义ID
        color = [170, 66, 66]  # 颜色设置

    # 树木资产参数配置类
    class tree_asset_params(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/trees"  # 资产文件夹路径

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

        min_state_ratio = [
            0.2,
            0.05,
            0.05,
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
            0.9,
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
        per_link_semantic = False  # 每个链接是否有独立的语义
        semantic_id = TREE_SEMANTIC_ID  # 语义ID
        color = [70, 200, 100]  # 颜色设置

        semantic_masked_links = {}  # 语义屏蔽链接

    # 一般物体资产参数配置类
    class object_asset_params(base_asset_params):
        num_assets = 2  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"  # 资产文件夹路径

        min_state_ratio = [
            0.25,
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

        semantic_id = OBJECT_SEMANTIC_ID  # 语义ID

        # color = [80,255,100]  # 颜色设置（注释掉了）

    # 左墙资产参数配置类
    class left_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "left_wall.urdf"  # 资产文件名

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        per_link_semantic = True  # 每个链接是否有独立的语义
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 200, 210]  # 颜色设置

    # 右墙资产参数配置类
    class right_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "right_wall.urdf"  # 资产文件名

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
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 200, 210]  # 颜色设置

    # 顶部墙资产参数配置类
    class top_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "top_wall.urdf"  # 资产文件名

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        per_link_semantic = True  # 每个链接是否有独立的语义
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 200, 210]  # 颜色设置

    # 底部墙资产参数配置类
    class bottom_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "bottom_wall.urdf"  # 资产文件名

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        per_link_semantic = True  # 每个链接是否有独立的语义
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 150, 150]  # 颜色设置

    # 前墙资产参数配置类
    class front_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "front_wall.urdf"  # 资产文件名

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        per_link_semantic = True  # 每个链接是否有独立的语义
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 200, 210]  # 颜色设置

    # 后墙资产参数配置类
    class back_wall(base_asset_params):
        num_assets = 1  # 该类型资产的数量

        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"  # 资产文件夹路径
        file = "back_wall.urdf"  # 资产文件名

        collision_mask = 1  # 相同碰撞掩码的物体不会发生碰撞

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
        semantic_id = -1  # 语义ID，默认为-1
        color = [100, 200, 210]  # 颜色设置
