from typing import Any
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

# 创建自定义日志记录器，用于记录AssetManager的调试信息
logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict, num_keep_in_env):
        # 初始化资产管理器，调用init_tensors方法初始化张量
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        # 从全局张量字典中提取相关张量和参数
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]  # 环境资产状态张量
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]  # 资产最小状态比例
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]  # 资产最大状态比例
        
        # 设置环境边界的最小值和最大值，并扩展维度以适应资产状态张量的形状
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env  # 需要在环境中保留的资产数量

    def prepare_for_sim(self):
        # 为仿真做准备，重置资产状态
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions):
        # 物理步骤前的操作（目前未实现）
        pass

    def post_physics_step(self):
        # 物理步骤后的操作（目前未实现）
        pass

    def step(self, actions):
        # 在每一步执行的操作（目前未实现）
        pass
        # 如果需要，可以实现该功能以在步骤中对环境资产执行特定操作。
        # 对于静态环境，实际上不需要执行任何操作。
        # 如果需要施加力，应在其他类中处理，最好将此类保留用于操作状态张量。

    def reset(self, num_obstacles_per_env):
        # 重置环境资产状态，使用给定的障碍物数量
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        # 根据环境ID和障碍物数量重置资产状态
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            # 如果请求的障碍物数量小于最小要求，则使用最小数量
            num_obstacles_per_env = self.num_keep_in_env

        # 随机采样资产状态比例
        sampled_asset_state_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        
        # 根据采样的比例插值计算资产的坐标位置
        self.env_asset_state_tensor[env_ids, :, 0:3] = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )[env_ids, :, 0:3]
        
        # 将采样比例转换为四元数表示
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        
        # 将不需要的障碍物放置在环境外
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0
