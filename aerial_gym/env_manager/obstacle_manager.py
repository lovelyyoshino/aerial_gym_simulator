from aerial_gym.env_manager.base_env_manager import BaseManager

# from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("obstacle_manager")

class ObstacleManager(BaseManager):
    def __init__(self, num_assets, config, device):
        super().__init__(config, device)  # 调用父类构造函数
        self.global_tensor_dict = {}  # 存储全局张量字典
        self.num_assets = num_assets  # 资产数量

        logger.debug("Obstacle Manager initialized")  # 记录初始化日志

    def prepare_for_sim(self, global_tensor_dict):
        # 准备模拟环境，提取障碍物的位置信息和速度信息
        if self.num_assets <= 1:
            return  # 如果资产数量小于等于1，则无需准备
        self.global_tensor_dict = global_tensor_dict  # 更新全局张量字典
        self.obstacle_position = global_tensor_dict["obstacle_position"]  # 获取障碍物位置
        self.obstacle_orientation = global_tensor_dict["obstacle_orientation"]  # 获取障碍物朝向
        self.obstacle_linvel = global_tensor_dict["obstacle_linvel"]  # 获取障碍物线速度
        self.obstacle_angvel = global_tensor_dict["obstacle_angvel"]  # 获取障碍物角速度

        # self.obstacle_force_tensors = global_tensor_dict["obstacle_force_tensor"]
        # self.obstacle_torque_tensors = global_tensor_dict["obstacle_torque_tensor"]

    def reset(self):
        # 重置障碍物管理器的状态
        # self.controller.reset()  # 假设有一个控制器需要重置
        return

    def reset_idx(self, env_ids):
        # 根据环境ID重置障碍物的状态
        # self.controller.reset_idx(env_ids)  # 假设有一个控制器需要重置特定的环境ID
        return

    def pre_physics_step(self, actions=None):
        # 在物理步骤之前更新障碍物的线速度和角速度
        if self.num_assets <= 1 or actions is None:
            return  # 如果资产数量小于等于1或者没有动作，则返回
        self.obstacle_linvel[:] = actions[:, :, 0:3]  # 更新障碍物线速度
        self.obstacle_angvel[:] = actions[:, :, 3:6]  # 更新障碍物角速度
        # self.update_states()  # 更新状态（已注释）
        # self.obstacle_wrench[:] = self.controller(actions)  # 计算障碍物的外力
        # self.obstacle_force_tensors[:] = self.obstacle_wrench[:, :, 0:3]  # 更新外力张量
        # self.obstacle_torque_tensors[:] = self.obstacle_wrench[:, :, 3:6]  # 更新外力矩张量

    def step(self):
        # 执行一步模拟（当前未实现）
        pass

    # def update_states(self):
    #     # 更新障碍物的状态
    #     self.obstacle_euler_angles[:] = ssa(get_euler_xyz_tensor(self.obstacle_orientation))  # 获取欧拉角
    #     self.obstacle_vehicle_orientation[:] = vehicle_frame_quat_from_quat(self.obstacle_orientation)  # 获取车辆框架方向
    #     self.obstacle_vehicle_linvel[:] = quat_rotate_inverse(  # 更新车辆框架下的线速度
    #         self.obstacle_vehicle_orientation, self.obstacle_linvel
    #     )
    #     self.obstacle_body_linvel[:] = quat_rotate_inverse(self.obstacle_orientation, self.obstacle_linvel)  # 更新障碍物的线速度
    #     self.obstacle_body_angvel[:] = quat_rotate_inverse(self.obstacle_orientation, self.obstacle_angvel)  # 更新障碍物的角速度
