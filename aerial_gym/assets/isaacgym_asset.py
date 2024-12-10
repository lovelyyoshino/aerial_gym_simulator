from urdfpy import URDF  # 导入URDF库，用于处理URDF文件
import numpy as np  # 导入NumPy库，用于数值计算

import trimesh as tm  # 导入trimesh库，用于处理三维网格

from isaacgym import gymapi  # 从isaacgym导入gymapi模块，提供与物理仿真相关的API

from aerial_gym.assets.base_asset import BaseAsset  # 从aerial_gym.assets.base_asset导入BaseAsset类


class IsaacGymAsset(BaseAsset):
    def __init__(self, gym, sim, asset_name, asset_file, loading_options):
        """
        初始化IsaacGymAsset对象
        
        参数:
        gym: Isaac Gym实例
        sim: 仿真环境实例
        asset_name: 资产名称
        asset_file: 资产文件路径
        loading_options: 加载选项
        """
        super().__init__(asset_name, asset_file, loading_options)  # 调用父类构造函数初始化基本属性
        self.gym = gym  # 保存传入的gym实例
        self.sim = sim  # 保存传入的sim实例
        self.load_from_file(self.file)  # 从指定文件加载资产

    def load_from_file(self, asset_file):
        """
        从文件加载资产
        
        参数:
        asset_file: 资产文件路径
        """
        file = asset_file.split("/")[-1]  # 获取文件名（去掉路径）
        
        # 使用gym API加载资产
        self.asset = self.gym.load_asset(
            self.sim, self.options.asset_folder, file, self.options.asset_options
        )

        # 如果需要放置力传感器，则进行以下操作
        if self.options.place_force_sensor:
            # 查找父链接的索引
            parent_link_idx = self.gym.find_asset_rigid_body_index(
                self.asset, self.options.force_sensor_parent_link
            )
            
            # 创建力传感器的变换矩阵
            self.force_sensor_transform = gymapi.Transform()
            self.force_sensor_transform.p = gymapi.Vec3(
                self.options.force_sensor_transform[0],
                self.options.force_sensor_transform[1],
                self.options.force_sensor_transform[2],
            )  # 设置位置
            
            self.force_sensor_transform.r = gymapi.Quat(
                self.options.force_sensor_transform[3],
                self.options.force_sensor_transform[4],
                self.options.force_sensor_transform[5],
                self.options.force_sensor_transform[6],
            )  # 设置旋转
            
            # 配置力传感器的属性
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True  # 启用前向动力学力
            sensor_props.enable_constraint_solver_forces = True  # 启用约束求解器力
            sensor_props.use_world_frame = False  # 不使用世界坐标系
            
            # 创建力传感器并保存句柄
            self.force_sensor_handle = self.gym.create_asset_force_sensor(
                self.asset, parent_link_idx, self.force_sensor_transform, sensor_props
            )
