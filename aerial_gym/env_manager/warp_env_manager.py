from aerial_gym.env_manager.base_env_manager import BaseManager

import warp as wp
import numpy as np
import torch

import trimesh as tm

from aerial_gym.utils.math import tf_apply

# 初始化warp库
wp.init()

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_env_manager")


class WarpEnv(BaseManager):
    def __init__(self, global_sim_dict, device):
        logger.debug("Initializing WarpEnv")  # 调试日志，初始化WarpEnv
        super().__init__(global_sim_dict["env_cfg"], device)  # 调用父类构造函数
        self.num_envs = global_sim_dict["num_envs"]  # 环境数量
        self.env_meshes = []  # 存储环境网格
        self.warp_mesh_id_list = []  # 存储warp网格ID
        self.warp_mesh_per_env = []  # 每个环境的warp网格
        self.global_vertex_to_asset_index_tensor = None  # 全局顶点到资产索引的张量
        self.vertex_maps_per_env_original = None  # 原始顶点映射
        self.global_env_mesh_list = []  # 全局环境网格列表
        self.global_vertex_counter = 0  # 全局顶点计数器
        self.global_vertex_segmentation_list = []  # 全局顶点分割列表
        self.global_vertex_to_asset_index_map = []  # 全局顶点到资产索引的映射

        # 常量初始化
        self.CONST_WARP_MESH_ID_LIST = None
        self.CONST_WARP_MESH_PER_ENV = None
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = None
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = None
        logger.debug("[DONE] Initializing WarpEnv")  # 调试日志，WarpEnv初始化完成

    def reset_idx(self, env_ids):
        if self.global_vertex_counter == 0:  # 如果没有顶点，则返回
            return
        logger.debug("Updating vertex maps per env")  # 调试日志，更新每个环境的顶点映射
        self.vertex_maps_per_env_updated[:] = tf_apply(
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 3:7],
            self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 0:3],
            self.VERTEX_MAPS_PER_ENV_ORIGINAL[:],
        )
        logger.debug("[DONE] Updating vertex maps per env")  # 调试日志，更新完成

        logger.debug("Refitting warp meshes")  # 调试日志，重新调整warp网格
        for i in env_ids:
            self.warp_mesh_per_env[i].refit()  # 调整每个环境的warp网格
        logger.debug("[DONE] Refitting warp meshes")  # 调试日志，调整完成

    def pre_physics_step(self, action):
        pass  # 物理步骤前的操作（待实现）

    def post_physics_step(self):
        pass  # 物理步骤后的操作（待实现）

    def step(self, action):
        pass  # 单步执行（待实现）

    def reset(self):
        return self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 重置环境

    def create_env(self, env_id):
        if len(self.env_meshes) <= env_id:  # 如果环境数量小于或等于环境ID，则添加新环境
            self.env_meshes.append([])
        else:
            raise ValueError("Environment already exists")  # 抛出错误，环境已存在

    def add_asset_to_env(self, asset_info_dict, env_id, global_asset_counter, segmentation_counter):
        warp_asset = asset_info_dict["warp_asset"]  # 获取warp资产
        # 使用变量分割掩码设置每个顶点的分割ID
        updated_vertex_segmentation = (
            warp_asset.asset_vertex_segmentation_value
            + segmentation_counter * warp_asset.variable_segmentation_mask
        )
        logger.debug(
            f"Asset {asset_info_dict['filename']} has {len(warp_asset.asset_unified_mesh.vertices)} vertices. Segmentation mask: {warp_asset.variable_segmentation_mask} and updated segmentation: {updated_vertex_segmentation}"
        )  # 调试日志，输出资产信息
        self.env_meshes[env_id].append(warp_asset.asset_unified_mesh)  # 将统一网格添加到环境中

        self.global_vertex_to_asset_index_map += [global_asset_counter] * len(
            warp_asset.asset_unified_mesh.vertices
        )  # 更新全局顶点到资产索引的映射
        self.global_vertex_counter += len(warp_asset.asset_unified_mesh.vertices)  # 更新全局顶点计数器
        self.global_vertex_segmentation_list += updated_vertex_segmentation.tolist()  # 更新全局顶点分割列表
        return None, len(
            np.unique(
                warp_asset.asset_vertex_segmentation_value * warp_asset.variable_segmentation_mask
            )
        )  # 返回None和唯一分割数量

    def prepare_for_simulation(self, global_tensor_dict):
        logger.debug("Preparing for simulation")  # 调试日志，准备仿真
        self.global_tensor_dict = global_tensor_dict  # 存储全局张量字典
        if self.global_vertex_counter == 0:  # 如果没有资产添加到环境，跳过准备
            logger.warning(
                "No assets have been added to the environment. Skipping preparation for simulation"
            )  # 警告日志，未添加资产
            self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = None
            self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = None
            self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = None
            self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = None
            return 1

        # 创建全局顶点到资产索引的张量
        self.global_vertex_to_asset_index_tensor = torch.tensor(
            self.global_vertex_to_asset_index_map,
            device=self.device,
            requires_grad=False,
        )
        # 创建原始顶点映射张量
        self.vertex_maps_per_env_original = torch.zeros(
            (self.global_vertex_counter, 3), device=self.device, requires_grad=False
        )
        # 更新的顶点映射用于warp环境
        self.vertex_maps_per_env_updated = self.vertex_maps_per_env_original.clone()

        ## 统一环境网格
        logger.debug("Unifying environment meshes")  # 调试日志，统一环境网格
        for i in range(len(self.env_meshes)):
            self.global_env_mesh_list.append(tm.util.concatenate(self.env_meshes[i]))  # 连接环境网格
        logger.debug("[DONE] Unifying environment meshes")  # 调试日志，统一完成

        # 准备warp网格
        logger.debug("Creating warp meshes")  # 调试日志，创建warp网格
        vertex_iterator = 0  # 顶点迭代器初始化
        for env_mesh in self.global_env_mesh_list:
            self.vertex_maps_per_env_original[
                vertex_iterator : vertex_iterator + len(env_mesh.vertices)
            ] = torch.tensor(env_mesh.vertices, device=self.device, requires_grad=False)  # 将网格顶点添加到原始映射
            faces_tensor = torch.tensor(
                env_mesh.faces,
                device=self.device,
                requires_grad=False,
                dtype=torch.int32,
            )  # 创建面索引张量
            vertex_velocities = torch.zeros(
                len(env_mesh.vertices), 3, device=self.device, requires_grad=False
            )  # 创建顶点速度张量
            segmentation_tensor = torch.tensor(
                self.global_vertex_segmentation_list[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                device=self.device,
                requires_grad=False,
            )  # 创建分割张量
            # 我们劫持这个字段并将其用于分割
            vertex_velocities[:, 0] = segmentation_tensor  # 将分割信息赋值给速度张量的第一列

            vertex_vec3_array = wp.from_torch(
                self.vertex_maps_per_env_updated[
                    vertex_iterator : vertex_iterator + len(env_mesh.vertices)
                ],
                dtype=wp.vec3,
            )  # 将更新的顶点映射转换为warp格式
            faces_wp_int32_array = wp.from_torch(faces_tensor.flatten(), dtype=wp.int32)  # 转换面索引为warp格式
            velocities_vec3_array = wp.from_torch(vertex_velocities, dtype=wp.vec3)  # 转换速度为warp格式

            wp_mesh = wp.Mesh(
                points=vertex_vec3_array,
                indices=faces_wp_int32_array,
                velocities=velocities_vec3_array,
            )  # 创建warp网格

            self.warp_mesh_per_env.append(wp_mesh)  # 将warp网格添加到每个环境
            self.warp_mesh_id_list.append(wp_mesh.id)  # 存储warp网格ID
            vertex_iterator += len(env_mesh.vertices)  # 更新顶点迭代器

        logger.debug("[DONE] Creating warp meshes")  # 调试日志，创建完成
        # 定义常量，以便在准备好环境后访问
        self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
        self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = self.global_vertex_to_asset_index_tensor
        self.VERTEX_MAPS_PER_ENV_ORIGINAL = self.vertex_maps_per_env_original

        # 更新全局张量字典
        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
        self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
        self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = (
            self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR
        )
        self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = self.VERTEX_MAPS_PER_ENV_ORIGINAL

        self.unfolded_env_vec_root_tensor = self.global_tensor_dict[
            "unfolded_env_asset_state_tensor"
        ]  # 获取展开的环境资产状态张量
        return 1  # 返回1，表示准备完成
