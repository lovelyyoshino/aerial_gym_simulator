from urdfpy import URDF
import numpy as np

import trimesh as tm

from aerial_gym.assets.base_asset import BaseAsset


from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


class WarpAsset(BaseAsset):
    def __init__(self, asset_name, asset_file, loading_options):
        # 初始化WarpAsset类，调用父类构造函数并加载资产文件
        super().__init__(asset_name, asset_file, loading_options)
        self.load_from_file(self.file)

    def load_from_file(self, asset_file):
        # 从指定的资产文件中加载URDF模型
        self.file = asset_file
        # 获取trimesh碰撞和可视化网格
        self.urdf_asset = URDF.load(asset_file)  # 加载URDF文件
        self.visual_mesh_items = self.urdf_asset.visual_trimesh_fk().items()  # 获取可视化网格项
        self.urdf_named_links = [key.name for key in self.urdf_asset.link_fk().keys()]  # 获取命名链接列表

        self.asset_meshes = []  # 存储资产网格
        self.asset_vertex_segmentation_value = []  # 存储每个顶点的分割值
        self.variable_segmentation_mask = []  # 存储变量分割掩码

        mesh_items = self.visual_mesh_items  # 默认使用可视化网格项

        # 如果选项要求使用碰撞网格而不是可视化网格，则进行相应处理
        if self.options.use_collision_mesh_instead_of_visual:
            self.collision_mesh_items = self.urdf_asset.collision_trimesh_fk().items()  # 获取碰撞网格项
            mesh_items = self.collision_mesh_items  # 使用碰撞网格项替代可视化网格项

            # 重命名仅包含具有碰撞网格的链接的命名链接，因为collision_mesh_fk函数不包括没有碰撞网格的链接。
            temp_named_links_with_collision_meshes = []
            for linkname in self.urdf_named_links:
                if self.urdf_asset.link_map[linkname].collision_mesh is not None:
                    temp_named_links_with_collision_meshes.append(linkname)
            self.urdf_named_links = temp_named_links_with_collision_meshes  # 更新命名链接列表

        mesh_index = 0  # 网格索引初始化
        self.segmentation_id = self.options.semantic_id  # 分割ID从选项中获取
        self.segmentation_counter = 0  # 分割计数器初始化

        if self.segmentation_id < 0:  # 如果分割ID小于0，则使用计数器作为分割ID
            self.segmentation_id = self.segmentation_counter

        for mesh, mesh_tf in mesh_items:
            # 在此上下文中，mesh指的是一个链接的网格
            generalized_mesh_vertices = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]  # 将顶点转换为齐次坐标
            # 将顶点变换到正确的坐标系
            generalized_mesh_vertices_tf = np.matmul(mesh_tf, generalized_mesh_vertices.T).T
            mesh.vertices[:] = generalized_mesh_vertices_tf[:, 0:3]  # 更新网格顶点位置

            # 如果配置指定资产应该有per_link_segmentation，
            # 则我们为每个链接分配递增的分割值，除非在配置中为该链接分配了特定的分割值。
            while self.segmentation_counter in self.options.semantic_masked_links.values():  # 跳过已定义的分割值
                self.segmentation_counter += 1  # 增加计数器

            links_to_segment = self.options.semantic_masked_links.keys()  # 获取需要分割的链接
            if len(links_to_segment) == 0:  # 如果没有预定义的链接，则使用所有命名链接
                links_to_segment = self.urdf_named_links

            if self.options.per_link_semantic:  # 如果启用逐链接语义
                if self.urdf_named_links[mesh_index] in links_to_segment:  # 检查当前链接是否在预定义列表中
                    if self.urdf_named_links[mesh_index] in self.options.semantic_masked_links:
                        object_segmentation_id = self.options.semantic_masked_links[
                            self.urdf_named_links[mesh_index]
                        ]  # 使用预定义的分割ID
                        variable_segmentation_mask_value = 0  # 不使用变量分割掩码
                    else:
                        object_segmentation_id = self.segmentation_counter  # 使用计数器作为分割ID
                        self.segmentation_counter += 1  # 增加计数器
                        variable_segmentation_mask_value = 1  # 使用变量分割掩码
                else:
                    object_segmentation_id = self.segmentation_counter  # 使用计数器作为分割ID
                    variable_segmentation_mask_value = 1  # 使用变量分割掩码
                logger.debug(
                    f"Mesh name {self.urdf_named_links[mesh_index]} has segmentation id {object_segmentation_id}"
                )
            else:
                if self.options.semantic_id < 0:  # 如果分割ID小于0
                    logger.debug("Segmentation id is negative. Using the counter.")
                    object_segmentation_id = self.segmentation_counter  # 使用计数器作为分割ID
                    variable_segmentation_mask_value = 1  # 使用变量分割掩码
                else:
                    object_segmentation_id = self.segmentation_id  # 使用选项中的分割ID
                    variable_segmentation_mask_value = 0  # 不使用变量分割掩码
                logger.debug(
                    f"Mesh name {self.urdf_named_links[mesh_index]} has segmentation id {object_segmentation_id}"
                    + f" and variable_segmentation_mask_value {variable_segmentation_mask_value}"
                )

            self.asset_meshes.append(mesh)  # 添加网格到资产网格列表
            self.asset_vertex_segmentation_value += [object_segmentation_id] * len(mesh.vertices)  # 为每个顶点添加分割值
            self.variable_segmentation_mask += [variable_segmentation_mask_value] * len(
                mesh.vertices
            )  # 为每个顶点添加变量分割掩码
            mesh_index += 1  # 增加网格索引

        self.asset_unified_mesh = tm.util.concatenate(self.asset_meshes)  # 合并所有网格为统一网格
        self.asset_vertex_segmentation_value = np.array(self.asset_vertex_segmentation_value)  # 转换为NumPy数组
        logger.debug(
            f"Asset {asset_file} has {len(self.asset_unified_mesh.vertices)} vertices. Segmentation mask: {self.variable_segmentation_mask}"
        )
        logger.debug(f"Asset vertex segmentation value: {self.asset_vertex_segmentation_value}")
        self.variable_segmentation_mask = np.array(self.variable_segmentation_mask)  # 转换为NumPy数组

        # 确保分割值、统一网格顶点和分割掩码长度一致
        assert len(self.asset_vertex_segmentation_value) == len(
            self.asset_unified_mesh.vertices
        ), f"len(self.asset_vertex_segmentation_value) = {len(self.asset_vertex_segmentation_value)}, len(self.asset_unified_mesh.vertices) = {len(self.asset_unified_mesh.vertices)}"

        assert len(self.variable_segmentation_mask) == len(
            self.asset_unified_mesh.vertices
        ), f"len(self.variable_segmentation_mask) = {len(self.variable_segmentation_mask)}, len(self.asset_unified_mesh.vertices) = {len(self.asset_unified_mesh.vertices)}"

        assert len(self.variable_segmentation_mask) == len(
            self.asset_vertex_segmentation_value
        ), f"len(self.variable_segmentation_mask) = {len(self.variable_segmentation_mask)}, len(self.asset_vertex_segmentation_value) = {len(self.asset_vertex_segmentation_value)}"
