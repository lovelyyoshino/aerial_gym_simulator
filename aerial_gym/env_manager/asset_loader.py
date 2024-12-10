import os
import random

from isaacgym import gymapi
from aerial_gym.assets.warp_asset import WarpAsset
from aerial_gym.assets.isaacgym_asset import IsaacGymAsset

from collections import deque

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_loader")

# 给后面env调用的
def asset_class_to_AssetOptions(asset_class):
    """
    将资产类转换为资产选项。

    参数:
    asset_class: 资产类对象，包含资产的配置参数。

    返回:
    asset_options: gymapi.AssetOptions对象，包含转换后的资产选项。
    """
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints  # 是否合并固定关节
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule  # 是否用胶囊替代圆柱
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments  # 是否翻转视觉附加物
    asset_options.fix_base_link = asset_class.fix_base_link  # 是否固定基础链接
    asset_options.density = asset_class.density  # 资产密度
    asset_options.angular_damping = asset_class.angular_damping  # 角阻尼
    asset_options.linear_damping = asset_class.linear_damping  # 线阻尼
    asset_options.max_angular_velocity = asset_class.max_angular_velocity  # 最大角速度
    asset_options.max_linear_velocity = asset_class.max_linear_velocity  # 最大线速度
    asset_options.disable_gravity = asset_class.disable_gravity  # 是否禁用重力
    return asset_options


class AssetLoader:
    def __init__(self, global_sim_dict, device):
        """
        初始化AssetLoader类。

        参数:
        global_sim_dict: 全局仿真字典，包含gym、sim和环境配置等信息。
        device: 设备信息，用于指定运行的设备。
        """
        self.global_sim_dict = global_sim_dict
        self.gym = self.global_sim_dict["gym"]  # 获取gym实例
        self.sim = self.global_sim_dict["sim"]  # 获取sim实例
        self.cfg = self.global_sim_dict["env_cfg"]  # 获取环境配置
        self.device = device  # 设备信息
        self.env_config = self.cfg.env_config  # 环境配置
        self.num_envs = self.cfg.env.num_envs  # 环境数量

        self.asset_buffer = {}  # 资产缓冲区
        self.global_asset_counter = 0  # 全局资产计数器

        self.max_loaded_semantic_id = 0  # 最大加载语义ID

    def randomly_pick_assets_from_folder(self, folder, num_assets=0):
        """
        从指定文件夹随机选择URDF文件。

        参数:
        folder: 文件夹路径。
        num_assets: 选择的资产数量，默认为0。

        返回:
        selected_files: 选中的URDF文件列表。
        """
        available_assets = []  # 可用资产列表
        for file in os.listdir(folder):
            if file.endswith(".urdf"):  # 判断文件是否为URDF文件
                available_assets.append(file)

        if num_assets == 0:
            return []

        selected_files = random.choices(available_assets, k=num_assets)  # 随机选择资产文件
        return selected_files

    def load_selected_file_from_config(
        self, asset_type, asset_class_config, selected_file, is_robot=False
    ):
        """
        从配置加载选定的资产文件。

        参数:
        asset_type: 资产类型。
        asset_class_config: 资产类配置。
        selected_file: 选定的文件名。
        is_robot: 是否为机器人，默认为False。

        返回:
        asset_class_dict: 资产类字典，包含资产的相关信息。
        """
        asset_options_for_class = asset_class_to_AssetOptions(asset_class_config)  # 转换为资产选项
        filepath = os.path.join(asset_class_config.asset_folder, selected_file)  # 构造文件路径

        # 检查资产是否已在缓冲区中
        if filepath in self.asset_buffer:
            return self.asset_buffer[filepath]

        logger.info(
            f"Loading asset: {selected_file} for the first time. Next use of this asset will be via the asset buffer."
        )

        asset_class_dict = {
            "asset_type": asset_type,
            "asset_options": asset_options_for_class,
            "semantic_id": asset_class_config.semantic_id,
            "collision_mask": asset_class_config.collision_mask,
            "color": asset_class_config.color,
            "semantic_masked_links": asset_class_config.semantic_masked_links,
            "keep_in_env": asset_class_config.keep_in_env,
            "filename": filepath,
            "asset_folder": asset_class_config.asset_folder,
            "per_link_semantic": asset_class_config.per_link_semantic,
            "min_state_ratio": asset_class_config.min_state_ratio,
            "max_state_ratio": asset_class_config.max_state_ratio,
            "place_force_sensor": asset_class_config.place_force_sensor,
            "force_sensor_parent_link": asset_class_config.force_sensor_parent_link,
            "force_sensor_transform": asset_class_config.force_sensor_transform,
            "use_collision_mesh_instead_of_visual": asset_class_config.use_collision_mesh_instead_of_visual,
            # 在这里处理位置、随机化等
        }
        max_list_vals = 0
        if len(list(asset_class_config.semantic_masked_links.values())) > 0:
            max_list_vals = max(list(asset_class_config.semantic_masked_links.values()))

        self.max_loaded_semantic_id = max(
            self.max_loaded_semantic_id, asset_class_config.semantic_id, max_list_vals
        )

        asset_name = asset_type  # 资产名称

        # 获取机器人传感器配置
        robot_sensor_config = self.global_sim_dict["robot_config"].sensor_config
        use_camera_collision_mesh = (
            robot_sensor_config.camera_config.use_collision_geometry
            if robot_sensor_config.enable_camera
            else False
        )

        if is_robot == False:
            if self.cfg.env.use_warp:
                warp_asset = WarpAsset(asset_name, filepath, asset_class_dict)  # 创建WarpAsset对象
                asset_class_dict["warp_asset"] = warp_asset  # 将WarpAsset添加到资产字典
                if use_camera_collision_mesh:
                    msg_str = (
                        "Warp cameras will render the mesh per each asset that is "
                        + "specified by the asset configuration file."
                        + "The parameter from the sensor configuration file"
                        + " to render collision meshes will be ignored."
                        + "This message is generated because you have set the"
                        + "use_collision_geometry parameter to True in the sensor"
                        + "configuration file."
                    )
                    logger.warning(msg_str)
            elif (
                use_camera_collision_mesh != asset_class_config.use_collision_mesh_instead_of_visual
            ):
                msg_str = (
                    "Choosing between collision and visual meshes per asset is not supported"
                    + "for Isaac Gym rendering pipeline. If the Isaac Gym rendering pipeline is selected, "
                    + "you can render only visual or only collision meshes for all assets."
                    + " Please make ensure that the appropriate option is set in the sensor configuration file."
                    + " Current simulation will run but will render only the mesh you set for the sensor "
                    + "configuration and the setting from the asset configuration will be ignored."
                    + "This message is generated because the use_collision_geometry parameter in the sensor"
                    + "configuration file is different from the use_collision_mesh_instead_of_visual parameter in the asset"
                    + "configuration file. \n\n\nThe simulation will still run but the rendering will be as per the sensor configuration file."
                )
                logger.warning(msg_str)

        IGE_asset = IsaacGymAsset(self.gym, self.sim, asset_name, filepath, asset_class_dict)  # 创建IsaacGymAsset对象
        asset_class_dict["isaacgym_asset"] = IGE_asset  # 将IsaacGymAsset添加到资产字典
        self.asset_buffer[filepath] = asset_class_dict  # 将资产信息存入缓冲区
        return asset_class_dict

    def select_and_order_assets(self):
        """
        选择并排序资产。

        返回:
        ordered_asset_list: 排序后的资产列表。
        keep_in_env_num: 在环境中保留的资产数量。
        """
        ordered_asset_list = deque()  # 用于存储排序后的资产列表
        keep_in_env_num = 0  # 在环境中保留的资产数量
        for (
            asset_type,
            asset_class_config,
        ) in self.env_config.asset_type_to_dict_map.items():
            num_assets = asset_class_config.num_assets  # 获取资产数量
            if (
                asset_type in self.env_config.include_asset_type
                and self.env_config.include_asset_type[asset_type] == False
            ):
                continue  # 如果资产类型不在包含列表中，则跳过

            if num_assets > 0:
                if asset_class_config.file is None:
                    selected_files = self.randomly_pick_assets_from_folder(
                        asset_class_config.asset_folder, num_assets
                    )  # 从文件夹中随机选择资产
                else:
                    selected_files = [asset_class_config.file] * num_assets  # 使用指定的文件

                for selected_file in selected_files:
                    asset_info_dict = self.load_selected_file_from_config(
                        asset_type, asset_class_config, selected_file
                    )  # 加载资产信息
                    if asset_info_dict["keep_in_env"]:
                        ordered_asset_list.appendleft(asset_info_dict)  # 将资产添加到列表前面
                        logger.debug(f"Asset {asset_type} kept in env")
                        keep_in_env_num += 1  # 更新保留资产数量
                    else:
                        ordered_asset_list.append(asset_info_dict)  # 将资产添加到列表后面

        # 打乱不一定保留在环境中的资产
        ordered_asset_list = list(ordered_asset_list)
        shuffle_subset = ordered_asset_list[keep_in_env_num:]  # 获取需要打乱的资产子集
        random.shuffle(shuffle_subset)  # 打乱子集
        ordered_asset_list[keep_in_env_num:] = shuffle_subset  # 更新排序后的列表
        return ordered_asset_list, keep_in_env_num

    def select_assets_for_sim(self):
        """
        为每个环境选择资产。

        返回:
        global_env_asset_dicts: 包含每个环境资产的字典列表。
        num_assets_kept_in_env: 在环境中保留的资产数量。
        """
        self.global_env_asset_dicts = []  # 存储每个环境的资产字典
        for i in range(self.num_envs):
            logger.debug(f"Loading assets for env: {i}")
            ordered_asset_list, num_assets_kept_in_env = self.select_and_order_assets()  # 选择并排序资产
            self.global_env_asset_dicts.append(ordered_asset_list)  # 将资产添加到全局字典
            logger.debug(f"Loaded assets for env {i}")
        return self.global_env_asset_dicts, num_assets_kept_in_env
