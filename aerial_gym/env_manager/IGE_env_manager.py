from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.IGE_viewer_control import IGEViewerControl
import torch
import os
from aerial_gym.utils.math import torch_rand_float_tensor
from aerial_gym.utils.helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    parse_sim_params,
)
import numpy as np
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacGymEnvManager")


class IsaacGymEnv(BaseManager):
    def __init__(self, config, sim_config, has_IGE_cameras, device):
        """
        初始化IsaacGym环境管理器
        :param config: 环境配置
        :param sim_config: 仿真配置
        :param has_IGE_cameras: 是否有IGE摄像头
        :param device: 设备类型（CPU/GPU）
        """
        super().__init__(config, device)
        self.sim_config = sim_config
        self.env_tensor_bounds_min = None
        self.env_tensor_bounds_max = None
        self.asset_handles = []
        self.env_handles = []
        self.num_rigid_bodies_robot = None
        self.has_IGE_cameras = has_IGE_cameras
        self.sim_has_dof = False
        self.dof_control_mode = "none"

        logger.info("Creating Isaac Gym Environment")
        self.gym, self.sim = self.create_sim()  # 创建仿真环境
        logger.info("Created Isaac Gym Environment")

        # 环境边界
        self.env_lower_bound_min = torch.tensor(
            self.cfg.env.lower_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_lower_bound_max = torch.tensor(
            self.cfg.env.lower_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_min = torch.tensor(
            self.cfg.env.upper_bound_min, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)
        self.env_upper_bound_max = torch.tensor(
            self.cfg.env.upper_bound_max, device=self.device, requires_grad=False
        ).expand(self.cfg.env.num_envs, -1)

        self.env_lower_bound = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )
        self.env_upper_bound = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )

        self.viewer = None
        self.graphics_are_stepped = True

    def create_sim(self):
        """
        创建一个gym对象，并用适当的仿真参数初始化
        :return: gym对象和仿真对象
        """
        logger.info("Acquiring gym object")
        self.gym = gymapi.acquire_gym()  # 获取gym对象
        logger.info("Acquired gym object")
        
        # 从命令行或配置文件解析参数
        args = get_args()

        # 将仿真和环境配置类合并为字典
        sim_config_dict = dict(class_to_dict(self.sim_config))
        env_config_dict = dict(class_to_dict(self.cfg))
        combined_dict = {**sim_config_dict, **env_config_dict}
        sim_cfg = update_cfg_from_args(combined_dict, args)

        self.simulator_params = parse_sim_params(args, sim_cfg)

        logger.info("Fixing devices")
        args.sim_device = self.device
        if self.simulator_params.use_gpu_pipeline == "False":
            logger.critical(
                "The use_gpu_pipeline is set to False, this will result in slower simulation times"
            )
        self.sim_device_type, self.sim_device_id = gymutil.parse_device_str(args.sim_device)
        logger.info(
            "Sim Device type: {}, Sim Device ID: {}".format(
                self.sim_device_type, self.sim_device_id
            )
        )
        if self.sim_config.viewer.headless and not self.has_IGE_cameras:
            self.graphics_device_id = -1
            logger.critical(
                "\n Setting graphics device to -1."
                + "\n This is done because the simulation is run in headless mode and no Isaac Gym cameras are used."
                + "\n No need to worry. The simulation and warp rendering will work as expected."
            )
        else:
            self.graphics_device_id = self.sim_device_id
        logger.info("Graphics Device ID: {}".format(self.graphics_device_id))
        
        logger.info("Creating Isaac Gym Simulation Object")
        warn_msg1 = (
            "If you have set the CUDA_VISIBLE_DEVICES environment variable, please ensure that you set it\n"
            + "to a particular one that works for your system to use the viewer or Isaac Gym cameras.\n"
            + "If you want to run parallel simulations on multiple GPUs with camera sensors,\n"
            + "please disable Isaac Gym and use warp (by setting use_warp=True), set the viewer to headless."
        )
        logger.warning(warn_msg1)
        warn_msg2 = (
            "If you see a segfault in the next lines, it is because of the discrepancy between the CUDA device and the graphics device.\n"
            + "Please ensure that the CUDA device and the graphics device are the same."
        )
        logger.warning(warn_msg2)
        
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, args.physics_engine, self.simulator_params
        )
        logger.info("Created Isaac Gym Simulation Object")
        return self.gym, self.sim

    def create_ground_plane(self):
        """
        创建地面平面
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # 设置平面的法向量
        self.gym.add_ground(self.sim, plane_params)  # 添加地面到仿真中
        return

    def create_env(self, env_id):
        """
        创建一个具有给定ID的环境
        :param env_id: 环境ID
        :return: 环境句柄
        """
        min_bound_vec3 = gymapi.Vec3(
            self.cfg.env.lower_bound_min[0],
            self.cfg.env.lower_bound_min[1],
            self.cfg.env.lower_bound_min[2],
        )
        max_bound_vec3 = gymapi.Vec3(
            self.cfg.env.upper_bound_max[0],
            self.cfg.env.upper_bound_max[1],
            self.cfg.env.upper_bound_max[2],
        )
        env_handle = self.gym.create_env(
            self.sim,
            min_bound_vec3,
            max_bound_vec3,
            int(np.sqrt(self.cfg.env.num_envs)),  # 环境的排列方式
        )
        if len(self.env_handles) <= env_id:
            self.env_handles.append(env_handle)  # 添加环境句柄
            self.asset_handles.append([])
        else:
            raise ValueError("Environment already exists")  # 如果环境已存在，抛出异常
        return env_handle

    def reset(self):
        """
        重置环境
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def add_asset_to_env(
        self,
        asset_info_dict,
        env_handle,
        env_id,
        global_asset_counter,
        segmentation_counter,
    ):
        """
        将资产添加到环境中
        :param asset_info_dict: 资产信息字典
        :param env_handle: 环境句柄
        :param env_id: 环境ID
        :param global_asset_counter: 全局资产计数器
        :param segmentation_counter: 分割计数器
        :return: 资产句柄和更新后的分割计数器
        """
        local_segmentation_ctr_for_isaacgym_asset = segmentation_counter
        if asset_info_dict["semantic_id"] < 0:
            asset_segmentation_id = local_segmentation_ctr_for_isaacgym_asset
            local_segmentation_ctr_for_isaacgym_asset += 1
        else:
            asset_segmentation_id = asset_info_dict["semantic_id"]
            local_segmentation_ctr_for_isaacgym_asset += 1

        asset_handle = self.gym.create_actor(
            env_handle,
            asset_info_dict["isaacgym_asset"].asset,
            gymapi.Transform(),
            "env_asset_" + str(global_asset_counter),
            env_id,
            asset_info_dict["collision_mask"],
            asset_segmentation_id,
        )

        if asset_info_dict["asset_type"] == "robot":
            self.num_rigid_bodies_robot = self.gym.get_actor_rigid_body_count(
                env_handle, asset_handle
            )

        if asset_info_dict["per_link_semantic"]:
            rigid_body_names_all = self.gym.get_actor_rigid_body_names(env_handle, asset_handle)

            if not type(asset_info_dict["semantic_masked_links"]) == dict:
                raise ValueError("semantic_masked_links should be a dictionary")

            links_to_label = asset_info_dict["semantic_masked_links"].keys()
            if len(links_to_label) == 0:
                links_to_label = rigid_body_names_all

            for name in rigid_body_names_all:

                # 跳过已经在字典中的值，这些值是为感兴趣对象预定义的
                while (
                    local_segmentation_ctr_for_isaacgym_asset
                    in asset_info_dict["semantic_masked_links"].values()
                ):
                    local_segmentation_ctr_for_isaacgym_asset += 1

                if name in links_to_label:
                    if name in asset_info_dict["semantic_masked_links"]:
                        segmentation_value = asset_info_dict["semantic_masked_links"][name]
                        logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                    else:
                        segmentation_value = local_segmentation_ctr_for_isaacgym_asset
                        local_segmentation_ctr_for_isaacgym_asset += 1
                        logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                else:
                    segmentation_value = local_segmentation_ctr_for_isaacgym_asset
                    logger.debug(f"Setting segmentation id for {name} to {segmentation_value}")
                index = rigid_body_names_all.index(name)
                self.gym.set_rigid_body_segmentation_id(
                    env_handle, asset_handle, index, segmentation_value
                )

        color = asset_info_dict["color"]
        if asset_info_dict["color"] is None:
            color = np.random.randint(low=50, high=200, size=3)

        self.gym.set_rigid_body_color(
            env_handle,
            asset_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(color[0] / 255, color[1] / 255, color[2] / 255),
        )

        self.asset_handles[env_id].append(asset_handle)  # 添加资产句柄
        return (
            asset_handle,
            local_segmentation_ctr_for_isaacgym_asset - segmentation_counter,
        )

    def prepare_for_simulation(self, env_manager, global_tensor_dict):
        """
        准备仿真
        :param env_manager: 环境管理器
        :param global_tensor_dict: 全局张量字典
        :return: 布尔值，表示准备是否成功
        """
        if not self.gym.prepare_sim(self.sim):
            raise RuntimeError("Failed to prepare Isaac Gym Environment")

        self.create_viewer(env_manager)  # 创建查看器
        self.has_viewer = self.viewer is not None

        # 检查每个环境是否有相同数量的资产
        self.num_envs = len(self.env_handles)
        self.num_assets_per_env = [len(assets) for assets in self.asset_handles]

        if not all(
            [num_assets == self.num_assets_per_env[0] for num_assets in self.num_assets_per_env]
        ):
            raise ValueError("All environments should have the same number of assets")

        self.num_assets_per_env = self.num_assets_per_env[0]

        # 检查所有环境是否有相同数量的刚体
        self.num_rigid_bodies_per_env = [
            self.gym.get_env_rigid_body_count(self.env_handles[i]) for i in range(self.num_envs)
        ]

        if not all(
            [
                num_rigid_bodies == self.num_rigid_bodies_per_env[0]
                for num_rigid_bodies in self.num_rigid_bodies_per_env
            ]
        ):
            raise ValueError("All environments should have the same number of rigid bodies.")

        self.num_rigid_bodies_per_env = self.num_rigid_bodies_per_env[0]

        self.unfolded_vec_root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.unfolded_vec_root_tensor = gymtorch.wrap_tensor(self.unfolded_vec_root_tensor)

        self.global_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.global_contact_force_tensor = gymtorch.wrap_tensor(
            self.global_contact_force_tensor
        ).view(self.num_envs, self.num_rigid_bodies_per_env, -1)

        self.vec_root_tensor = self.unfolded_vec_root_tensor.view(
            self.num_envs, self.num_assets_per_env, -1
        )

        self.global_tensor_dict = global_tensor_dict

        # 填充所有公共环境张量
        self.global_tensor_dict["vec_root_tensor"] = self.vec_root_tensor
        # 如果你的仿真有多个机器人，使用的不仅仅是索引0
        self.global_tensor_dict["robot_state_tensor"] = self.vec_root_tensor[:, 0, :]
        self.global_tensor_dict["env_asset_state_tensor"] = self.vec_root_tensor[:, 1:, :]
        self.global_tensor_dict["unfolded_env_asset_state_tensor"] = self.unfolded_vec_root_tensor
        self.global_tensor_dict["unfolded_env_asset_state_tensor_const"] = self.global_tensor_dict[
            "unfolded_env_asset_state_tensor"
        ].clone()

        self.global_tensor_dict["rigid_body_state_tensor"] = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        self.global_tensor_dict["global_force_tensor"] = torch.zeros(
            (self.global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
            device=self.device,
            requires_grad=False,
        )
        self.global_tensor_dict["global_torque_tensor"] = torch.zeros(
            (self.global_tensor_dict["rigid_body_state_tensor"].shape[0], 3),
            device=self.device,
            requires_grad=False,
        )

        self.global_tensor_dict["unfolded_dof_state_tensor"] = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )
        # 如果不为None，将张量视为(num_envs, num_dofs, 2)
        if not self.global_tensor_dict["unfolded_dof_state_tensor"] is None:
            self.sim_has_dof = True
            self.global_tensor_dict["dof_state_tensor"] = self.global_tensor_dict[
                "unfolded_dof_state_tensor"
            ].view(self.num_envs, -1, 2)

        self.global_tensor_dict["global_contact_force_tensor"] = self.global_contact_force_tensor
        self.global_tensor_dict["robot_contact_force_tensor"] = self.global_contact_force_tensor[
            :, 0, :
        ]

        # 填充机器人张量
        self.global_tensor_dict["robot_position"] = self.global_tensor_dict["robot_state_tensor"][
            :, :3
        ]
        self.global_tensor_dict["robot_orientation"] = self.global_tensor_dict[
            "robot_state_tensor"
        ][:, 3:7]
        self.global_tensor_dict["robot_linvel"] = self.global_tensor_dict["robot_state_tensor"][
            :, 7:10
        ]
        self.global_tensor_dict["robot_angvel"] = self.global_tensor_dict["robot_state_tensor"][
            :, 10:
        ]
        self.global_tensor_dict["robot_body_angvel"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 10:13]
        )
        self.global_tensor_dict["robot_body_linvel"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 7:10]
        )
        self.global_tensor_dict["robot_euler_angles"] = torch.zeros_like(
            self.global_tensor_dict["robot_state_tensor"][:, 7:10]
        )

        idx = self.num_rigid_bodies_robot
        self.global_tensor_dict["robot_force_tensor"] = self.global_tensor_dict[
            "global_force_tensor"
        ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]

        self.global_tensor_dict["robot_torque_tensor"] = self.global_tensor_dict[
            "global_torque_tensor"
        ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]

        # ==============================
        # 填充障碍物张量
        if self.num_assets_per_env > 0:
            self.global_tensor_dict["obstacle_position"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 0:3]
            self.global_tensor_dict["obstacle_orientation"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 3:7]
            self.global_tensor_dict["obstacle_linvel"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 7:10]
            self.global_tensor_dict["obstacle_angvel"] = self.global_tensor_dict[
                "env_asset_state_tensor"
            ][:, :, 10:]
            self.global_tensor_dict["obstacle_body_angvel"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 10:13]
            )
            self.global_tensor_dict["obstacle_body_linvel"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
            )
            self.global_tensor_dict["obstacle_euler_angles"] = torch.zeros_like(
                self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
            )

            # 假设每个障碍物被压缩为一个基本链接
            self.global_tensor_dict["obstacle_force_tensor"] = self.global_tensor_dict[
                "global_force_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]

            self.global_tensor_dict["obstacle_torque_tensor"] = self.global_tensor_dict[
                "global_torque_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]

        self.global_tensor_dict["env_bounds_max"] = self.env_upper_bound
        self.global_tensor_dict["env_bounds_min"] = self.env_lower_bound
        self.global_tensor_dict["gravity"] = torch.tensor(
            self.sim_config.sim.gravity, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.global_tensor_dict["dt"] = self.sim_config.sim.dt
        if self.viewer is not None:
            self.viewer.init_tensors(global_tensor_dict)
        return True

    def create_viewer(self, env_manager):
        """
        创建查看器
        :param env_manager: 环境管理器
        """
        self.robot_handles = [ah[0] for ah in self.asset_handles]
        logger.warning(f"Headless: {self.sim_config.viewer.headless}")
        if not self.sim_config.viewer.headless:
            logger.info("Creating viewer")
            self.viewer = IGEViewerControl(
                self.gym, self.sim, env_manager, self.sim_config.viewer, self.device
            )
            self.viewer.set_actor_and_env_handles(self.robot_handles, self.env_handles)
            self.viewer.set_camera_lookat()
            logger.info("Created viewer")
        else:
            logger.info("Headless mode. Viewer not created.")
        return

    def pre_physics_step(self, actions):
        """
        在物理步骤之前执行任何必要的操作
        :param actions: 动作输入
        """
        # 将力和扭矩应用于适当的刚体
        if self.cfg.env.write_to_sim_at_every_timestep:
            self.write_to_sim()  # 将数据写入仿真
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.global_tensor_dict["global_force_tensor"]),
            gymtorch.unwrap_tensor(self.global_tensor_dict["global_torque_tensor"]),
            gymapi.LOCAL_SPACE,
        )
        if self.sim_has_dof:
            self.dof_control_mode = self.global_tensor_dict["dof_control_mode"]

            if self.dof_control_mode == "position":
                self.dof_application_function = self.gym.set_dof_position_target_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_position_setpoint_tensor"]
                )
            elif self.dof_control_mode == "velocity":
                self.dof_application_function = self.gym.set_dof_velocity_target_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_velocity_setpoint_tensor"]
                )
            elif self.dof_control_mode == "effort":
                self.dof_application_function = self.gym.set_dof_actuation_force_tensor
                self.dof_application_tensor = gymtorch.unwrap_tensor(
                    self.global_tensor_dict["dof_effort_tensor"]
                )
            else:
                raise ValueError("Invalid dof control mode")  # 无效的控制模式
            self.dof_application_function(self.sim, self.dof_application_tensor)  # 应用控制
        return

    def physics_step(self):
        """
        执行物理步骤
        """
        self.gym.simulate(self.sim)  # 执行仿真
        self.graphics_are_stepped = False
        return

    def post_physics_step(self):
        """
        在物理步骤之后执行任何必要的操作
        """
        # 更新状态张量
        self.gym.fetch_results(self.sim, True)
        self.refresh_tensors()  # 刷新张量
        return

    def refresh_tensors(self):
        """
        刷新所有张量的状态
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def step_graphics(self):
        """
        处理图形步骤
        """
        if not self.graphics_are_stepped:
            self.gym.step_graphics(self.sim)
            self.graphics_are_stepped = True

    def render_viewer(self):
        """
        渲染查看器
        """
        if self.viewer is not None:
            # 如果查看器不更新，则不浪费时间进行图形步骤
            if not self.graphics_are_stepped and self.viewer.enable_viewer_sync:
                self.step_graphics()
            self.viewer.render()  # 渲染查看器
        return

    def reset(self):
        """
        重置环境
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        """
        重置指定环境的索引
        :param env_ids: 环境ID列表
        """
        self.env_lower_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )[env_ids]
        self.env_upper_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )[env_ids]

    def write_to_sim(self):
        """
        将张量写入仿真
        """
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.global_tensor_dict["unfolded_env_asset_state_tensor"]),
        )
        if self.sim_has_dof:
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.global_tensor_dict["unfolded_dof_state_tensor"]),
            )
        return
