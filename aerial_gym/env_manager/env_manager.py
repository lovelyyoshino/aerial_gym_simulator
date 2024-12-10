from aerial_gym.env_manager.IGE_env_manager import IsaacGymEnv

# 从aerial_gym的环境管理模块中导入基本管理器
from aerial_gym.env_manager.base_env_manager import BaseManager  
# 从aerial_gym的环境管理模块中导入资产管理器
from aerial_gym.env_manager.asset_manager import AssetManager  
# 从aerial_gym的环境管理模块中导入扭曲环境管理器
from aerial_gym.env_manager.warp_env_manager import WarpEnv  
# 从aerial_gym的环境管理模块中导入资产加载器
from aerial_gym.env_manager.asset_loader import AssetLoader  

from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.env_manager.obstacle_manager import ObstacleManager

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.robot_registry import robot_registry

import torch

from aerial_gym.utils.logging import CustomLogger

import math, random

logger = CustomLogger("env_manager")


class EnvManager(BaseManager):
    """
    该类管理环境。可以处理机器人的创建、环境和资产管理器。该类处理张量的创建和销毁。

    此外，环境管理器可以在主环境类中调用，通过抽象接口来操控环境。

    此脚本可以尽可能保持通用，以处理不同类型的环境，而针对特定情况的更改可以在各个机器人或环境管理器中进行。
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        # 初始化环境管理器的基本信息
        self.robot_name = robot_name  # 机器人名称
        self.controller_name = controller_name  # 控制器名称
        self.sim_config = sim_config_registry.make_sim(sim_name)  # 创建仿真配置

        # 调用父类的构造函数
        super().__init__(env_config_registry.make_env(env_name), device)

        # 可选参数配置
        if num_envs is not None:
            self.cfg.env.num_envs = num_envs  # 环境数量
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp  # 是否使用warp
        if headless is not None:
            self.sim_config.viewer.headless = headless  # 是否无头模式

        self.num_envs = self.cfg.env.num_envs  # 存储环境数量
        self.use_warp = self.cfg.env.use_warp  # 使用warp的标志

        self.asset_manager = None  # 资产管理器
        self.tensor_manager = None  # 张量管理器
        self.env_args = args  # 环境参数

        self.keep_in_env = None  # 保持在环境中的资产数量

        self.global_tensor_dict = {}  # 全局张量字典

        logger.info("Populating environments.")
        self.populate_env(env_cfg=self.cfg, sim_cfg=self.sim_config)  # 填充环境
        logger.info("[DONE] Populating environments.")
        self.prepare_sim()  # 准备仿真

        # 初始化仿真步骤张量
        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def create_sim(self, env_cfg, sim_cfg):
        """
        该函数创建环境和机器人管理器。为IsaacGym环境实例创建环境所需的必要内容。
        """
        logger.info("Creating simulation instance.")
        logger.info("Instantiating IGE object.")

        # === 需要在这里检查，否则IGE在不同CUDA GPU上会崩溃 ====
        has_IGE_cameras = False
        robot_config = robot_registry.get_robot_config(self.robot_name)  # 获取机器人配置
        if robot_config.sensor_config.enable_camera == True and self.use_warp == False:
            has_IGE_cameras = True  # 如果启用相机且未使用warp，则设置标志
        # ===============================================================================================

        # 创建IsaacGym环境实例
        self.IGE_env = IsaacGymEnv(env_cfg, sim_cfg, has_IGE_cameras, self.device)

        # 定义一个全局字典以存储在环境、资产和机器人管理器之间共享的仿真对象和重要参数
        self.global_sim_dict = {}
        self.global_sim_dict["gym"] = self.IGE_env.gym
        self.global_sim_dict["sim"] = self.IGE_env.sim
        self.global_sim_dict["env_cfg"] = self.cfg
        self.global_sim_dict["use_warp"] = self.IGE_env.cfg.env.use_warp
        self.global_sim_dict["num_envs"] = self.IGE_env.cfg.env.num_envs
        self.global_sim_dict["sim_cfg"] = sim_cfg

        logger.info("IGE object instantiated.")

        # 如果使用warp，则创建warp环境
        if self.cfg.env.use_warp:
            logger.info("Creating warp environment.")
            self.warp_env = WarpEnv(self.global_sim_dict, self.device)
            logger.info("Warp environment created.")

        # 创建资产加载器
        self.asset_loader = AssetLoader(self.global_sim_dict, self.device)

        logger.info("Creating robot manager.")
        # 创建机器人管理器，会调用robots下面的机器人管理模块
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg  # 存储机器人配置
        logger.info("[DONE] Creating robot manager.")

        logger.info("[DONE] Creating simulation instance.")

    def populate_env(self, env_cfg, sim_cfg):
        """
        该函数用必要的资产和机器人填充环境。
        """
        # 使用环境和机器人管理器创建仿真实例
        self.create_sim(env_cfg, sim_cfg)

        # 创建机器人
        self.robot_manager.create_robot(self.asset_loader)

        # 首先为环境选择资产:
        self.global_asset_dicts, keep_in_env_num = self.asset_loader.select_assets_for_sim()

        # 检查保持在环境中的资产数量是否一致
        if self.keep_in_env is None:
            self.keep_in_env = keep_in_env_num
        elif self.keep_in_env != keep_in_env_num:
            raise Exception(
                "Inconsistent number of assets kept in the environment. The number of keep_in_env assets must be equal for all environments. Check."
            )

        # 将资产添加到环境中
        segmentation_ctr = 100  # 分割计数器

        self.global_asset_counter = 0  # 全局资产计数器
        self.step_counter = 0  # 步数计数器

        self.asset_min_state_ratio = None  # 资产最小状态比率
        self.asset_max_state_ratio = None  # 资产最大状态比率

        # 初始化碰撞和截断张量
        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions  # 环境动作数量
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]  # 碰撞张量
        self.truncation_tensor = self.global_tensor_dict["truncations"]  # 截断张量

        # 在填充环境之前，需要创建地面
        if self.cfg.env.create_ground_plane:
            logger.info("Creating ground plane in Isaac Gym Simulation.")
            self.IGE_env.create_ground_plane()  # 创建地面
            logger.info("[DONE] Creating ground plane in Isaac Gym Simulation")

        # 为每个环境填充资产
        for i in range(self.cfg.env.num_envs):
            logger.debug(f"Populating environment {i}")
            if i % 1000 == 0:
                logger.info(f"Populating environment {i}")

            env_handle = self.IGE_env.create_env(i)  # 创建环境句柄
            if self.cfg.env.use_warp:
                self.warp_env.create_env(i)  # 创建warp环境句柄

            # 将机器人资产添加到环境中
            self.robot_manager.add_robot_to_env(
                self.IGE_env, env_handle, self.global_asset_counter, i, segmentation_ctr
            )
            self.global_asset_counter += 1  # 更新资产计数器

            self.num_obs_in_env = 0  # 该环境中的观察数量
            # 将常规资产添加到环境中
            for asset_info_dict in self.global_asset_dicts[i]:
                asset_handle, ige_seg_ctr = self.IGE_env.add_asset_to_env(
                    asset_info_dict,
                    env_handle,
                    i,
                    self.global_asset_counter,
                    segmentation_ctr,
                )
                self.num_obs_in_env += 1  # 更新观察数量
                warp_segmentation_ctr = 0
                if self.cfg.env.use_warp:
                    empty_handle, warp_segmentation_ctr = self.warp_env.add_asset_to_env(
                        asset_info_dict,
                        i,
                        self.global_asset_counter,
                        segmentation_ctr,
                    )
                # 在WARP中添加后更新此值
                self.global_asset_counter += 1
                segmentation_ctr += max(ige_seg_ctr, warp_segmentation_ctr)  # 更新分割计数器
                # 更新资产状态比率
                if self.asset_min_state_ratio is None or self.asset_max_state_ratio is None:
                    self.asset_min_state_ratio = torch.tensor(
                        asset_info_dict["min_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                    self.asset_max_state_ratio = torch.tensor(
                        asset_info_dict["max_state_ratio"], requires_grad=False
                    ).unsqueeze(0)
                else:
                    self.asset_min_state_ratio = torch.vstack(
                        (
                            self.asset_min_state_ratio,
                            torch.tensor(asset_info_dict["min_state_ratio"], requires_grad=False),
                        )
                    )
                    self.asset_max_state_ratio = torch.vstack(
                        (
                            self.asset_max_state_ratio,
                            torch.tensor(asset_info_dict["max_state_ratio"], requires_grad=False),
                        )
                    )

        # 检查环境是否有0个对象。如果是，则跳过此步骤
        if self.asset_min_state_ratio is not None:
            self.asset_min_state_ratio = self.asset_min_state_ratio.to(self.device)
            self.asset_max_state_ratio = self.asset_max_state_ratio.to(self.device)
            self.global_tensor_dict["asset_min_state_ratio"] = self.asset_min_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
            self.global_tensor_dict["asset_max_state_ratio"] = self.asset_max_state_ratio.view(
                self.cfg.env.num_envs, -1, 13
            )
        else:
            self.global_tensor_dict["asset_min_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )
            self.global_tensor_dict["asset_max_state_ratio"] = torch.zeros(
                (self.cfg.env.num_envs, 0, 13), device=self.device
            )

        self.global_tensor_dict["num_obstacles_in_env"] = self.num_obs_in_env  # 该环境中的障碍物数量

    def prepare_sim(self):
        """
        该函数为环境准备仿真。
        """
        # 准备仿真环境
        if not self.IGE_env.prepare_for_simulation(self, self.global_tensor_dict):
            raise Exception("Failed to prepare the simulation")
        if self.cfg.env.use_warp:
            if not self.warp_env.prepare_for_simulation(self.global_tensor_dict):
                raise Exception("Failed to prepare the simulation")

        # 创建资产管理器
        self.asset_manager = AssetManager(self.global_tensor_dict, self.keep_in_env)
        self.asset_manager.prepare_for_sim()  # 准备资产管理器

        # 准备机器人管理器
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        self.obstacle_manager = ObstacleManager(
            self.IGE_env.num_assets_per_env, self.cfg, self.device
        )
        self.obstacle_manager.prepare_for_sim(self.global_tensor_dict)  # 准备障碍物管理器
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]  # 机器人动作数量

    def reset_idx(self, env_ids=None):
        """
        该函数重置给定环境索引的环境。
        """
        # 首先重置Isaac Gym环境，因为它决定了环境的边界
        # 然后重置资产管理器，重新定位环境中的资产
        # 如果使用warp，则重置warp环境，该环境从资产中读取状态张量并变换网格
        # 最后重置机器人管理器，重置机器人状态张量和传感器
        logger.debug(f"Resetting environments {env_ids}.")
        self.IGE_env.reset_idx(env_ids)
        self.asset_manager.reset_idx(env_ids, self.global_tensor_dict["num_obstacles_in_env"])
        if self.cfg.env.use_warp:
            self.warp_env.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        self.IGE_env.write_to_sim()  # 将状态写入仿真
        self.sim_steps[env_ids] = 0  # 重置仿真步骤

    def log_memory_use(self):
        """
        该函数记录GPU的内存使用情况。
        """
        logger.warning(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0)/1024/1024/1024}GB"
        )
        logger.warning(
            f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0)/1024/1024/1024}GB"
        )

        # 计算此类对象使用的系统RAM
        total_memory = 0
        for key, value in self.__dict__.items():
            total_memory += value.__sizeof__()
        logger.warning(
            f"Total memory used by the objects of this class: {total_memory/1024/1024}MB"
        )

    def reset(self):
        # 重置所有环境
        self.reset_idx(env_ids=torch.arange(self.cfg.env.num_envs))

    def pre_physics_step(self, actions, env_actions):
        # 首先让机器人计算动作
        self.robot_manager.pre_physics_step(actions)
        # 然后资产管理器在这里应用动作
        self.asset_manager.pre_physics_step(env_actions)
        # 将动作应用于障碍物管理器
        self.obstacle_manager.pre_physics_step(env_actions)
        # 然后仿真器在这里应用它们
        self.IGE_env.pre_physics_step(actions)
        # 如果使用warp，则在这里应用warp环境的动作
        # 如果更改网格，则需要调用refit()（开销大）。
        if self.use_warp:
            self.warp_env.pre_physics_step(actions)

    def reset_tensors(self):
        # 重置碰撞和截断张量
        self.collision_tensor[:] = 0
        self.truncation_tensor[:] = 0

    def simulate(self, actions, env_actions):
        # 执行仿真步骤
        self.pre_physics_step(actions, env_actions)  # 预物理步骤
        self.IGE_env.physics_step()  # 物理步骤
        self.post_physics_step(actions, env_actions)  # 后物理步骤

    def post_physics_step(self, actions, env_actions):
        # 执行后物理步骤
        self.IGE_env.post_physics_step()
        self.robot_manager.post_physics_step()  # 机器人后处理
        if self.use_warp:
            self.warp_env.post_physics_step()  # warp环境后处理
        self.asset_manager.post_physics_step()  # 资产后处理

    def compute_observations(self):
        # 计算观察结果
        self.collision_tensor[:] += (
            torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
            > self.cfg.env.collision_force_threshold  # 碰撞力超过阈值则记录碰撞
        )

    def reset_terminated_and_truncated_envs(self):
        # 重置结束和截断的环境
        collision_envs = self.collision_tensor.nonzero(as_tuple=False).squeeze(-1)  # 碰撞的环境
        truncation_envs = self.truncation_tensor.nonzero(as_tuple=False).squeeze(-1)  # 截断的环境
        envs_to_reset = (
            (self.collision_tensor * int(self.cfg.env.reset_on_collision) + self.truncation_tensor)
            .nonzero(as_tuple=False)
            .squeeze(-1)  # 需要重置的环境
        )
        # 重置发生碰撞的环境
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)  # 重置环境
        return envs_to_reset  # 返回需要重置的环境

    def render(self, render_components="sensors"):
        # 渲染环境
        if render_components == "viewer":
            self.render_viewer()  # 渲染查看器
        elif render_components == "sensors":
            self.render_sensors()  # 渲染传感器

    def render_sensors(self):
        # 在物理步骤后渲染传感器
        if self.robot_manager.has_IGE_sensors:
            self.IGE_env.step_graphics()  # 更新图形
        self.robot_manager.capture_sensors()  # 捕获传感器数据

    def render_viewer(self):
        # 渲染查看器GUI
        self.IGE_env.render_viewer()

    def post_reward_calculation_step(self):
        # 在计算奖励后步骤进行操作
        envs_to_reset = self.reset_terminated_and_truncated_envs()  # 重置结束和截断的环境
        # 在重置后渲染，以确保传感器从新机器人状态中更新
        self.render(render_components="sensors")
        return envs_to_reset  # 返回需要重置的环境

    def step(self, actions, env_actions=None):
        """
        该函数为环境步骤仿真。
        actions: 发送给机器人的动作。
        env_actions: 发送给环境实体的动作。
        """
        self.reset_tensors()  # 重置张量
        if env_actions is not None:
            if self.global_tensor_dict["env_actions"] is None:
                # 初始化环境动作
                self.global_tensor_dict["env_actions"] = env_actions
                self.global_tensor_dict["prev_env_actions"] = env_actions
                self.prev_env_actions = self.global_tensor_dict["prev_env_actions"]
                self.env_actions = self.global_tensor_dict["env_actions"]
            logger.warning(
                f"Env actions shape: {env_actions.shape}, Previous env actions shape: {self.env_actions.shape}"
            )
            self.prev_env_actions[:] = self.env_actions  # 更新上一个环境动作
            self.env_actions[:] = env_actions  # 更新当前环境动作
        
        # 计算每个环境步骤的物理步骤数量
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        # 执行物理步骤
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)  # 仿真
            self.compute_observations()  # 计算观察结果
        self.sim_steps[:] = self.sim_steps[:] + 1  # 更新仿真步骤
        self.step_counter += 1  # 更新步骤计数器
        if self.step_counter % self.cfg.env.render_viewer_every_n_steps == 0:
            self.render(render_components="viewer")  # 渲染查看器

    def get_obs(self):
        # 仅返回所有张量的字典。任务所需的任何内容可以用于计算奖励。
        return self.global_tensor_dict  # 返回全局张量字典
