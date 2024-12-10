from isaacgym import gymapi
from isaacgym import gymtorch
import os

from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.registry.robot_registry import robot_registry
import torch

# 获取所有的传感器类
from aerial_gym.sensors.isaacgym_camera_sensor import IsaacGymCameraSensor
from aerial_gym.sensors.warp.warp_sensor import WarpSensor
from aerial_gym.sensors.imu_sensor import IMUSensor

from aerial_gym.utils.logging import CustomLogger
import pytorch3d.transforms as p3d_transforms

logger = CustomLogger("robot_manager")

class RobotManagerIGE(BaseManager):
    def __init__(self, global_sim_dict, robot_name, controller_name, device):
        # 初始化RobotManagerIGE
        logger.debug("Initializing RobotManagerIGE")
        self.gym = global_sim_dict["gym"]  # 获取gym实例
        self.sim = global_sim_dict["sim"]  # 获取sim实例
        self.env_config = global_sim_dict["env_cfg"]  # 环境配置
        self.use_warp = global_sim_dict["use_warp"]  # 是否使用warp
        self.num_envs = global_sim_dict["num_envs"]  # 环境数量
        
        # 从机器人注册表创建机器人，并使用该机器人的配置
        self.robot, robot_config = robot_registry.make_robot(
            robot_name, controller_name, self.env_config, device
        )

        # 超级类初始化，机器人注册表告诉我们机器人配置
        super().__init__(robot_config, device)

        self.robot_handles = []  # 机器人句柄列表

        self.camera_sensor = None
        self.warp_sensor = None
        self.lidar_sensor = None
        self.imu_sensor = None
        self.has_IGE_sensors = False

        self.robot_inertia = None
        self.robot_mass = None
        self.robot_masses = torch.zeros(self.num_envs, device=self.device)  # 初始化机器人质量
        self.robot_inertias = torch.zeros((self.num_envs, 3, 3), device=self.device)  # 初始化机器人惯性

        self.dof_control_mode = "none"  # 初始化自由度控制模式

        # 根据传感器配置初始化相应的传感器（给机器人创建传感器）
        if self.use_warp == False:
            if self.cfg.sensor_config.enable_camera:
                logger.debug("Initializing Isaac Gym camera sensor")
                self.camera_sensor = IsaacGymCameraSensor(
                    self.cfg.sensor_config.camera_config,
                    self.num_envs,
                    self.gym,
                    self.sim,
                    self.device,
                )
                logger.debug("[DONE] Initializing Isaac Gym camera sensor")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )
        elif self.use_warp == True and (
            self.cfg.sensor_config.enable_camera and self.cfg.sensor_config.enable_lidar
        ):
            logger.warning(
                "Warp is enabled. Appropriate camera sensors will be spawned using warp."
            )
            logger.error(
                "This error is here because you have enabled both camera and lidar sensors with warp."
            )
            logger.error(
                "There is no reason for the simulation to kill itself really, but both have not been extensively tested together."
            )
            logger.error(
                "if you really need to use both, just comment out the exception here and these lines and things should mostly work okay :) "
            )
            logger.error(
                "You might need to declare another tensor for sensor data for the other sensor though because they currently use the same tensor."
            )
            raise ValueError(
                "Both camera and lidar are enabled. But there is no reason for this error other than preventing undesired behaviors. Just comment out this error line and things should be okay."
            )

        logger.debug("[DONE] Initializing RobotManagerIGE")

        return

    def create_robot(self, asset_loader_class):
        # 从名称注册表创建机器人资产，并使用已创建机器人的配置
        logger.debug("Creating robot asset for Isaac Gym")
        robot_asset_class = self.cfg.robot_asset
        self.robot_asset_dict = asset_loader_class.load_selected_file_from_config(
            "robot", robot_asset_class, robot_asset_class.file, is_robot=True
        )
        logger.debug("[DONE] Creating robot asset for Isaac Gym")

        return

    def prepare_for_sim(self, global_tensor_dict):
        # 准备模拟环境
        self.global_tensor_dict = global_tensor_dict

        self.global_tensor_dict["robot_mass"] = self.robot_masses  # 设置机器人质量
        self.global_tensor_dict["robot_inertia"] = self.robot_inertias  # 设置机器人惯性

        self.global_tensor_dict["robot_actions"] = torch.zeros(
            (self.num_envs, self.robot.num_actions), device=self.device
        )  # 初始化机器人动作张量

        self.global_tensor_dict["robot_prev_actions"] = torch.zeros_like(
            self.global_tensor_dict["robot_actions"]
        )  # 初始化上一个动作张量

        self.actions = self.global_tensor_dict["robot_actions"]  # 当前动作
        self.prev_actions = self.global_tensor_dict["robot_prev_actions"]  # 上一个动作

        self.global_tensor_dict["dof_control_mode"] = self.dof_control_mode  # 设置自由度控制模式

        self.robot.init_tensors(self.global_tensor_dict)  # 初始化机器人张量

        # 如果不使用warp，初始化传感器
        if not self.use_warp:
            logger.error("Not using warp. Initializing sensors")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )

            if self.cfg.sensor_config.enable_camera:
                self.image_tensor = torch.zeros(
                    (
                        self.num_envs,
                        self.cfg.sensor_config.camera_config.num_sensors,
                        self.cfg.sensor_config.camera_config.height,
                        self.cfg.sensor_config.camera_config.width,
                    ),
                    device=self.device,
                    requires_grad=False,
                )
                self.global_tensor_dict["depth_range_pixels"] = self.image_tensor  # 深度范围像素

                if self.cfg.sensor_config.camera_config.segmentation_camera:
                    self.segmentation_tensor = torch.zeros(
                        (
                            self.num_envs,
                            self.cfg.sensor_config.camera_config.num_sensors,
                            self.cfg.sensor_config.camera_config.height,
                            self.cfg.sensor_config.camera_config.width,
                        ),
                        dtype=torch.int32,
                        device=self.device,
                        requires_grad=False,
                    )
                    self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor  # 分割像素
                    logger.critical(
                        f"Segmentation pixels shape: {self.global_tensor_dict['segmentation_pixels'].shape}"
                    )
                logger.critical(
                    f"Depth range pixels shape: {self.global_tensor_dict['depth_range_pixels'].shape}"
                )

                self.camera_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)  # 初始化相机传感器
        else:
            # 确保同时只使用相机或激光雷达
            assert not (
                self.cfg.sensor_config.enable_camera and self.cfg.sensor_config.enable_lidar
            ), "Do not use both camera and lidar sensors together for now."

            self.warp_sensor_config = None
            if self.cfg.sensor_config.enable_camera:
                self.warp_sensor_config = self.cfg.sensor_config.camera_config
                self.warp_sensor_class = WarpSensor
            elif self.cfg.sensor_config.enable_lidar:
                self.warp_sensor_config = self.cfg.sensor_config.lidar_config
                self.warp_sensor_class = WarpSensor

            if self.warp_sensor_config is not None:
                logger.debug("Initializing warp sensor")
                # 在准备传感器张量之前准备模拟的张量
                image_tensor_dims = 3 * (self.warp_sensor_config.return_pointcloud == True)
                if self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] is None:
                    logger.critical(
                        "Warp camera is enabled but there is nothing in the environment. No rendering will take place and the camera tensor will not be populated."
                    )
                else:
                    if image_tensor_dims == 0:
                        self.image_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    else:
                        self.image_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                                image_tensor_dims,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    self.global_tensor_dict["depth_range_pixels"] = self.image_tensor

                    if self.warp_sensor_config.segmentation_camera:
                        self.segmentation_tensor = torch.zeros(
                            (
                                self.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            dtype=torch.int32,
                            device=self.device,
                            requires_grad=False,
                        )
                        self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor
                    self.warp_sensor = self.warp_sensor_class(
                        self.warp_sensor_config,
                        self.num_envs,
                        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"],
                        self.device,
                    )
                    self.warp_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)  # 初始化warp传感器
                    logger.debug("[DONE] Initializing warp sensor")
                    logger.debug("Capturing warp sensor")
                    self.warp_sensor.update()  # 更新warp传感器
                    logger.debug("[DONE] Capturing warp sensor")

        if self.cfg.sensor_config.enable_imu:
            logger.debug("Initializing IMU sensor")
            # 获取每个资产的力张量
            self.force_sensor_tensor = gymtorch.wrap_tensor(
                self.gym.acquire_force_sensor_tensor(self.sim)
            )
            self.global_tensor_dict["force_sensor_tensor"] = self.force_sensor_tensor  # 力传感器张量

            self.imu_sensor = IMUSensor(
                self.cfg.sensor_config.imu_config, self.num_envs, self.device
            )
            self.imu_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)  # 初始化IMU传感器
            logger.debug("[DONE] Initializing IMU sensor")

        elif self.use_warp == False and self.camera_sensor is not None:
            self.has_IGE_sensors = True  # 记录是否有IGE传感器
        return

    def add_robot_to_env(
        self,
        simulation_env_class,
        env_handle,
        global_asset_counter,
        env_id,
        segmentation_counter,
    ):
        # 将机器人添加到环境中
        self.actor_handle, _ = simulation_env_class.add_asset_to_env(
            self.robot_asset_dict,
            env_handle,
            env_id,
            global_asset_counter,
            segmentation_counter,
        )
        self.robot_handles.append(self.actor_handle)  # 保存机器人句柄
        # 当前机器人没有分割ID，如有需要可更改
        if self.camera_sensor is not None:
            for i in range(self.camera_sensor.cfg.num_sensors):
                self.camera_sensor.add_sensor_to_env(env_id, env_handle, self.actor_handle)  # 为环境添加相机传感器

        if self.robot_inertia is None or self.robot_mass is None:
            # 获取机器人质量和惯性
            rbp = self.gym.get_actor_rigid_body_properties(env_handle, self.actor_handle)
            self.robot_mass = 0.0
            self.robot_inertia = torch.zeros((3, 3), device=self.device)  # 初始化机器人惯性
            body_inertia = torch.zeros((3, 3), device=self.device)  # 初始化身体惯性
            state_list = self.gym.get_actor_rigid_body_states(
                env_handle, self.actor_handle, gymapi.STATE_ALL
            )
            item_ctr = 0

            quat = torch.zeros((1, 4), dtype=torch.float32, device=self.device)  # 初始化四元数
            com = torch.zeros((1, 4), device=self.device)  # 初始化质心
            transformation_mat = torch.zeros((4, 4), device=self.device)  # 初始化变换矩阵

            robot_com = torch.zeros((1, 4), device=self.device)  # 初始化机器人质心
            robot_mass = 0.0  # 初始化机器人质量

            for item, properties in zip(state_list, rbp):
                obj_com = torch.zeros((1, 4), device=self.device)  # 初始化物体质心
                obj_mass = properties.mass  # 获取物体质量
                obj_com[0, 0] = properties.com.x
                obj_com[0, 1] = properties.com.y
                obj_com[0, 2] = properties.com.z
                obj_com[0, 3] = 1.0

                position = item[0][0]  # 物体位置
                rotation = item[0][1]  # 物体旋转

                quat[0, 0] = float(rotation[0])  # 四元数赋值
                quat[0, 1] = float(rotation[1])
                quat[0, 2] = float(rotation[2])
                quat[0, 3] = float(rotation[3])

                # 将四元数转换为旋转矩阵
                rotmat = p3d_transforms.quaternion_to_matrix(quat[:, [3, 0, 1, 2]])[0]

                transformation_mat[0:3, 0:3] = rotmat  # 更新变换矩阵
                transformation_mat[0, 3] = float(position[0])
                transformation_mat[1, 3] = float(position[1])
                transformation_mat[2, 3] = float(position[2])
                transformation_mat[3, 3] = float(1.0)

                obj_com_in_root_link_frame = torch.matmul(transformation_mat, obj_com.T).T  # 物体在根链接框架中的质心
                logger.debug(f"Obj COM: {obj_com_in_root_link_frame}, Robot mass: {obj_mass}")

                robot_com += obj_mass * obj_com_in_root_link_frame  # 计算机器人质心
                robot_mass += obj_mass  # 累加质量
            robot_com /= robot_mass  # 计算机器人质心
            robot_com[0, 3] = 1.0  # 设置齐次坐标

            logger.debug(f"Robot COM: {robot_com}, Robot mass: {robot_mass}")

            for item, properties in zip(state_list, rbp):
                position = item[0][0]
                rotation = item[0][1]
                logger.debug(f"Item: {item_ctr} position: {position}, rotation: {rotation}")

                com[0, 0] = properties.com.x
                com[0, 1] = properties.com.y
                com[0, 2] = properties.com.z
                com[0, 3] = 1.0

                body_inertia[0, 0] = properties.inertia.x.x
                body_inertia[0, 1] = properties.inertia.x.y
                body_inertia[0, 2] = properties.inertia.x.z
                body_inertia[1, 0] = properties.inertia.y.x
                body_inertia[1, 1] = properties.inertia.y.y
                body_inertia[1, 2] = properties.inertia.y.z
                body_inertia[2, 0] = properties.inertia.z.x
                body_inertia[2, 1] = properties.inertia.z.y
                body_inertia[2, 2] = properties.inertia.z.z

                quat[0, 0] = float(rotation[0])
                quat[0, 1] = float(rotation[1])
                quat[0, 2] = float(rotation[2])
                quat[0, 3] = float(rotation[3])

                # 将四元数转换为旋转矩阵
                rotmat = p3d_transforms.quaternion_to_matrix(quat[:, [3, 0, 1, 2]])[0]

                transformed_inertia = torch.matmul(rotmat, torch.matmul(body_inertia, rotmat.T))  # 转换惯性
                logger.debug(
                    f"intial inertia: {body_inertia.view(1, 9)} \n transformed_inertia: {transformed_inertia.view(1, 9)}"
                )

                transformation_mat[0:3, 0:3] = rotmat
                transformation_mat[0, 3] = float(position[0])
                transformation_mat[1, 3] = float(position[1])
                transformation_mat[2, 3] = float(position[2])
                transformation_mat[3, 3] = float(1.0)

                com_in_root_link_frame = torch.matmul(transformation_mat, com.T).squeeze(1)  # 物体在根链接框架中的质心

                obj_com_in_robot_com_frame = -(
                    com_in_root_link_frame - robot_com.T.squeeze(1)
                )  # 计算物体在机器人质心框架中的质心
                obj_com_in_robot_com_frame[3] = 1.0

                logger.debug(f"COM in root link frame: {com_in_root_link_frame}")
                logger.debug(f"COM in robot COM frame: {obj_com_in_robot_com_frame}")

                # 使用平行轴定理计算根链接框架中的惯性
                transformed_inertia[0, 0] += properties.mass * (
                    obj_com_in_robot_com_frame[1] ** 2 + obj_com_in_robot_com_frame[2] ** 2
                )
                transformed_inertia[1, 1] += properties.mass * (
                    obj_com_in_robot_com_frame[0] ** 2 + obj_com_in_robot_com_frame[2] ** 2
                )
                transformed_inertia[2, 2] += properties.mass * (
                    obj_com_in_robot_com_frame[0] ** 2 + obj_com_in_robot_com_frame[1] ** 2
                )
                transformed_inertia[0, 1] += -(
                    properties.mass * obj_com_in_robot_com_frame[0] * obj_com_in_robot_com_frame[1]
                )
                transformed_inertia[0, 2] += -(
                    properties.mass * obj_com_in_robot_com_frame[0] * obj_com_in_robot_com_frame[2]
                )
                transformed_inertia[1, 2] += -(
                    properties.mass * obj_com_in_robot_com_frame[1] * obj_com_in_robot_com_frame[2]
                )
                transformed_inertia[1, 0] = transformed_inertia[0, 1]
                transformed_inertia[2, 0] = transformed_inertia[0, 2]
                transformed_inertia[2, 1] = transformed_inertia[1, 2]

                self.robot_mass += properties.mass  # 累加总质量
                self.robot_inertia += transformed_inertia  # 累加总惯性
                item_ctr += 1
            logger.warning(
                f"\nRobot mass: {self.robot_mass},\nInertia: {self.robot_inertia},\nRobot COM: {robot_com}"
            )
            logger.warning(
                "Calculated robot mass and inertia for this robot. This code assumes that your robot is the same across environments."
            )
            logger.critical(
                "If your robot differs across environments you need to perform this computation for each different robot here."
            )
        else:
            logger.debug(
                "It's the same robot as before. Not calculating the inertia and mass again. Change this if your robot differs across envs."
            )

        # 设置机器人的驱动模式以进行DOF控制
        props = self.gym.get_actor_dof_properties(env_handle, self.actor_handle)
        try:
            if len(props["driveMode"]) > 0:
                if self.cfg.reconfiguration_config.dof_mode == "position":
                    props["driveMode"].fill(gymapi.DOF_MODE_POS)
                    for j_index in range(len(props["stiffness"])):
                        props["stiffness"][j_index] = self.cfg.reconfiguration_config.stiffness[
                            j_index
                        ]
                elif self.cfg.reconfiguration_config.dof_mode == "velocity":
                    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
                    for j_index in range(len(props["damping"])):
                        props["damping"][j_index] = self.cfg.reconfiguration_config.damping[j_index]
                elif self.cfg.reconfiguration_config.dof_mode == "effort":
                    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                else:
                    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                self.dof_control_mode = self.cfg.reconfiguration_config.dof_mode  # 更新自由度控制模式
                self.gym.set_actor_dof_properties(env_handle, self.actor_handle, props)
        except Exception as e:
            logger.error(
                f"Something unexpected happened while setting parameters for the DOF modes of the robot. Please check if the correct reconfiguration_config params are set in the robot config file."
            )
            raise e

        self.robot_masses[env_id] = self.robot_mass  # 保存机器人质量
        self.robot_inertias[env_id] = self.robot_inertia  # 保存机器人惯性
        return segmentation_counter + 1  # 返回下一个分割计数

    def reset(self):
        # 重置机器人状态
        self.reset_idx(torch.arange(self.cfg.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        # 重置指定环境的机器人状态
        self.robot.reset_idx(env_ids)
        if self.warp_sensor is not None:
            self.warp_sensor.reset_idx(env_ids)  # 重置warp传感器
        if self.imu_sensor is not None:
            self.imu_sensor.reset_idx(env_ids)  # 重置IMU传感器
        if self.camera_sensor is not None:
            self.camera_sensor.reset_idx(env_ids)  # 重置相机传感器

    def pre_physics_step(self, actions):
        # 在物理步骤之前更新动作
        self.prev_actions[:] = self.actions[:]  # 保存上一个动作
        self.actions[:] = actions  # 更新当前动作
        self.robot.step(self.actions)  # 机器人执行当前动作

    def post_physics_step(self):
        # 在物理步骤之后更新传感器
        if self.imu_sensor is not None:
            self.imu_sensor.update()  # 更新IMU传感器

    def capture_sensors(self):
        # 捕获传感器数据
        if self.warp_sensor is not None:
            self.warp_sensor.update()  # 更新warp传感器
        if self.camera_sensor is not None:
            self.camera_sensor.update()  # 更新相机传感器
