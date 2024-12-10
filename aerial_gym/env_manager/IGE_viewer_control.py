from isaacgym import gymapi
import numpy as np

from aerial_gym.utils.math import quat_from_euler_xyz

import sys
import torch
import time

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_rotate_inverse, quat_rotate

logger = CustomLogger("IGE_viewer_control")


class IGEViewerControl:
    """
    此类用于控制环境的查看器。
    类实例化时需要以下参数：
    - ref_env: 参考环境
    - pos: 相机的位置
    - lookat: 相机正在查看的点（对象或身体）

    此类还提供了控制查看器的方法：
    - set_camera_pos: 设置相机的位置
    - set_camera_lookat: 设置相机查看的点（对象或身体）
    - set_camera_ref_env: 设置参考环境
    """

    def __init__(self, gym, sim, env_manager, config, device):
        self.sim = sim  # 仿真对象
        self.gym = gym  # gym API对象
        self.config = config  # 配置参数
        self.env_manager = env_manager  # 环境管理器
        self.headless = config.headless  # 是否为无头模式
        self.device = device  # 设备

        # 设置相机属性
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True  # 启用张量
        camera_props.width = self.config.width  # 相机宽度
        camera_props.height = self.config.height  # 相机高度
        camera_props.far_plane = self.config.max_range  # 远平面
        camera_props.near_plane = self.config.min_range  # 近平面
        camera_props.horizontal_fov = self.config.horizontal_fov_deg  # 水平视场角
        camera_props.use_collision_geometry = self.config.use_collision_geometry  # 是否使用碰撞几何体
        self.camera_follow_transform_local_offset = torch.tensor(
            self.config.camera_follow_transform_local_offset, device=self.device
        )  # 相机跟随变换的局部偏移
        self.camera_follow_position_global_offset = torch.tensor(
            self.config.camera_follow_position_global_offset, device=self.device
        )  # 相机跟随位置的全局偏移

        # 相机超采样暂时未更改
        self.camera_properties = camera_props

        # 相机的局部变换
        self.local_transform = gymapi.Transform()
        self.local_transform.p = gymapi.Vec3(
            self.config.camera_position[0],
            self.config.camera_position[1],
            self.config.camera_position[2],
        )
        angle_euler = torch.deg2rad(torch.tensor(self.config.camera_orientation_euler_deg))  # 将角度转换为弧度
        angle_quat = quat_from_euler_xyz(angle_euler[0], angle_euler[1], angle_euler[2])  # 从欧拉角创建四元数
        self.local_transform.r = gymapi.Quat(
            angle_quat[0], angle_quat[1], angle_quat[2], angle_quat[3]
        )

        self.lookat = gymapi.Vec3(
            self.config.lookat[0], self.config.lookat[1], self.config.lookat[2]
        )  # 相机查看的目标位置

        if self.config.camera_follow_type == "FOLLOW_TRANSFORM":
            self.camera_follow_type = gymapi.FOLLOW_TRANSFORM  # 跟随变换类型
        elif self.config.camera_follow_type == "FOLLOW_POSITION":
            self.camera_follow_type = gymapi.FOLLOW_POSITION  # 跟随位置类型

        self.cam_handle = None  # 相机句柄

        self.enable_viewer_sync = True  # 启用查看器同步
        self.viewer = None  # 查看器
        self.camera_follow = False  # 相机是否跟随

        self.camera_image_tensor = None  # 相机图像张量

        self.current_target_env = 0  # 当前目标环境索引

        self.sync_frame_time = True  # 是否同步帧时间

        self.pause_sim = False  # 是否暂停仿真

        self.create_viewer()  # 创建查看器

    def set_actor_and_env_handles(self, actor_handles, env_handles):
        """
        设置演员和环境句柄
        """
        self.actor_handles = actor_handles  # 演员句柄
        self.env_handles = env_handles  # 环境句柄

    def init_tensors(self, global_tensor_dict):
        """
        初始化张量
        """
        self.robot_positions = global_tensor_dict["robot_position"]  # 机器人位置张量
        self.robot_vehicle_orientations = global_tensor_dict["robot_orientation"]  # 机器人方向张量

    def create_viewer(self):
        """
        创建查看器的相机传感器。设置相机属性并将其附加到参考环境。
        """
        logger.debug("Creating viewer")
        if self.headless == True:  # 如果是无头模式
            logger.warn("Headless mode enabled. Not creating viewer.")
            return
        # 订阅键盘快捷键
        self.viewer = self.gym.create_viewer(self.sim, self.camera_properties)

        # 设置查看器的事件键绑定。
        # 允许用户使用ESC键退出查看器。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")

        # 允许用户使用V键切换查看器同步。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # 同步帧时间
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "sync_frame_time")

        # 使用F键切换跟随和未附加相机模式。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "toggle_camera_follow")

        # 使用P键切换跟随位置和变换模式。
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "toggle_camera_follow_type"
        )
        # 使用R键重置所有环境。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_all_envs")
        # 使用UP和DOWN键切换目标环境。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "switch_target_env_up")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "switch_target_env_down"
        )
        # 使用SPACE键暂停仿真。
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause_simulation")
        str_instructions = (
            "使用键盘控制查看器的说明:\n"
            + "ESC: 退出\n"
            + "V: 切换查看器同步\n"
            + "S: 同步帧时间\n"
            + "F: 切换相机跟随\n"
            + "P: 切换相机跟随类型\n"
            + "R: 重置所有环境\n"
            + "UP: 向上切换目标环境\n"
            + "DOWN: 向下切换目标环境\n"
            + "SPACE: 暂停仿真\n"
        )
        logger.warning(str_instructions)

        return self.viewer

    def handle_keyboard_events(self):
        """
        处理键盘事件
        """
        # 检查键盘事件
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:  # 处理退出事件
                sys.exit()
            elif evt.action == "reset_all_envs" and evt.value > 0:  # 处理重置所有环境事件
                self.reset_all_envs()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # 处理切换查看器同步事件
                self.toggle_viewer_sync()
            elif evt.action == "toggle_camera_follow" and evt.value > 0:  # 处理切换相机跟随事件
                self.toggle_camera_follow()
            elif evt.action == "toggle_camera_follow_type" and evt.value > 0:  # 处理切换相机跟随类型事件
                self.toggle_camera_follow_type()
            elif evt.action == "switch_target_env_up" and evt.value > 0:  # 处理切换目标环境向上事件
                self.switch_target_env_up()
            elif evt.action == "switch_target_env_down" and evt.value > 0:  # 处理切换目标环境向下事件
                self.switch_target_env_down()
            elif evt.action == "pause_simulation" and evt.value > 0:  # 处理暂停仿真事件
                self.pause_simulation()
            elif evt.action == "sync_frame_time" and evt.value > 0:  # 处理同步帧时间事件
                self.toggle_sync_frame_time()

    def reset_all_envs(self):
        """
        重置所有环境。
        """
        logger.warning("Resetting all environments.")
        self.env_manager.reset()  # 重置环境管理器
        self.env_manager.global_tensor_dict["truncations"][:] = 1  # 设置截断标志

    def toggle_sync_frame_time(self):
        """
        切换同步帧时间。
        """
        self.sync_frame_time = not self.sync_frame_time  # 切换状态
        logger.warning("Sync frame time: {}".format(self.sync_frame_time))

    def get_viewer_image(self):
        """
        获取查看器的图像。
        """
        return self.camera_image_tensor  # 返回相机图像张量

    def toggle_viewer_sync(self):
        """
        切换查看器同步。
        """
        self.enable_viewer_sync = not self.enable_viewer_sync  # 切换状态
        logger.warning("Viewer sync: {}".format(self.enable_viewer_sync))

    def toggle_camera_follow_type(self):
        """
        切换相机跟随模式。
        """
        self.camera_follow_type = (
            gymapi.FOLLOW_TRANSFORM
            if self.camera_follow_type == gymapi.FOLLOW_POSITION
            else gymapi.FOLLOW_POSITION
        )  # 切换跟随类型
        logger.warning("Camera follow type: {}".format(self.camera_follow_type))

    def toggle_camera_follow(self):
        """
        切换相机跟随状态。
        """
        self.camera_follow = not self.camera_follow  # 切换状态
        logger.warning("Camera follow: {}".format(self.camera_follow))
        self.set_camera_lookat()  # 更新相机查看目标

    def switch_target_env_up(self):
        """
        向上切换目标环境。
        """
        self.current_target_env = (self.current_target_env + 1) % len(self.actor_handles)  # 更新目标环境索引
        logger.warning("Switching target environment to: {}".format(self.current_target_env))
        self.set_camera_lookat()  # 更新相机查看目标

    def switch_target_env_down(self):
        """
        向下切换目标环境。
        """
        self.current_target_env = (self.current_target_env - 1) % len(self.actor_handles)  # 更新目标环境索引
        logger.warning("Switching target environment to: {}".format(self.current_target_env))
        self.set_camera_lookat()  # 更新相机查看目标

    def pause_simulation(self):
        """
        暂停仿真。
        """
        self.pause_sim = not self.pause_sim  # 切换暂停状态
        logger.warning(
            "Simulation Paused. You can control the viewer at a reduced rate with full keyboard control."
        )
        while self.pause_sim:  # 当仿真处于暂停状态时
            self.render()  # 渲染查看器
            time.sleep(0.1)  # 等待一段时间
            # self.gym.poll_viewer_events(self.viewer)  # 处理查看器事件
        return

    def set_camera_lookat(self, pos=None, quat_or_target=None):
        """
        设置相机查看目标。
        """
        if pos is None:  # 如果没有提供位置
            pos = self.config.camera_position  # 使用配置中的相机位置
        if quat_or_target is None:  # 如果没有提供查看目标
            quat_or_target = self.config.lookat  # 使用配置中的查看目标
        self.local_transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])  # 设置相机位置
        if self.camera_follow:  # 如果相机跟随
            robot_position = self.robot_positions[self.current_target_env]  # 获取当前目标环境的机器人位置
            robot_vehicle_orientation = self.robot_vehicle_orientations[self.current_target_env]  # 获取当前目标环境的机器人方向
            self.lookat = gymapi.Vec3(robot_position[0], robot_position[1], robot_position[2])  # 设置查看目标为机器人位置
            if self.camera_follow_type == gymapi.FOLLOW_TRANSFORM:  # 根据跟随类型计算相机位置
                viewer_position = robot_position + quat_rotate(
                    robot_vehicle_orientation.unsqueeze(0),
                    self.camera_follow_transform_local_offset.unsqueeze(0),
                ).squeeze(0)
            else:
                viewer_position = robot_position + self.camera_follow_position_global_offset  # 使用全局偏移计算相机位置

            self.local_transform.p = gymapi.Vec3(
                viewer_position[0], viewer_position[1], viewer_position[2]
            )
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env_handles[self.current_target_env],
                self.local_transform.p,
                self.lookat,
            )  # 设置查看器摄像机位置和目标
        if self.camera_follow == False:  # 如果不跟随
            target_pos = quat_or_target  # 使用提供的目标位置
            self.lookat = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])  # 更新查看目标
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env_handles[self.current_target_env],
                self.local_transform.p,
                self.lookat,
            )  # 设置查看器摄像机位置和目标

    def render(self):
        """
        绘制查看器。
        """
        if self.gym.query_viewer_has_closed(self.viewer):  # 检查查看器是否被关闭
            logger.critical("Viewer has been closed. Exiting simulation.")
            sys.exit()  # 退出仿真
        self.handle_keyboard_events()  # 处理键盘事件
        if self.enable_viewer_sync:  # 如果启用查看器同步
            if self.camera_follow:  # 如果相机跟随
                self.set_camera_lookat()  # 更新相机查看目标
            self.gym.draw_viewer(self.viewer, self.sim, False)  # 绘制查看器
            if self.sync_frame_time:  # 如果同步帧时间
                self.gym.sync_frame_time(self.sim)  # 同步帧时间
        else:
            self.gym.poll_viewer_events(self.viewer)  # 处理查看器事件
