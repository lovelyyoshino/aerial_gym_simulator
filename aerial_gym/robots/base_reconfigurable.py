from aerial_gym.robots.base_multirotor import BaseMultirotor
import torch

from aerial_gym.utils.math import torch_rand_float_tensor, pd_control

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("reconfigurable_robot_class")


class BaseReconfigurable(BaseMultirotor):
    def __init__(self, robot_config, controller_name, env_config, device):
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )

        # 初始化关节最小状态
        self.joint_init_state_min = torch.tensor(
            self.cfg.reconfiguration_config.init_state_min, device=self.device, dtype=torch.float32
        ).T.expand(self.num_envs, -1, -1)

        # 初始化关节最大状态
        self.joint_init_state_max = torch.tensor(
            self.cfg.reconfiguration_config.init_state_max, device=self.device, dtype=torch.float32
        ).T.expand(self.num_envs, -1, -1)

        # 初始化关节响应参数
        self.init_joint_response_params(self.cfg)

    def init_joint_response_params(self, cfg):
        # 初始化关节刚度
        self.joint_stiffness = torch.tensor(
            cfg.reconfiguration_config.stiffness, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

        # 初始化关节阻尼
        self.joint_damping = torch.tensor(
            cfg.reconfiguration_config.damping, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        # 从全局张量字典获取自由度状态和控制模式
        self.dof_states = global_tensor_dict["dof_state_tensor"]
        self.dof_control_mode = global_tensor_dict["dof_control_mode"]

        # 初始化自由度努力和目标张量
        self.dof_effort_tensor = torch.zeros_like(self.dof_states[..., 0])
        self.dof_position_setpoint_tensor = torch.zeros_like(self.dof_states[..., 0])
        self.dof_velocity_setpoint_tensor = torch.zeros_like(self.dof_states[..., 0])

        # 获取自由度状态的位置和速度
        self.dof_states_position = self.dof_states[..., 0]
        self.dof_states_velocity = self.dof_states[..., 1]

        # 更新全局张量字典
        global_tensor_dict["dof_position_setpoint_tensor"] = self.dof_position_setpoint_tensor
        global_tensor_dict["dof_velocity_setpoint_tensor"] = self.dof_velocity_setpoint_tensor
        global_tensor_dict["dof_effort_tensor"] = self.dof_effort_tensor

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # 随机初始化自由度状态在最小和最大范围之间
        self.dof_states[env_ids, :] = torch_rand_float_tensor(
            lower=self.joint_init_state_min[env_ids],
            upper=self.joint_init_state_max[env_ids],
        )

    def call_arm_controller(self):
        """
        调用四旋翼的臂控制器。此函数在每个仿真步骤调用。
        """
        if self.dof_control_mode == "effort":
            # 可以在此实现自定义非线性响应
            # 目前实现了一个简单的PD控制器
            pos_err = self.dof_position_setpoint_tensor - self.dof_states_position  # 位置误差
            vel_err = self.dof_velocity_setpoint_tensor - self.dof_states_velocity  # 速度误差
            self.dof_effort_tensor[:] = pd_control(
                pos_err,
                vel_err,
                self.joint_stiffness,
                self.joint_damping,
            )
        else:
            return

    def set_dof_position_targets(self, dof_pos_target):
        # 设置目标位置
        self.dof_position_setpoint_tensor[:] = dof_pos_target

    def set_dof_velocity_targets(self, dof_vel_target):
        # 设置目标速度
        self.dof_velocity_setpoint_tensor[:] = dof_vel_target

    def step(self, action_tensor):
        """
        更新四旋翼的状态。此函数在每个仿真步骤调用。
        """
        super().update_states()  # 更新状态
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")
        self.action_tensor[:] = action_tensor
        # 调用控制器进行控制分配
        super().call_controller()
        super().simulate_drag()  # 模拟阻力
        super().apply_disturbance()  # 应用扰动
        self.call_arm_controller()  # 调用臂控制器
