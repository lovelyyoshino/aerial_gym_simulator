from aerial_gym.robots.base_reconfigurable import BaseReconfigurable
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import pd_control

# 创建一个自定义日志记录器，用于记录与“morphy_robot_class”相关的信息
logger = CustomLogger("morphy_robot_class")
import torch


class Morphy(BaseReconfigurable):
    def __init__(self, robot_config, controller_name, env_config, device):
        # 初始化Morphy类，调用父类的构造函数
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )

    def init_joint_response_params(self, cfg):
        """
        初始化关节响应参数，包括刚度和阻尼
        :param cfg: 配置对象，包含重新配置的参数
        """
        # 将刚度参数转换为张量，并扩展到环境数量
        self.joint_stiffness = torch.tensor(
            cfg.reconfiguration_config.stiffness, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        
        # 将阻尼参数转换为张量，并扩展到环境数量
        self.joint_damping = torch.tensor(
            cfg.reconfiguration_config.damping, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        
        # 获取自定义非线性刚度和线性阻尼参数
        self.stiffness_param = cfg.reconfiguration_config.custom_nonlinear_stiffness
        self.damping_param = cfg.reconfiguration_config.custom_linear_damping

    def call_arm_controller(self):
        """
        调用四旋翼手臂的控制器。该函数在每个仿真步骤中被调用。
        """
        if self.dof_control_mode == "effort":
            # 计算关节的力矩
            self.dof_effort_tensor[:] = (
                0.01625
                * (0.07 * 0.07)
                * arm_response_func(
                    (self.dof_states_position[:] - 7.2 * torch.pi / 180.0),  # 位置误差
                    self.dof_states_velocity[:],  # 速度误差
                    self.stiffness_param,  # 刚度参数
                    self.damping_param,  # 阻尼参数
                )
            )
            # 考虑重力的影响，调整关节的力矩
            self.dof_effort_tensor[:] -= (
                9.81 * 0.01625 * 0.07 * torch.cos(self.dof_states_position[:])
            )
        else:
            return


@torch.jit.script
def arm_response_func(pos_error, vel_error, lin_damper, nonlin_spring):
    """
    计算手臂响应，基于位置误差和速度误差
    :param pos_error: 位置误差
    :param vel_error: 速度误差
    :param lin_damper: 线性阻尼参数
    :param nonlin_spring: 非线性弹簧参数
    :return: 计算出的响应力
    """
    # 计算响应力：线性阻尼项 + 非线性弹簧项
    return lin_damper * vel_error + nonlin_spring * torch.sign(pos_error) * pos_error**2
