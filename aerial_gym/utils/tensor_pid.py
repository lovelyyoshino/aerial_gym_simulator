import torch
from isaacgym.torch_utils import tensor_clamp

class TensorPID:
    def __init__(
        self,
        num_envs,  # 环境数量
        num_dims,  # 每个环境的维度数
        Kp,  # 比例增益
        Kd,  # 微分增益
        Ki,  # 积分增益
        dt,  # 时间步长
        integral_min_limit,  # 积分项最小限制
        integral_max_limit,  # 积分项最大限制
        derivative_saturation_min_limit,  # 微分项饱和最小限制
        derivative_saturation_max_limit,  # 微分项饱和最大限制
        output_min_limit,  # 输出最小限制
        output_max_limit,  # 输出最大限制
        device=torch.device("cuda"),  # 使用的设备，默认为CUDA
    ):
        # 初始化PID控制器参数
        self.device = device
        self.Kp = torch.tensor(Kp, device=self.device)  # 将比例增益转换为张量并放置在指定设备上
        self.Kd = torch.tensor(Kd, device=self.device)  # 将微分增益转换为张量并放置在指定设备上
        self.Ki = torch.tensor(Ki, device=self.device)  # 将积分增益转换为张量并放置在指定设备上
        self.dt = dt  # 设置时间步长
        self.integral_min_limit = torch.tensor(integral_min_limit, device=self.device)  # 积分项最小限制
        self.integral_max_limit = torch.tensor(integral_max_limit, device=self.device)  # 积分项最大限制
        self.derivative_saturation_min_limit = torch.tensor(
            derivative_saturation_min_limit, device=self.device
        )  # 微分项饱和最小限制
        self.derivative_saturation_max_limit = torch.tensor(
            derivative_saturation_max_limit, device=self.device
        )  # 微分项饱和最大限制
        self.output_min_limit = torch.tensor(output_min_limit, device=self.device)  # 输出最小限制
        self.output_max_limit = torch.tensor(output_max_limit, device=self.device)  # 输出最大限制
        self.integral = torch.zeros((num_envs, num_dims), device=self.device)  # 初始化积分项为零
        self.prev_error = torch.zeros((num_envs, num_dims), device=self.device)  # 初始化前一个误差为零
        self.reset_state = torch.ones((num_envs, num_dims), device=self.device)  # 重置状态初始化为1

    def update(self, error):
        # 更新PID控制器的输出
        self.integral += error * self.dt  # 更新积分项
        # 计算PID各项
        proportional_term = self.Kp * error  # 比例项
        derivative_term = self.Kd * (1 - self.reset_state) * (error - self.prev_error) / self.dt  # 微分项
        integral_term = self.Ki * self.integral  # 积分项
        # 限制积分项以避免数值不稳定
        integral_term = tensor_clamp(
            integral_term, self.integral_min_limit, self.integral_max_limit
        )
        derivative_term = tensor_clamp(
            derivative_term,
            self.derivative_saturation_min_limit,
            self.derivative_saturation_max_limit,
        )
        # 计算PID输出
        output = proportional_term + derivative_term + integral_term
        # 限制输出以避免数值不稳定
        output = tensor_clamp(output, self.output_min_limit, self.output_max_limit)
        self.prev_error = error  # 更新前一个误差
        self.reset_state[:, :] = 0.0  # 重置状态更新为0
        return output  # 返回PID控制器的输出

    def reset(self):
        # 重置PID控制器的状态
        self.integral[:, :] = 0  # 清空积分项
        self.prev_error[:, :] = 0  # 清空前一个误差
        self.reset_state[:, :] = 1.0  # 重置状态设置为1

    def reset_idx(self, env_idx):
        # 根据环境索引重置特定环境的PID控制器状态
        self.integral[env_idx, :] = 0  # 清空该环境的积分项
        self.prev_error[env_idx, :] = 0.0  # 清空该环境的前一个误差
        self.reset_state[env_idx, :] = 1.0  # 重置该环境的状态设置为1
