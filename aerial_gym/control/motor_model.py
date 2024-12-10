import torch

from aerial_gym.utils.math import torch_rand_float_tensor, tensor_clamp


class MotorModel:
    def __init__(self, num_envs, motors_per_robot, dt, config, device="cuda:0"):
        # 初始化电机模型
        # num_envs: 环境数量
        # motors_per_robot: 每个机器人电机数量
        # dt: 时间步长
        # config: 配置参数
        # device: 设备类型（如GPU或CPU）
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.device = device
        self.num_motors_per_robot = motors_per_robot
        self.max_thrust = self.cfg.max_thrust  # 最大推力
        self.min_thrust = self.cfg.min_thrust  # 最小推力
        
        # 初始化电机时间常数的上下限
        self.motor_time_constant_increasing_min = torch.tensor(
            self.cfg.motor_time_constant_increasing_min, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_increasing_max = torch.tensor(
            self.cfg.motor_time_constant_increasing_max, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_decreasing_min = torch.tensor(
            self.cfg.motor_time_constant_decreasing_min, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_decreasing_max = torch.tensor(
            self.cfg.motor_time_constant_decreasing_max, device=self.device
        ).expand(self.num_envs, self.num_motors_per_robot)
        
        # 最大推力变化率
        self.max_rate = torch.tensor(self.cfg.max_thrust_rate, device=self.device).expand(
            self.num_envs, self.num_motors_per_robot
        )
        self.init_tensors()  # 初始化张量

    def init_tensors(self, global_tensor_dict=None):
        # 初始化电机推力和时间常数的张量
        self.current_motor_thrust = torch_rand_float_tensor(
            torch.tensor(self.min_thrust, device=self.device, dtype=torch.float32).expand(
                self.num_envs, self.num_motors_per_robot
            ),
            torch.tensor(self.max_thrust, device=self.device, dtype=torch.float32).expand(
                self.num_envs, self.num_motors_per_robot
            ),
        )
        self.motor_time_constants_increasing = torch_rand_float_tensor(
            self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max
        )
        self.motor_time_constants_decreasing = torch_rand_float_tensor(
            self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max
        )
        self.motor_rate = torch.zeros(
            (self.num_envs, self.num_motors_per_robot), device=self.device
        )
        
        # 如果使用转速（RPM）控制方式
        if self.cfg.use_rps:
            self.motor_thrust_constant_min = (
                torch.ones(
                    self.num_envs,
                    self.num_motors_per_robot,
                    device=self.device,
                    requires_grad=False,
                )
                * self.cfg.motor_thrust_constant_min
            )
            self.motor_thrust_constant_max = (
                torch.ones(
                    self.num_envs,
                    self.num_motors_per_robot,
                    device=self.device,
                    requires_grad=False,
                )
                * self.cfg.motor_thrust_constant_max
            )
            self.motor_thrust_constant = torch_rand_float_tensor(
                self.motor_thrust_constant_min, self.motor_thrust_constant_max
            )
        # 根据配置选择混合因子函数
        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = discrete_mixing_factor
        else:
            self.mixing_factor_function = continuous_mixing_factor

    def update_motor_thrusts(self, ref_thrust):
        # 更新电机推力
        # ref_thrust: 参考推力
        # 将参考推力限制在最小和最大推力范围内
        ref_thrust = torch.clamp(ref_thrust, self.min_thrust, self.max_thrust)
        thrust_error = ref_thrust - self.current_motor_thrust  # 计算推力误差
        
        # 选择合适的时间常数
        motor_time_constants = torch.where(
            torch.sign(self.current_motor_thrust) * torch.sign(thrust_error) < 0,
            self.motor_time_constants_decreasing,
            self.motor_time_constants_increasing,
        )
        
        # 计算混合因子
        mixing_factor = self.mixing_factor_function(self.dt, motor_time_constants)
        
        # 根据是否使用转速控制方式更新当前推力
        if self.cfg.use_rps:
            self.current_motor_thrust[:] = compute_thrust_with_rpm_time_constant(
                ref_thrust,
                self.current_motor_thrust,
                mixing_factor,
                self.motor_thrust_constant,
                self.max_rate,
                self.dt,
            )
        else:
            self.current_motor_thrust[:] = compute_thrust_with_force_time_constant(
                ref_thrust,
                self.current_motor_thrust,
                mixing_factor,
                self.max_rate,
                self.dt,
            )
        return self.current_motor_thrust  # 返回当前推力

    def reset_idx(self, env_ids):
        # 重置指定环境的电机状态
        self.motor_time_constants_increasing[env_ids] = torch_rand_float_tensor(
            self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max
        )[env_ids]

        self.motor_time_constants_decreasing[env_ids] = torch_rand_float_tensor(
            self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max
        )[env_ids]
        
        # 重置当前推力
        self.current_motor_thrust[env_ids] = torch_rand_float_tensor(
            self.min_thrust, self.max_thrust
        )[env_ids]
        
        if self.cfg.use_rps:
            self.motor_thrust_constant[env_ids] = torch_rand_float_tensor(
                self.motor_thrust_constant_min, self.motor_thrust_constant_max
            )[env_ids]
        
        self.first_order_linear_mixing_factor[env_ids] = self.mixing_factor_function(
            self.dt, self.motor_time_constants
        )[env_ids]

    def reset(self):
        # 重置所有环境的电机状态
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


@torch.jit.script
def motor_model_rate(error, mixing_factor, max_rate):
    # 计算电机模型的变化率
    # error: 推力误差
    # mixing_factor: 混合因子
    # max_rate: 最大变化率
    return tensor_clamp(mixing_factor * (error), -max_rate, max_rate)  # 限制变化率

@torch.jit.script
def discrete_mixing_factor(dt, time_constant):
    # 计算离散混合因子
    # dt: 时间步长
    # time_constant: 时间常数
    return 1.0 / (dt + time_constant)

@torch.jit.script
def continuous_mixing_factor(dt, time_constant):
    # 计算连续混合因子
    # dt: 时间步长
    # time_constant: 时间常数
    return 1.0 / time_constant

@torch.jit.script
def compute_thrust_with_rpm_time_constant(
    ref_thrust, current_thrust, mixing_factor, thrust_constant, max_rate, dt
):
    # 计算基于转速时间常数的推力
    # ref_thrust: 参考推力
    # current_thrust: 当前推力
    # mixing_factor: 混合因子
    # thrust_constant: 推力常数
    # max_rate: 最大变化率
    # dt: 时间步长
    current_rpm = torch.sqrt(current_thrust / thrust_constant)  # 当前转速
    desired_rpm = torch.sqrt(ref_thrust / thrust_constant)  # 目标转速
    rpm_error = desired_rpm - current_rpm  # 转速误差
    current_rpm += motor_model_rate(rpm_error, mixing_factor, max_rate) * dt  # 更新当前转速
    return thrust_constant * current_rpm**2  # 返回计算后的推力

@torch.jit.script
def compute_thrust_with_force_time_constant(
    ref_thrust, current_thrust, mixing_factor, max_rate, dt
):
    # 计算基于力时间常数的推力
    # ref_thrust: 参考推力
    # current_thrust: 当前推力
    # mixing_factor: 混合因子
    # max_rate: 最大变化率
    # dt: 时间步长
    thrust_error = ref_thrust - current_thrust  # 推力误差
    current_thrust[:] += motor_model_rate(thrust_error, mixing_factor, max_rate) * dt  # 更新当前推力
    return current_thrust  # 返回计算后的推力
