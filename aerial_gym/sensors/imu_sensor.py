from aerial_gym.sensors.base_sensor import BaseSensor
import torch
from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    tensor_clamp,
    quat_rotate_inverse,
    quat_mul,
    torch_rand_float_tensor,
    quat_from_euler_xyz_tensor,
)

class IMUSensor(BaseSensor):
    def __init__(self, sensor_config, num_envs, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.world_frame = self.cfg.world_frame  # 传感器是否使用世界坐标系
        self.gravity_compensation = self.cfg.gravity_compensation  # 是否启用重力补偿

    def init_tensors(self, global_tensor_dict=None):
        self.global_tensor_dict = global_tensor_dict
        super().init_tensors(self.global_tensor_dict)

        self.force_sensor_tensor = self.global_tensor_dict["force_sensor_tensor"]

        # 初始化加速度计和陀螺仪的偏差和噪声标准差
        # 前3个值为加速度偏差/噪声，后3个值为陀螺仪偏差/噪声
        self.bias_std = torch.tensor(
            self.cfg.bias_std,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).expand(self.num_envs, -1)
        self.imu_noise_std = torch.tensor(
            self.cfg.imu_noise_std, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_measurement_value = torch.tensor(
            self.cfg.max_measurement_value, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.max_bias_init_value = torch.tensor(
            self.cfg.max_bias_init_value, device=self.device, requires_grad=False
        )

        self.accel_t = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)

        # 初始化传感器的旋转角度
        self.min_sensor_euler_rotation_rad = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)
        self.max_sensor_euler_rotation_rad = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, -1)

        # 从随机采样的欧拉角生成传感器的四元数
        self.sensor_quats = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )

        # 重力值初始化
        self.g_world = self.gravity * (
            1 - int(self.gravity_compensation)
        )  # 如果启用了重力补偿，则从加速度中减去重力

        # 与噪声和偏差相关的张量和变量
        self.enable_noise = int(self.cfg.enable_noise)  # 是否启用噪声
        self.enable_bias = int(self.cfg.enable_bias)  # 是否启用偏差
        self.bias = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)  # 偏差初始化
        self.noise = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)  # 噪声初始化
        self.imu_meas = torch.zeros((self.num_envs, 6), device=self.device, requires_grad=False)  # IMU测量初始化

        self.global_tensor_dict["imu_measurement"] = self.imu_meas  # 将IMU测量值存入全局张量字典

    def sample_noise(self):
        # 采样噪声
        self.noise = (
            torch.randn((self.num_envs, 6), device=self.device) * self.imu_noise_std / self.sqrt_dt
        )

    def update_bias(self):
        # 更新偏差
        self.bias_update_step = (
            torch.randn((self.num_envs, 6), device=self.device) * self.bias_std * self.sqrt_dt
        )
        self.bias += self.bias_update_step  # 将更新的偏差加到当前偏差上

    def update(self):
        """
        更新IMU传感器的测量值
        world_frame: 如果加速度和角速度在世界坐标系下
        """
        self.accel_t = self.force_sensor_tensor[:, 0:3] / self.robot_masses.unsqueeze(1)  # 计算加速度
        if self.world_frame:
            # 在世界坐标系下计算加速度和角速度
            acceleration = quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats),
                (self.accel_t - self.g_world),
            )
            ang_rate = quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats),
                self.robot_body_angvel,
            )
        else:
            # 从真实传感器框架旋转到扰动传感器框架
            acceleration = quat_rotate_inverse(
                self.sensor_quats, self.accel_t
            ) - quat_rotate_inverse(
                quat_mul(self.robot_orientation, self.sensor_quats), self.g_world
            )
            ang_rate = quat_rotate_inverse(self.sensor_quats, self.robot_body_angvel)

        self.sample_noise()  # 采样噪声
        self.update_bias()  # 更新偏差
        # 计算带偏差和噪声的加速度和角速度测量值
        accel_meas = (
            acceleration
            + self.enable_bias * self.bias[:, :3]
            + self.enable_noise * self.noise[:, :3]
        )
        ang_rate_meas = (
            ang_rate + self.enable_bias * self.bias[:, 3:] + self.enable_noise * self.noise[:, 3:]
        )
        # 将加速度计和陀螺仪的测量值限制在最大值范围内
        accel_meas = tensor_clamp(
            accel_meas,
            -self.max_measurement_value[:, 0:3],
            self.max_measurement_value[:, 0:3],
        )
        ang_rate_meas = tensor_clamp(
            ang_rate_meas,
            -self.max_measurement_value[:, 3:],
            self.max_measurement_value[:, 3:],
        )
        self.imu_meas[:, :3] = accel_meas  # 更新加速度测量值
        self.imu_meas[:, 3:] = ang_rate_meas  # 更新角速度测量值
        return

    def reset(self):
        # 重置偏差和四元数
        self.bias.zero_()  # 清零偏差
        self.bias[:] = self.max_bias_init_value * (2.0 * (torch.rand_like(self.bias) - 0.5))  # 随机初始化偏差
        self.sensor_quats[:] = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )  # 随机初始化传感器四元数

    def reset_idx(self, env_ids):
        # 重置指定环境的偏差和四元数
        self.bias[env_ids, :] = (
            self.max_bias_init_value * (2.0 * (torch.rand_like(self.bias) - 0.5))
        )[env_ids, :]
        self.sensor_quats[env_ids] = quat_from_euler_xyz_tensor(
            torch_rand_float_tensor(
                self.min_sensor_euler_rotation_rad, self.max_sensor_euler_rotation_rad
            )
        )[env_ids]

    def get_observation(self):
        # 获取传感器观测值（未实现）
        pass
