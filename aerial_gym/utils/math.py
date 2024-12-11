import torch
from torch import Tensor
from pytorch3d.transforms import matrix_to_quaternion


@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor
    # 计算反对称矩阵的Vee映射，返回一个向量
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1
    )
    return vee_map


@torch.jit.script
def torch_rand_float_vec(lower, upper, shape, device):
    # type: (torch.Tensor, torch.Tensor, Tuple[int, int, int], str) -> torch.Tensor
    # 在给定范围内生成随机浮点数向量
    return torch.rand(*shape, device=device) * (upper - lower) + lower


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    # 返回最小有符号角度
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


@torch.jit.script
def torch_rand_float_tensor(lower, upper):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    # 在给定范围内生成随机浮点数张量
    return (upper - lower) * torch.rand_like(upper) + lower


@torch.jit.script
def quat_rotate(q, v):
    # 旋转向量v使用四元数q
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    # 获取沿指定轴的单位基向量经过四元数q旋转后的结果
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def exponential_reward_function(
    magnitude: float, base_width: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    # 指数奖励函数，根据输入值计算奖励
    return magnitude * torch.exp(-(value * value) / base_width)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, base_width: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential penalty function"""
    # 指数惩罚函数，根据输入值计算惩罚
    return magnitude * (torch.exp(-(value * value) / base_width) - 1.0)


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    # 将a的绝对值与b的符号结合，返回新的Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    # 从四元数q中提取欧拉角（roll, pitch, yaw）
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)


@torch.jit.script
def get_euler_xyz_tensor(q):
    # 从四元数q中提取欧拉角并以张量形式返回
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack(
        [roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)], dim=-1
    )


@torch.jit.script_if_tracing
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle"""
    # 返回最小有符号角度
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


@torch.jit.script
def quat_from_euler_xyz_tensor(euler_xyz_tensor: torch.Tensor) -> torch.Tensor:
    # 从欧拉角张量转换为四元数
    roll = euler_xyz_tensor[..., 0]
    pitch = euler_xyz_tensor[..., 1]
    yaw = euler_xyz_tensor[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def vehicle_frame_quat_from_quat(body_quat: torch.Tensor) -> torch.Tensor:
    # 从车体四元数获取车辆框架下的四元数
    body_euler = get_euler_xyz_tensor(body_quat) * torch.tensor(
        [0.0, 0.0, 1.0], device=body_quat.device
    )
    return quat_from_euler_xyz_tensor(body_euler)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    # 从欧拉角创建四元数
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_interpolate_ratio(min, max, ratio):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # 根据比例在min和max之间插值
    return min + (max - min) * ratio


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    # 在给定范围内生成随机浮点数
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    # 生成二维随机方向向量
    angle = torch_rand_float(-torch.pi, torch.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    # 限制t的值在[min_t, max_t]之间
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    # 将[-1, 1]区间的x缩放到[lower, upper]区间
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    # 将[lower, upper]区间的x反缩放回[-1, 1]区间
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    # numpy版本的unscale函数
    return (2.0 * x - upper - lower) / (upper - lower)


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    # 将numpy数组或其他数据类型转换为PyTorch张量
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    # 四元数相乘
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_to_rotation_matrix(a):
    # 将四元数转换为旋转矩阵
    shape = a.shape
    a = a.reshape(-1, 4)

    # convert the quaternion to a rotation matrix
    x, y, z, w = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    xx, xy, xz, xw = x * x, x * y, x * z, x * w
    yy, yz, yw = y * y, y * z, y * w
    zz, zw = z * z, z * w
    wx, wy, wz = xw, yw, zw
    # 使用上述定义的变量来创建旋转矩阵
    m = torch.stack(
        [
            1 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    return m.view(shape[0], 3, 3)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    # 对向量进行归一化处理，避免除零错误
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_conjugate(a):
    # 计算四元数的共轭
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inverse(a):
    # 计算四元数的逆
    return quat_conjugate(a)


@torch.jit.script
def quat_apply(a, b):
    # 应用四元数a于向量b
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_inverse(a, b):
    # 应用四元数的逆于向量b
    return quat_apply(quat_inverse(a), b)


@torch.jit.script
def quat_rotate(q, v):
    # 使用四元数q旋转向量v
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    # 使用四元数q的逆旋转向量v
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_unit(a):
    # 归一化四元数
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    # 从角度和轴创建四元数
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    # 规范化角度到[-π, π]区间
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    # 计算变换的逆，包括四元数和位移
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    # 应用变换，将位移t和向量v通过四元数q组合
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    # 应用四元数q于向量v
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    # 合并两个变换
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    # 获取经过四元数q旋转的基向量v
    return quat_rotate(q, v)


@torch.jit.script
def pd_control(
    pos_error,
    vel_error,
    stiffness,
    damping,
):
    # PD控制器，计算控制输出
    return stiffness * pos_error + damping * vel_error
