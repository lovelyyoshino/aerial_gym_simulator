from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)

from aerial_gym.utils.math import quat_rotate_inverse, get_euler_xyz_tensor
import torch


class DCE_RL_Navigation_Task(NavigationTask):
    def __init__(self, task_config, **kwargs):
        # 设置动作空间维度为3
        task_config.action_space_dim = 3
        # 设置课程学习的最小级别为36
        task_config.curriculum.min_level = 36
        logger.critical("Hardcoding number of envs to 16 if it is greater than that.")
        # 如果环境数量大于16，则强制设置为16
        task_config.num_envs = 16 if task_config.num_envs > 16 else task_config.num_envs
        # 调用父类构造函数进行初始化
        super().__init__(task_config=task_config, **kwargs)

    # 修改观察值返回方式以使代码正常工作
    # 这是原始代码。

    def process_obs_for_task(self):
        # 计算目标位置与机器人当前位置之间的向量，并将其转换到机器人的坐标系中
        vec_to_target = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        # 计算到达目标的距离
        dist_to_tgt = torch.norm(vec_to_target, dim=1)
        # 将归一化后的目标向量存储在任务观察中
        self.task_obs["observations"][:, 0:3] = vec_to_target / dist_to_tgt.unsqueeze(1)
        # 将距离标准化并存储在任务观察中
        self.task_obs["observations"][:, 3] = dist_to_tgt / 5.0
        # 获取机器人的欧拉角并存储前两个角度
        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6] = euler_angles[:, 0:2]
        # 存储一个零值（可能是占位符）
        self.task_obs["observations"][:, 6] = 0.0
        # 存储机器人的线速度和角速度
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        # 存储机器人的动作信息
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        # 存储图像潜变量
        self.task_obs["observations"][:, 17:81] = self.image_latents


@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """计算最小有符号角度"""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi
