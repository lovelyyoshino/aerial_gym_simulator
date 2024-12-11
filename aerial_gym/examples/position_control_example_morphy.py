from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)  # 创建一个自定义日志记录器，用于输出程序运行信息
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = get_args()  # 获取命令行参数
    logger.warning(
        "This example demonstrates the use of geometric controllers with the Morphy robot in an empty environment."
    )  # 输出警告信息，说明示例的用途

    # 构建仿真环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim_2ms",  # 仿真名称
        env_name="empty_env_2ms",  # 环境名称
        robot_name="morphy",       # 机器人名称
        controller_name="lee_position_control",  # 控制器名称
        args=None,                 # 额外参数（此处为None）
        device="cuda:0",          # 使用CUDA设备
        num_envs=16,              # 环境数量
        headless=args.headless,   # 是否无头模式
        use_warp=args.use_warp,   # 是否使用warp功能
    )
    
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")  # 初始化动作张量，大小为(num_envs, 4)

    env_manager.reset()  # 重置环境
    arm1_pitch_list = []  # 存储第一个机械臂俯仰角度的列表
    arm1_yaw_list = []    # 存储第一个机械臂偏航角度的列表

    arm2_pitch_list = []  # 存储第二个机械臂俯仰角度的列表
    arm2_yaw_list = []    # 存储第二个机械臂偏航角度的列表

    arm3_pitch_list = []  # 存储第三个机械臂俯仰角度的列表
    arm3_yaw_list = []    # 存储第三个机械臂偏航角度的列表

    arm4_pitch_list = []  # 存储第四个机械臂俯仰角度的列表
    arm4_yaw_list = []    # 存储第四个机械臂偏航角度的列表

    for i in range(10000):  # 循环进行10000次迭代
        if i % 500 == 0:  # 每500步打印一次信息并更新目标设定点
            logger.info(f"Step {i}, changing target setpoint.")  # 输出当前步骤的信息
            actions[:, 0:3] = 0 * 1.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)  # 随机生成新的目标位置
            actions[:, 3] = 0 * torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)  # 随机生成新的目标旋转
            
            env_manager.reset()  # 重置环境
            if i == 0:
                continue  # 如果是第一次循环，则跳过绘图部分

            # 绘制各个机械臂的俯仰和偏航角度变化曲线
            plt.plot(arm1_pitch_list, label="arm1_pitch")
            plt.plot(arm1_yaw_list, label="arm1_roll")

            plt.plot(arm2_pitch_list, label="arm2_pitch")
            plt.plot(arm2_yaw_list, label="arm2_roll")

            plt.plot(arm3_pitch_list, label="arm3_pitch")
            plt.plot(arm3_yaw_list, label="arm3_roll")

            plt.plot(arm4_pitch_list, label="arm4_pitch")
            plt.plot(arm4_yaw_list, label="arm4_roll")
            plt.legend()  # 显示图例
            plt.show()  # 展示图形
        
        env_manager.step(actions=actions)  # 执行一步仿真，传入动作
        
        # 获取自由度状态并绘制它们
        dof_states = env_manager.global_tensor_dict["dof_state_tensor"]  # 从全局张量字典中获取自由度状态
        robot_0_dof_states = dof_states[0].cpu().numpy()  # 将第一个机器人的自由度状态转换为NumPy数组
        
        # 收集每个机械臂的俯仰和偏航角度数据
        arm1_pitch_list.append(robot_0_dof_states[0, 0])
        arm1_yaw_list.append(robot_0_dof_states[1, 0])

        arm2_pitch_list.append(robot_0_dof_states[2, 0])
        arm2_yaw_list.append(robot_0_dof_states[3, 0])

        arm3_pitch_list.append(robot_0_dof_states[4, 0])
        arm3_yaw_list.append(robot_0_dof_states[5, 0])

        arm4_pitch_list.append(robot_0_dof_states[6, 0])
        arm4_yaw_list.append(robot_0_dof_states[7, 0])
