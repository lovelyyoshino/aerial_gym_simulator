from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots

    # 设置matplotlib的主题为科学主题
    plt.style.use(["science", "vibrant"])
except:
    # 如果没有安装scienceplots，则设置plt主题为seaborn colorblind
    plt.style.use("seaborn-v0_8-colorblind")

import csv

angle_list = []  # 用于存储从CSV文件读取的角度数据


def read_csv(filename):
    """
    从指定的CSV文件中读取时间和角度值，并将符合条件的数据添加到angle_list中。
    
    参数:
        filename (str): CSV文件的路径
    """
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for t, theta in reader:
            try:
                # 只保留时间大于0.06且角度小于15的记录
                if float(t) > 0.06 and float(theta) < 15.0:
                    angle_list.append([float(t), float(theta)])
            except:
                pass


filename = "./stored_data/joint_step.csv"  # 指定要读取的CSV文件路径

read_csv(filename)  # 调用函数读取CSV文件
time_stamp = np.array([x[0] for x in angle_list])  # 提取时间戳
angle_rad = np.array([x[1] * torch.pi / 180.0 for x in angle_list])  # 将角度转换为弧度


def mass_spring_damper(y, t, k_p, k_v):
    """
    描述质量-弹簧-阻尼器系统的动态方程。

    参数:
        y (list): 当前状态 [theta, omega]
        t (float): 当前时间（未使用）
        k_p (float): 弹簧常数
        k_v (float): 阻尼系数
    
    返回:
        dydt (list): 状态变化率 [domega, dtheta]
    """
    theta, omega = y  # 解包当前状态
    dydt = [omega, -k_v * omega - k_p * torch.sign(theta) * theta**2]  # 计算状态变化率
    return dydt


if __name__ == "__main__":
    args = get_args()  # 获取命令行参数

    logger.warning(
        "此示例演示了在空环境中记录Morphy机器人臂数据。"
    )

    env_manager = SimBuilder().build_env(
        sim_name="base_sim_2ms",
        env_name="empty_env_2ms",
        robot_name="morphy",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    # 检查机器人的基座链接是否固定
    if env_manager.robot_manager.robot.cfg.robot_asset.fix_base_link == False:
        logger.error(
            "该机器人基座链接未固定。请确保在morphy_config.py中将其设置为固定以使本示例正常工作。"
        )
        exit(1)

    actions = 0.0 * 0.3 * 9.81 * torch.ones((env_manager.num_envs, 4)).to("cuda:0")  # 初始化动作
    actions[:, [0, 1, 2]] = 0.0  # 设置前3个动作为0
    dof_pos = torch.zeros((env_manager.num_envs, 8)).to("cuda:0")  # 初始化自由度位置
    env_manager.reset()  # 重置环境
    arm1_pitch_list = []  # 存储第一个机械臂的俯仰角列表
    arm1_yaw_list = []  # 存储第一个机械臂的偏航角列表

    arm2_pitch_list = []  # 存储第二个机械臂的俯仰角列表
    arm2_yaw_list = []  # 存储第二个机械臂的偏航角列表

    arm3_pitch_list = []  # 存储第三个机械臂的俯仰角列表
    arm3_yaw_list = []  # 存储第三个机械臂的偏航角列表

    arm4_pitch_list = []  # 存储第四个机械臂的俯仰角列表
    arm4_yaw_list = []  # 存储第四个机械臂的偏航角列表
    popt = [5834.85432241, 229.23612708]  # 模型参数

    logger.warning(
        "请确保morphy_config.py中的fix_base_link设置为True。同时确保机械臂的初始状态已根据配置文件适当设置。"
    )

    for i in range(100000):
        if i % (1500) == 0:  # 每1500步更新一次目标设定点
            logger.info(f"步骤 {i}, 更改目标设定点。")
            actions[:, 0:3] = 0 * 1.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)  # 随机生成新的动作
            actions[:, 3] = 0 * torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset()  # 重置环境
            env_manager.robot_manager.robot.set_dof_position_targets(dof_pos)  # 设置自由度位置目标
            env_manager.robot_manager.robot.set_dof_velocity_targets(dof_pos)  # 设置自由度速度目标
            if i == 0:
                continue
            
            x_labels = torch.arange(0, len(arm1_pitch_list), 1).cpu().numpy()  # 创建X轴标签

            x = [0.25, 0]  # 初始状态
            N = 7500  # 仿真步数
            dt = 0.002  # 时间步长
            t = []
            d = []
            v = []
            for i in range(N):
                t.append(i * dt)  # 记录时间
                d.append(x[0])  # 记录位移
                v.append(x[1])  # 记录速度
                x_inp = torch.tensor(x).to("cuda:0")  # 转换为张量并移动到GPU
                x_inp[0] -= 7.2 * torch.pi / 180.0  # 调整输入
                xdot = mass_spring_damper(x_inp, 0, *popt)  # 计算状态变化
                for j in range(2):
                    x[j] += xdot[j].cpu().numpy() * dt  # 更新状态

            fig, ax = plt.subplots(figsize=(6, 2.5))  # 创建绘图对象
            ax.plot(
                x_labels * 0.01,
                arm1_pitch_list,
                label="模拟响应",
                marker="o",
                markersize=4,
            )
            ax.plot(t, d, "--", label="识别模型", linewidth=2, color="black")
            ax.plot(time_stamp, angle_rad, label="真实响应")
            ax.set(xlabel="时间 (s)", ylabel=r"$\theta_j$ (rad)")  # 设置坐标轴标签
            ax.legend()  # 显示图例
            # 保存图形为PDF
            # plt.savefig('morphy_response.pdf')
            plt.show()  # 显示图形
            exit(0)  # 退出程序
        env_manager.step(actions=actions)  # 执行动作，更新环境
        # 获取自由度状态并进行绘制
        dof_states = env_manager.global_tensor_dict["dof_state_tensor"]  # 获取自由度状态张量
        robot_0_dof_states = dof_states[0].cpu().numpy()  # 将第一个机器人的状态转为NumPy数组
        # 绘制状态
        arm1_pitch_list.append(robot_0_dof_states[0, 0])  # 添加第一个机械臂的俯仰角
        arm1_yaw_list.append(robot_0_dof_states[1, 0])  # 添加第一个机械臂的偏航角

        arm2_pitch_list.append(robot_0_dof_states[2, 0])  # 添加第二个机械臂的俯仰角
        arm2_yaw_list.append(robot_0_dof_states[3, 0])  # 添加第二个机械臂的偏航角

        arm3_pitch_list.append(robot_0_dof_states[4, 0])  # 添加第三个机械臂的俯仰角
        arm3_yaw_list.append(robot_0_dof_states[5, 0])  # 添加第三个机械臂的偏航角

        arm4_pitch_list.append(robot_0_dof_states[6, 0])  # 添加第四个机械臂的俯仰角
        arm4_yaw_list.append(robot_0_dof_states[7, 0])  # 添加第四个机械臂的偏航角
