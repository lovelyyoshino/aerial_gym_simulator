from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
from matplotlib import pyplot as plt

# 控制模式字典，映射不同的控制模式到对应的控制器名称
CONTROLLER_MODES = {
    "attitude": "lmf2_attitude_control",
    "velocity": "lmf2_velocity_control",
    "acceleration": "lmf2_acceleration_control",
}

# 字典映射，用于将控制模式映射到相应的观测数据
DICT_MAP = {
    "attitude": "robot_euler_angles",
    "velocity": "robot_vehicle_linvel",
    "acceleration": "imu_measurement",
}

# Y轴标签字典，用于绘图时标记每个子图的Y轴
Y_AXIS_LABELS = {
    0: "X",
    1: "Y",
    2: "Z",
    3: "Yaw Rate",
}

if __name__ == "__main__":
    # 设置控制模式为速度控制
    CONTROL_MODE_NAME = "velocity"
    DICT_MAP_ENTRY = DICT_MAP[CONTROL_MODE_NAME]  # 获取当前控制模式对应的字典条目
    CONTROLLER_NAME = CONTROLLER_MODES[CONTROL_MODE_NAME]  # 获取当前控制模式对应的控制器名称
    args = get_args()  # 获取命令行参数
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",  # 模拟环境名称
        env_name="empty_env",  # 环境名称
        robot_name="lmf2",  # 机器人名称
        controller_name=CONTROLLER_NAME,  # 控制器名称
        args=None,
        device="cuda:0",  # 使用CUDA设备
        num_envs=args.num_envs,  # 环境数量
        headless=args.headless,  # 是否无头模式
        use_warp=args.use_warp,  # 是否使用warp
    )
    
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")  # 初始化动作张量
    env_manager.reset()  # 重置环境
    tensor_dict = env_manager.get_obs()  # 获取初始观测值
    observations = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")  # 初始化观测张量
    ACTION_MAGNITUDE = 1.0  # 动作幅度
    SIM_DURATION_IN_SECONDS = 5.0  # 模拟持续时间（秒）
    SIM_DT = 0.01  # 模拟时间步长
    TIME_CONSTANT_MAGNITUDE = ACTION_MAGNITUDE * 0.63212  # 时间常数幅度
    num_sim_steps = int(SIM_DURATION_IN_SECONDS / SIM_DT)  # 计算模拟步骤总数
    observation_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cuda:0")  # 初始化观测序列
    actions_sequence = torch.zeros((num_sim_steps, env_manager.num_envs, 4)).to("cuda:0")  # 初始化动作序列
    time_elapsed_np = torch.arange(0, SIM_DURATION_IN_SECONDS, SIM_DT).cpu().numpy()  # 创建时间数组
    print(f"\n\n\n\nPerforming System Identification for {CONTROL_MODE_NAME} control mode\n\n")
    
    # 遍历四种动作索引
    for action_index in range(4):
        actions[:] = 0.0  # 将所有动作初始化为0
        print("Action Index: ", action_index)
        time_constant = 0.0  # 初始化时间常数
        
        # 执行模拟步骤
        for i in range(num_sim_steps):
            if i == num_sim_steps // 2:
                actions[:, action_index] = ACTION_MAGNITUDE  # 在中间步骤施加动作
            
            env_manager.step(actions)  # 执行动作并更新环境状态
            observations[:, 0:3] = tensor_dict[DICT_MAP_ENTRY][:, 0:3]  # 更新观测值
            observations[:, 3] = tensor_dict["robot_angvel"][:, 2]  # 更新角速度观测
            
            # 根据控制模式判断时间常数
            if CONTROL_MODE_NAME == "attitude":
                if action_index > 0:
                    if (
                        observations[0, action_index - 1] > TIME_CONSTANT_MAGNITUDE
                        and time_constant == 0.0
                    ):
                        time_constant = (i - num_sim_steps // 2) * SIM_DT  # 计算时间常数
            else:
                if observations[0, action_index] > TIME_CONSTANT_MAGNITUDE and time_constant == 0.0:
                    time_constant = (i - num_sim_steps // 2) * SIM_DT  # 计算时间常数
            
            observation_sequence[i] = observations  # 保存观测序列
            
            # 根据控制模式保存动作序列
            if CONTROL_MODE_NAME == "attitude":
                actions_sequence[i, :, 0:2] = actions[:, 1:3]
                actions_sequence[i, :, 3] = actions[:, 3]
            else:
                actions_sequence[i] = actions

        # 绘制系统响应图
        observation_sequence_np = observation_sequence.clone().cpu().numpy()  # 转换观测序列为NumPy格式
        actions_sequence_np = actions_sequence.clone().cpu().numpy()  # 转换动作序列为NumPy格式
        fig, axs = plt.subplots(4, 1)  # 创建4个子图
        print("Time Constant: ", time_constant)  # 打印时间常数
        fig.suptitle(
            f"System ID for {CONTROL_MODE_NAME} with activation in action {action_index}",
            fontsize=16,
        )
        
        # 绘制每个观测和动作的曲线
        for i in range(4):
            axs[i].plot(time_elapsed_np, observation_sequence_np[:, 0, i])  # 绘制观测曲线
            axs[i].plot(time_elapsed_np, actions_sequence_np[:, 0, i])  # 绘制动作曲线
            axs[i].set_ylabel(Y_AXIS_LABELS[i])  # 设置Y轴标签
            axs[i].set_xlabel("Time")  # 设置X轴标签
            axs[i].set_ylim(-2, 2)  # 设置Y轴范围
        plt.show(block=False)  # 显示图形但不阻塞程序执行
        env_manager.reset()  # 重置环境以准备下一个动作

    plt.show()  # 最后显示所有图形
