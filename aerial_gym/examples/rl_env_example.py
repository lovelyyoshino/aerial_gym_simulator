import time  # 导入时间模块，用于计算程序运行时间
from aerial_gym.utils.logging import CustomLogger  # 从自定义日志模块导入CustomLogger类

logger = CustomLogger(__name__)  # 创建一个CustomLogger实例，记录当前模块的日志
from aerial_gym.registry.task_registry import task_registry  # 从任务注册表中导入task_registry
import torch  # 导入PyTorch库，用于张量操作和深度学习

if __name__ == "__main__":  # 如果该脚本是主程序执行
    logger.print_example_message()  # 打印示例消息到日志
    logger.warning("\n\n\nJust an example task interface.\n\n\n")  # 输出警告信息，说明这是一个示例任务接口
    start = time.time()  # 记录开始时间
    rl_task_env = task_registry.make_task(  # 创建强化学习任务环境
        "position_setpoint_task",  # 指定任务名称为"position_setpoint_task"，这个在task任务表中有定义
    )
    rl_task_env.reset()  # 重置任务环境，以便开始新的任务
    actions = torch.zeros(  # 初始化动作张量，形状为（环境数量，动作数量）
        (
            rl_task_env.sim_env.num_envs,  # 获取当前环境的数量
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,  # 获取每个机器人控制器的动作数量
        )
    ).to("cuda:0")  # 将张量移动到GPU上进行加速计算
    actions[:] = 0.0  # 将所有动作初始化为0.0
    with torch.no_grad():  # 在不需要计算梯度的上下文中执行以下代码
        for i in range(10000):  # 循环10000次
            if i == 100:  # 当循环计数达到100时
                start = time.time()  # 重新记录开始时间
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)  # 执行动作并获取观察值、奖励、终止标志、截断标志和其他信息
    end = time.time()  # 记录结束时间
