import time  # 导入时间模块，用于计算运行时间
from aerial_gym.utils.logging import CustomLogger  # 从自定义日志模块导入CustomLogger类

logger = CustomLogger(__name__)  # 创建一个CustomLogger实例，记录当前模块的日志
from aerial_gym.registry.task_registry import task_registry  # 从任务注册表中导入task_registry
import torch  # 导入PyTorch库

if __name__ == "__main__":  # 如果该脚本是主程序执行
    logger.print_example_message()  # 打印示例消息以说明程序的用途
    start = time.time()  # 记录开始时间
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=16)  
    # 创建一个导航任务环境，设置为可视化模式（headless=False），并指定环境数量为16
    rl_task_env.reset()  # 重置环境，以便开始新的任务
    actions = torch.zeros(
        (rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)
    ).to("cuda:0")  # 创建一个全零的动作张量，维度为(环境数量, 动作空间维度)，并将其移动到GPU上
    actions[:, 0] = -1.0  # 将所有环境的第一个动作设为-1.0
    logger.info(
        "\n\n\n\n\n\n This script provides an example of the RL task interface with a zero action command in a cluttered environment."
    )  # 记录信息，说明此脚本提供了在复杂环境中使用零动作命令的RL任务接口示例
    logger.info(
        "This is to indicate the kind of interface that is available to the RL algorithm and the users for interacting with a Task environment.\n\n\n\n\n"
    )  # 记录信息，指明RL算法和用户与任务环境交互时可用的接口类型
    with torch.no_grad():  # 在不计算梯度的上下文中进行操作，以节省内存和加快速度
        for i in range(10000):  # 循环10000次
            if i == 100:  # 当循环计数器等于100时
                start = time.time()  # 重新记录开始时间
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)  
            # 执行一步操作，获取观察值、奖励、是否终止、是否截断以及其他信息
    end = time.time()  # 记录结束时间
