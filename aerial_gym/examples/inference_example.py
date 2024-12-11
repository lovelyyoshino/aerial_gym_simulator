import time
from aerial_gym.utils.logging import CustomLogger

from aerial_gym.sim2real.sample_factory_inference import RL_Nav_Interface

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch


class EMA:
    def __init__(self, beta):
        # 初始化指数移动平均（EMA）类
        # beta: 平滑因子，取值范围在0到1之间
        self.beta = beta
        self.average = None  # 存储当前的平均值

    def update(self, value):
        # 更新EMA的平均值
        # value: 新输入的值
        if self.average is None:
            self.average = value  # 如果没有初始平均值，则直接赋值
        else:
            # 使用公式更新平均值
            self.average = (1 - self.beta) * self.average + self.beta * value
        return self.average  # 返回更新后的平均值


num_envs = 16  # 设置环境数量为16

if __name__ == "__main__":
    logger.warning(
        "\n\nExample file simulating a Sample Factory trained policy in cluttered environments using a Task Definition for navigation."
    )
    logger.warning(
        "Usage: python3 inference_example.py --env=navigation_task --experiment=lmf2_sim2real_241024 --train_dir=../sim2real/weights --load_checkpoint_kind=best"
    )
    logger.warning(
        "Please make sure a camera sensor is enabled on the robot as per specifications of the task.\n\n"
    )
    
    start = time.time()  # 记录开始时间
    
    # 创建导航任务环境
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=num_envs)

    # 初始化强化学习模型接口
    rl_model = RL_Nav_Interface(num_envs=num_envs)
    action_filter = EMA(0.8)  # 创建EMA实例，用于平滑动作输出

    rl_task_env.reset()  # 重置环境
    actions = torch.zeros(
        (rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)
    ).to("cuda:0")  # 在GPU上创建一个零张量用于存放动作
    actions[:, 0] = -1.0  # 将第一个动作维度初始化为-1.0

    with torch.no_grad():  # 禁用梯度计算以节省内存和加快速度
        for i in range(10000):  # 主循环，最多执行10000次
            if i == 100:
                start = time.time()  # 每100步重置计时器
            
            for j in range(5):  # 每个主循环中进行5次环境步骤
                obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)  # 执行动作并获取反馈
            
            # 更新观察值使用EMA平滑处理
            obs["observations"] = action_filter.update(obs["observations"])
            
            # 检查是否有需要重置的环境
            reset_list = (terminated + truncated).nonzero().squeeze().tolist()
            if ((type(reset_list) is int) and (reset_list > 0)) or len(reset_list) > 0:
                rl_model.reset(reset_list)  # 重置指定的环境
            
            actions[:] = rl_model.step(obs=obs)  # 根据当前观察值生成新的动作
        
    end = time.time()  # 记录结束时间
