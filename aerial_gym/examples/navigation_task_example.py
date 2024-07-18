import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task("navigation_task", headless=False, num_envs=16)
    rl_task_env.reset()
    actions = torch.zeros(
        (rl_task_env.task_config.num_envs, rl_task_env.task_config.action_space_dim)
    ).to("cuda:0")
    actions[:, 0] = -1.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()