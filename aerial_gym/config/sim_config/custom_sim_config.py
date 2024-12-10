"""
这是一个自定义的仿真配置文件。用于定义仿真参数。
"""

from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class CustomSimConfig(BaseSimConfig):
    class sim(BaseSimConfig.sim):
        dt = 0.001  # 自定义参数，仿真的时间步长（秒）
        gravity = [+1.0, 0.0, 0.0]  # [m/s^2] 自定义参数，重力加速度向量

        class physx(BaseSimConfig.sim.physx):
            num_threads = 5  # 自定义参数，物理引擎使用的线程数
            solver_type = 1  # 求解器类型：0表示PGS，1表示TGS
            num_position_iterations = 10  # 自定义参数，位置求解迭代次数
            num_velocity_iterations = 15  # 自定义参数，速度求解迭代次数
            contact_offset = 0.01  # [m] 自定义参数，接触偏移量
            rest_offset = 0.01  # [m] 自定义参数，静止偏移量
            bounce_threshold_velocity = 0.5  # [m/s] 自定义参数，反弹阈值速度
            max_depenetration_velocity = 1.0  # 最大穿透速度
            max_gpu_contact_pairs = 2**20  # 自定义参数，最大GPU接触对数量
            default_buffer_size_multiplier = 5  # 默认缓冲区大小乘数
            contact_collection = 0  # 接触收集策略：0表示从不，1表示最后一次子步骤，2表示所有子步骤（默认=2）
