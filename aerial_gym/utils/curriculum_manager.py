import torch

class CurriculumManager:
    def __init__(self, num_envs, min_level, max_level, level_step, device="cuda:0"):
        # 初始化课程管理器
        # num_envs: 环境数量
        # min_level: 最小级别
        # max_level: 最大级别
        # level_step: 级别步长
        # device: 使用的设备，默认为"cuda:0"
        self.num_envs = num_envs
        self.min_level = min_level
        self.max_level = max_level
        self.level_step = level_step
        self.current_level = min_level  # 当前级别初始化为最小级别
        self.device = device
        self.level_list = self._create_level_list()  # 创建级别列表
        self.max_level_obtained = max(self.current_level, 0)  # 已获得的最大级别

    def _create_level_list(self):
        # 创建一个包含所有可用级别的列表
        level_list = []
        for i in range(self.min_level, self.max_level + 1, self.level_step):
            level_list.append(i)  # 将每个级别添加到列表中
        return level_list

    def increase_curriculum_level(self):
        # 增加当前课程级别
        self.current_level = min(self.current_level + self.level_step, self.max_level)  # 确保不超过最大级别
        self.max_level_obtained = max(self.current_level, self.max_level_obtained)  # 更新已获得的最大级别

    def get_current_level(self):
        # 获取当前课程级别
        return self.current_level

    def decrease_curriculum_level(self):
        # 减少当前课程级别
        self.current_level = max(self.current_level - self.level_step, self.min_level)  # 确保不低于最小级别
