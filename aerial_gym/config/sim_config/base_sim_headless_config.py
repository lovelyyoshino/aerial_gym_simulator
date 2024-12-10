from .base_sim_config import BaseSimConfig  # 从同一目录下导入BaseSimConfig类

# 定义一个名为BaseSimHeadlessConfig的类，继承自BaseSimConfig
class BaseSimHeadlessConfig(BaseSimConfig):
    # 在BaseSimHeadlessConfig中定义一个内部类viewer，继承自BaseSimConfig.viewer
    class viewer(BaseSimConfig.viewer):
        headless = True  # 设置headless属性为True，表示无头模式（不显示图形界面）
