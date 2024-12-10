from abc import ABC

# 定义一个抽象基类 BaseAsset，继承自 ABC（Abstract Base Class）
class BaseAsset(ABC):
    # 初始化方法，接收资产名称、资产文件和加载选项作为参数
    def __init__(self, asset_name, asset_file, loading_options):
        self.name = asset_name  # 将资产名称赋值给实例变量 name
        self.file = asset_file   # 将资产文件赋值给实例变量 file
        # 将加载选项保存为类的实例，使用动态创建的 LoadingOptions 类
        self.options = type("LoadingOptions", (object,), loading_options)

    # 抽象方法 load_from_file，子类需要实现该方法以从文件中加载资产
    def load_from_file(self, asset_file):
        raise NotImplementedError("load_from_file not implemented")
