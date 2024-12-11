from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, path):
        super().__init__()
        # 定义输入层到隐藏层的全连接层，输入维度为input_dim，输出维度为256
        self.input_fc = nn.Linear(input_dim, 256)
        # 定义第一个隐藏层，全连接层，输入维度为256，输出维度为128
        self.hidden_fc1 = nn.Linear(256, 128)
        # 定义第二个隐藏层，全连接层，输入维度为128，输出维度为64
        self.hidden_fc2 = nn.Linear(128, 64)
        # 定义输出层，全连接层，输入维度为64，输出维度为output_dim
        self.output_fc = nn.Linear(64, output_dim)

        # 使用有序字典构建网络结构
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("0", self.input_fc),      # 输入层
                    ("elu1", nn.ELU()),        # 第一个激活函数（ELU）
                    ("2", self.hidden_fc1),    # 第一个隐藏层
                    ("elu2", nn.ELU()),        # 第二个激活函数（ELU）
                    ("4", self.hidden_fc2),    # 第二个隐藏层
                    ("elu3", nn.ELU()),        # 第三个激活函数（ELU）
                    ("mu", self.output_fc),     # 输出层
                ]
            )
        )
        # 从指定路径加载预训练模型参数
        self.load_network(path)

    def load_network(self, path):
        # 加载模型状态字典
        sd = torch.load(path)["model"]

        # 清理状态字典并加载它
        od2 = OrderedDict()
        for key in sd:
            # 替换键名以匹配当前模型的结构
            key2 = str(key).replace("a2c_network.actor_mlp.", "")
            key2 = key2.replace("a2c_network.", "")
            # 跳过不需要的键
            if "a2c_network" in key2 or "value" in key2 or "sigma" in key2:
                continue
            else:
                print(key2)  # 打印有效的键名
                od2[key2] = sd[str(key)]  # 将清理后的键值对添加到新的有序字典中
        # 严格加载状态字典
        self.network.load_state_dict(od2, strict=True)
        print("Loaded MLP network from {}".format(path))  # 打印加载成功的信息

    def forward(self, x):
        # 前向传播方法，将输入x传入网络并返回输出
        return self.network(x)
