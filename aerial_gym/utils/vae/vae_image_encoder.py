import torch
import os
from aerial_gym.utils.vae.VAE import VAE


def clean_state_dict(state_dict):
    """
    清理状态字典，将模型权重中的特定前缀去除，以便于加载。
    
    参数:
        state_dict (dict): 包含模型权重的字典，可能包含多余的前缀。

    返回:
        dict: 清理后的状态字典。
    """
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")  # 去掉"module."前缀
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")  # 替换"dronet."为"encoder."
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    封装VAE类以实现高效推断的类，用于aerial_gym类。
    """

    def __init__(self, config, device="cuda:0"):
        """
        初始化VAE图像编码器，加载模型权重并设置设备。

        参数:
            config (object): 配置对象，包含模型参数和路径等信息。
            device (str): 指定使用的计算设备（默认为"cuda:0"）。
        """
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)  # 创建VAE模型
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)  # 构建权重文件路径
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))  # 加载并清理状态字典
        self.vae_model.load_state_dict(state_dict)  # 将权重加载到模型中
        self.vae_model.eval()  # 设置模型为评估模式

    def encode(self, image_tensors):
        """
        编码一组图像到潜在空间。可以返回均值和采样的潜在变量。

        参数:
            image_tensors (torch.Tensor): 输入的图像张量。

        返回:
            torch.Tensor: 编码后的潜在变量（均值或采样值）。
        """
        with torch.no_grad():  # 禁用梯度计算
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)  # 调整维度以适应VAE输入
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )  # 如果分辨率不匹配，则进行插值调整
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)  # 编码图像得到潜在变量

        if self.config.return_sampled_latent:
            returned_val = z_sampled  # 根据配置返回采样的潜在变量
        else:
            returned_val = means  # 否则返回均值
        return returned_val

    def decode(self, latent_spaces):
        """
        解码潜在空间以重构完整图像。

        参数:
            latent_spaces (torch.Tensor): 潜在空间张量。

        返回:
            torch.Tensor: 重构的图像。
        """
        with torch.no_grad():  # 禁用梯度计算
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )  # 检查潜在空间大小是否与网络一致
            decoded_image = self.vae_model.decode(latent_spaces)  # 解码潜在空间
        return decoded_image

    def get_latent_dims_size(self):
        """
        获取潜在空间的维度大小。

        返回:
            int: 潜在空间的维度大小。
        """
        return self.config.latent_dims
