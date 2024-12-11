import torch
import torch.nn as nn


class ImgDecoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, with_logits=False):
        """
        Parameters
        ----------
        latent_dim: int
            The latent dimension.
        """

        super(ImgDecoder, self).__init__()
        print("[ImgDecoder] Starting create_model")
        self.with_logits = with_logits  # 是否返回logits值
        self.n_channels = input_dim  # 输入图像的通道数
        self.dense = nn.Linear(latent_dim, 512)  # 全连接层，将潜在维度映射到512维
        self.dense1 = nn.Linear(512, 9 * 15 * 128)  # 将512维映射到特定形状以便后续卷积操作
        # Pytorch文档：output_padding仅用于查找输出形状，但实际上并不向输出添加零填充
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)  # 转置卷积层
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=(2, 2), output_padding=(0, 1), dilation=1
        )  # 转置卷积层，逐步上采样
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=6, stride=4, padding=(2, 2), output_padding=(0, 0), dilation=1
        )  # 转置卷积层
        self.deconv6 = nn.ConvTranspose2d(
            32, 16, kernel_size=6, stride=2, padding=(0, 0), output_padding=(0, 1)
        )  # 转置卷积层
        self.deconv7 = nn.ConvTranspose2d(
            16, self.n_channels, kernel_size=4, stride=2, padding=2
        )  # 最后一层转置卷积，用于生成最终输出图像
        print("[ImgDecoder] Done with create_model")
        print("Defined decoder.")

    def forward(self, z):
        return self.decode(z)  # 前向传播调用解码函数

    def decode(self, z):
        x = self.dense(z)  # 对输入z进行全连接变换
        x = torch.relu(x)  # ReLU激活函数
        x = self.dense1(x)  # 再次全连接变换
        x = x.view(x.size(0), 128, 9, 15)  # 调整张量形状为适合卷积操作的格式

        x = self.deconv1(x)  # 第一个转置卷积
        x = torch.relu(x)  # 激活

        x = self.deconv2(x)  # 第二个转置卷积
        x = torch.relu(x)  # 激活

        x = self.deconv4(x)  # 第三个转置卷积
        x = torch.relu(x)  # 激活

        x = self.deconv6(x)  # 第四个转置卷积
        x = torch.relu(x)  # 激活

        x = self.deconv7(x)  # 最后一个转置卷积
        # print(f"- After deconv 7, mean: {x.mean():.3f} var: {x.var():.3f}")
        if self.with_logits:
            return x  # 如果with_logits为True，直接返回未经过sigmoid处理的结果

        x = torch.sigmoid(x)  # 否则应用sigmoid激活函数
        # print(f"- After sigmoid, mean: {x.mean():.3f} var: {x.var():.3f}")
        return x  # 返回最终生成的图像


class ImgEncoder(nn.Module):
    """
    ResNet8 architecture as encoder.
    """

    def __init__(self, input_dim, latent_dim):
        """
        Parameters:
        ----------
        input_dim: int
            Number of input channels in the image.
        latent_dim: int
            Number of latent dimensions
        """
        super(ImgEncoder, self).__init__()
        self.input_dim = input_dim  # 输入图像的通道数
        self.latent_dim = latent_dim  # 潜在空间的维度
        self.define_encoder()  # 定义编码器结构
        self.elu = nn.ELU()  # 使用ELU激活函数
        print("Defined encoder.")

    def define_encoder(self):
        # 定义卷积层
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)  # 第一层卷积
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)  # 第二层卷积
        nn.init.xavier_uniform_(self.conv0_1.weight, gain=nn.init.calculate_gain("linear"))  # 权重初始化
        nn.init.zeros_(self.conv0_1.bias)  # 偏置初始化

        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)  # 第三层卷积
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 第四层卷积
        nn.init.xavier_uniform_(self.conv1_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv1_1.bias)

        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  # 第五层卷积
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 第六层卷积
        nn.init.xavier_uniform_(self.conv2_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv2_1.bias)

        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2)  # 第七层卷积

        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 跳跃连接
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))  # 跳跃连接

        self.dense0 = nn.Linear(3 * 6 * 128, 512)  # 全连接层，将最后的特征映射到512维
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)  # 输出潜在空间的均值和对数方差

        print("Encoder network initialized.")

    def forward(self, img):
        return self.encode(img)  # 前向传播调用编码函数

    def encode(self, img):
        """
        Encodes the input image.
        """

        # conv0
        x0_0 = self.conv0(img)  # 第一层卷积
        x0_1 = self.conv0_1(x0_0)  # 第二层卷积
        x0_1 = self.elu(x0_1)  # ELU激活

        x1_0 = self.conv1_0(x0_1)  # 第三层卷积
        x1_1 = self.conv1_1(x1_0)  # 第四层卷积

        x0_jump_2 = self.conv0_jump_2(x0_1)  # 跳跃连接

        x1_1 = x1_1 + x0_jump_2  # 残差连接

        x1_1 = self.elu(x1_1)  # 激活

        x2_0 = self.conv2_0(x1_1)  # 第五层卷积
        x2_1 = self.conv2_1(x2_0)  # 第六层卷积

        x1_jump3 = self.conv1_jump_3(x1_1)  # 跳跃连接

        x2_1 = x2_1 + x1_jump3  # 残差连接

        x2_1 = self.elu(x2_1)  # 激活

        x3_0 = self.conv3_0(x2_1)  # 第七层卷积

        x = x3_0.view(x3_0.size(0), -1)  # 展平张量

        x = self.dense0(x)  # 全连接层
        x = self.elu(x)  # 激活
        x = self.dense1(x)  # 输出均值和对数方差
        return x  # 返回潜在表示


class Lambda(nn.Module):
    """Lambda function that accepts tensors as input."""

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func  # 存储传入的函数

    def forward(self, x):
        return self.func(x)  # 执行存储的函数


class VAE(nn.Module):
    """Variational Autoencoder for reconstruction of depth images."""

    def __init__(self, input_dim=1, latent_dim=64, with_logits=False, inference_mode=False):
        """
        Parameters
        ----------
        input_dim: int
            The number of input channels in an image.
        latent_dim: int
            The latent dimension.
        """

        super(VAE, self).__init__()

        self.with_logits = with_logits  # 是否返回logits值
        self.input_dim = input_dim  # 输入图像的通道数
        self.latent_dim = latent_dim  # 潜在空间的维度
        self.inference_mode = inference_mode  # 推理模式标志
        self.encoder = ImgEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)  # 初始化编码器
        self.img_decoder = ImgDecoder(
            input_dim=1, latent_dim=self.latent_dim, with_logits=self.with_logits
        )  # 初始化解码器

        self.mean_params = Lambda(lambda x: x[:, : self.latent_dim])  # 均值参数提取
        self.logvar_params = Lambda(lambda x: x[:, self.latent_dim :])  # 对数方差参数提取

    def forward(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # 编码
        z = self.encoder(img)

        # 重参数化技巧
        mean = self.mean_params(z)  # 提取均值
        logvar = self.logvar_params(z)  # 提取对数方差
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
        if self.inference_mode:
            eps = torch.zeros_like(eps)  # 在推理模式下，不使用噪声
        z_sampled = mean + eps * std  # 重新采样潜在变量

        # 解码
        img_recon = self.img_decoder(z_sampled)  # 通过解码器生成重建图像
        return img_recon, mean, logvar, z_sampled  # 返回重建图像、均值、对数方差和采样的潜在变量

    def forward_test(self, img):
        """Do a forward pass of the VAE. Generates a reconstructed image based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """

        # 编码
        z = self.encoder(img)

        # 重参数化技巧
        mean = self.mean_params(z)  # 提取均值
        logvar = self.logvar_params(z)  # 提取对数方差
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
        if self.inference_mode:
            eps = torch.zeros_like(eps)  # 在推理模式下，不使用噪声
        z_sampled = mean + eps * std  # 重新采样潜在变量

        # 解码
        img_recon = self.img_decoder(z_sampled)  # 通过解码器生成重建图像
        return img_recon, mean, logvar, z_sampled  # 返回重建图像、均值、对数方差和采样的潜在变量

    def encode(self, img):
        """Do a forward pass of the VAE. Generates a latent vector based on img
        Parameters
        ----------
        img: torch.Tensor
            The input image.
        """
        z = self.encoder(img)  # 编码输入图像

        means = self.mean_params(z)  # 提取均值
        logvars = self.logvar_params(z)  # 提取对数方差
        std = torch.exp(0.5 * logvars)  # 计算标准差
        eps = torch.randn_like(logvars)  # 生成与对数方差相同形状的随机噪声
        if self.inference_mode:
            eps = torch.zeros_like(eps)  # 在推理模式下，不使用噪声
        z_sampled = means + eps * std  # 重新采样潜在变量

        return z_sampled, means, std  # 返回采样的潜在变量、均值和标准差

    def decode(self, z):
        """Do a forward pass of the VAE. Generates a reconstructed image based on z
        Parameters
        ----------
        z: torch.Tensor
            The latent vector.
        """
        img_recon = self.img_decoder(z)  # 通过解码器生成重建图像
        if self.with_logits:
            return torch.sigmoid(img_recon)  # 如果需要logits，则应用sigmoid
        return img_recon  # 返回重建图像

    def set_inference_mode(self, mode):
        self.inference_mode = mode  # 设置推理模式
