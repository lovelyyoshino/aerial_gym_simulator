import numpy as np
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch

seed = 0

if __name__ == "__main__":
    logger.warning("\n\n\nEnvironment to save a normal and faceid image.\n\n\n")

    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 构建环境管理器，初始化仿真环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="forest_env",  # "forest_env", #"empty_env", # empty_env
        robot_name="base_quadrotor",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=2,
        headless=False,
        use_warp=True,
    )
    
    # 初始化动作张量，大小为(num_envs, 4)，用于控制机器人
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")

    # 重置环境
    env_manager.reset()
    seg_frames = []  # 存储分割图像帧
    depth_frames = []  # 存储深度图像帧
    merged_image_frames = []  # 存储合并图像帧
    
    for i in range(101):
        if i % 100 == 0 and i > 0:
            print("i", i)
            env_manager.reset()  # 每100步重置环境
            
            # 保存帧为GIF格式
            seg_frames[0].save(
                f"seg_frames_{i}.gif",
                save_all=True,
                append_images=seg_frames[1:],
                duration=100,
                loop=0,
            )
            depth_frames[0].save(
                f"depth_frames_{i}.gif",
                save_all=True,
                append_images=depth_frames[1:],
                duration=100,
                loop=0,
            )
            merged_image_frames[0].save(
                f"merged_image_frames_{i}.gif",
                save_all=True,
                append_images=merged_image_frames[1:],
                duration=100,
                loop=0,
            )
            seg_frames = []  # 清空存储的帧
            depth_frames = []
            merged_image_frames = []

        # 执行一步仿真
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")  # 渲染传感器组件
        
        # 重置已崩溃或被截断的环境
        env_manager.reset_terminated_and_truncated_envs()
        
        try:
            # 定义一个方向向量，用于计算点积
            one_vec = torch.zeros_like(env_manager.global_tensor_dict["depth_range_pixels"])
            one_vec[..., 0] = 1.0
            one_vec[..., 1] = 1 / 2.0
            one_vec[..., 2] = 1 / 3.0
            one_vec[:] = one_vec / torch.norm(one_vec, dim=-1, keepdim=True)  # 单位化向量
            
            # 计算与深度范围像素的余弦相似度
            cosine_vec = torch.abs(
                torch.sum(one_vec * env_manager.global_tensor_dict["depth_range_pixels"], dim=-1)
            )

            # 将余弦值转换为图像格式
            image1 = (255.0 * cosine_vec)[0, 0].cpu().numpy().astype(np.uint8)

            # 获取分割图像
            seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e
        
        # 对分割图像进行归一化处理
        seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
        seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
            seg_image1.max() - seg_image1.min()
        )

        # 使用plasma色图对分割图像进行着色
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
        seg_image1_plasma_int = (255.0 * seg_image1_normalized_plasma).astype(np.uint8)
        mod_image = 10 * np.mod(seg_image1_plasma_int, 25).astype(np.uint8)
        # 设置通道为不透明
        mod_image[:, :, 3] = 255
        seg_image1_discrete = Image.fromarray(mod_image)  # 创建离散分割图像
        seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))  # 创建彩色分割图像

        depth_image1 = Image.fromarray(image1)  # 创建深度图像
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1  # R通道
        image_4d[:, :, 1] = image1  # G通道
        image_4d[:, :, 2] = image1  # B通道
        image_4d[:, :, 3] = 255.0  # Alpha通道
        
        # 合并图像，将深度图和分割图像结合在一起
        merged_image = np.concatenate((image_4d, seg_image1_discrete), axis=0)
        
        # 将当前帧添加到数组中
        seg_frames.append(seg_image1_discrete)
        depth_frames.append(depth_image1)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
