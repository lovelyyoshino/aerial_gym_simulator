import numpy as np
import random
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from PIL import Image
import matplotlib
import torch

if __name__ == "__main__":
    logger.warning("\n\n\nEnvironment to save a depth/range and segmentation image.\n\n\n")

    seed = 0
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 构建环境管理器，初始化仿真环境
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",  # "forest_env", #"empty_env", # empty_env
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
    seg_frames = []  # 存储分割图像的列表
    depth_frames = []  # 存储深度图像的列表
    merged_image_frames = []  # 存储合并图像的列表
    
    for i in range(10000):
        if i % 100 == 0 and i > 0:
            print("i", i)
            env_manager.reset()  # 每100次迭代重置环境
            
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
            
            # 单独保存每一帧图像
            seg_frames[0].save(f"seg_frame_{i}.png")
            depth_frames[0].save(f"depth_frame_{i}.png")
            merged_image_frames[0].save(f"merged_image_frame_{i}.png")
            seg_frames = []  # 清空存储分割图像的列表
            depth_frames = []  # 清空存储深度图像的列表
            merged_image_frames = []  # 清空存储合并图像的列表
        
        # 执行一步仿真
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")  # 渲染传感器组件
        
        # 重置已崩溃或被截断的环境
        env_manager.reset_terminated_and_truncated_envs()
        
        try:
            # 获取深度和分割图像
            image1 = (
                255.0 * env_manager.global_tensor_dict["depth_range_pixels"][0, 0].cpu().numpy()
            ).astype(np.uint8)
            seg_image1 = env_manager.global_tensor_dict["segmentation_pixels"][0, 0].cpu().numpy()
        except Exception as e:
            logger.error("Error in getting images")
            logger.error("Seems like the image tensors have not been created yet.")
            logger.error("This is likely due to absence of a functional camera in the environment")
            raise e
        
        # 对分割图像进行处理，将小于等于0的值替换为最小正值
        seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
        seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
            seg_image1.max() - seg_image1.min()
        )

        # 使用plasma色图对分割图像进行着色
        seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
        seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))
        depth_image1 = Image.fromarray(image1)
        
        # 创建一个4通道的图像（RGB + Alpha）
        image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
        image_4d[:, :, 0] = image1  # R通道
        image_4d[:, :, 1] = image1  # G通道
        image_4d[:, :, 2] = image1  # B通道
        image_4d[:, :, 3] = 255.0  # Alpha通道
        
        # 合并深度图像和分割图像
        merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
        
        # 将当前帧添加到各自的列表中
        seg_frames.append(seg_image1)
        depth_frames.append(depth_image1)
        merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))

        # # 可选：将帧保存为PNG文件
        # seg_image1.save(f"seg_image_{i}.png")
        # depth_image1.save(f"depth_image_{i}.png")
        # Image.fromarray(merged_image.astype(np.uint8)).save(f"merged_image_{i}.png")
