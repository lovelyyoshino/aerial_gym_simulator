from isaacgym import gymapi
from isaacgym import gymutil

import distutils


def class_to_dict(obj) -> dict:
    """
    将对象的属性转换为字典形式。
    
    参数:
        obj: 要转换的对象。

    返回:
        dict: 对象的属性字典表示。
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):  # 跳过私有属性
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):  # 如果属性是列表，递归处理每个元素
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)  # 递归处理非列表属性
        result[key] = element
    return result


def parse_sim_params(args, cfg):
    """
    解析并初始化仿真参数。

    参数:
        args: 命令行参数。
        cfg: 配置文件内容。

    返回:
        sim_params: 初始化后的仿真参数对象。
    """
    # 从 Isaac Gym Preview 2 的代码中获取
    sim_params = gymapi.SimParams()  # 创建一个新的仿真参数实例

    # 根据命令行参数设置一些值
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")  # 警告信息
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu  # 设置是否使用GPU
        sim_params.physx.num_subscenes = args.subscenes  # 设置子场景数量
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline  # 设置是否使用GPU管道

    # 如果配置文件中提供了仿真选项，则解析并更新/覆盖上述设置
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果在命令行中传入了线程数，则覆盖默认值
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params  # 返回最终的仿真参数


def update_cfg_from_args(cfg, args):
    """
    根据命令行参数更新配置字典。

    参数:
        cfg: 当前配置字典。
        args: 命令行参数。

    返回:
        cfg: 更新后的配置字典。
    """
    if cfg is None:
        raise ValueError("cfg is None")  # 检查配置是否为空
    if args.headless is not None:
        cfg["viewer"]["headless"] = args.headless  # 更新视图模式
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs  # 更新环境数量
    return cfg  # 返回更新后的配置


def get_args(additional_parameters=[]):
    """
    获取命令行参数。

    参数:
        additional_parameters: 附加的自定义参数。

    返回:
        args: 解析后的命令行参数对象。
    """
    custom_parameters = [
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),  # 转换为布尔值
            "default": False,
            "help": "Force display off at all times",  # 帮助信息
        },
        {
            "name": "--num_envs",
            "type": int,  # 类型为整数
            "default": "64",
            "help": "Number of environments to create. Overrides config file if provided.",  # 帮助信息
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),  # 转换为布尔值
            "default": True,
            "help": "Use warp for rendering",  # 帮助信息
        },
    ]
    # 解析命令行参数
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters + additional_parameters,
    )

    # 名称对齐
    args.sim_device_id = args.compute_device_id  # 设置设备ID
    args.sim_device = args.sim_device_type  # 设置设备类型
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"  # 如果是CUDA设备，添加设备ID
    return args  # 返回解析后的参数


def asset_class_to_AssetOptions(asset_class):
    """
    将资产类转换为资产选项。

    参数:
        asset_class: 输入的资产类对象。

    返回:
        asset_options: 转换后的资产选项对象。
    """
    asset_options = gymapi.AssetOptions()  # 创建资产选项实例
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints  # 设置固定关节合并选项
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule  # 替换圆柱体为胶囊体
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments  # 翻转视觉附着物
    asset_options.fix_base_link = asset_class.fix_base_link  # 固定基础链接
    asset_options.density = asset_class.density  # 设置密度
    asset_options.angular_damping = asset_class.angular_damping  # 设置角阻尼
    asset_options.linear_damping = asset_class.linear_damping  # 设置线性阻尼
    asset_options.max_angular_velocity = asset_class.max_angular_velocity  # 设置最大角速度
    asset_options.max_linear_velocity = asset_class.max_linear_velocity  # 设置最大线速度
    asset_options.disable_gravity = asset_class.disable_gravity  # 禁用重力
    return asset_options  # 返回资产选项对象
