from abc import ABC, abstractmethod

# 定义一个抽象基类 BaseSensorConfig，继承自 ABC（Abstract Base Class）
class BaseSensorConfig(ABC):
    # 传感器数量，默认为1
    num_sensors = 1  
    # 是否随机放置传感器，默认为False
    randomize_placement = False  
    # 最小平移范围，表示在x、y、z轴上的最小位移
    min_translation = [0.07, -0.06, 0.01]  
    # 最大平移范围，表示在x、y、z轴上的最大位移
    max_translation = [0.12, 0.03, 0.04]  
    # 最小欧拉旋转角度（单位：度），表示在x、y、z轴上的最小旋转
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]  
    # 最大欧拉旋转角度（单位：度），表示在x、y、z轴上的最大旋转
    max_euler_rotation_deg = [5.0, 5.0, 5.0]  
