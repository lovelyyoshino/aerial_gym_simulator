#!/usr/bin/env python3

import rosbag
import csv
import numpy as np
from sensor_msgs.msg import Imu


def csv_to_imu_msgs(csv_file):
    """
    将CSV文件中的IMU数据转换为IMU消息格式。

    参数:
        csv_file (str): 输入的CSV文件路径，包含IMU传感器的数据。

    返回:
        List[Tuple[float, Imu]]: 包含时间戳和对应IMU消息的列表。
    """
    imu_msgs = []  # 存储IMU消息的列表
    timestamp = 0.0  # 初始化时间戳

    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file)  # 创建CSV读取器
        for row in csv_reader:
            # 提取时间步长和数据
            timestep = float(row[0])  # 从CSV中获取时间戳

            ax = float(row[1])  # 获取线性加速度x分量
            ay = float(row[2])  # 获取线性加速度y分量
            az = float(row[3])  # 获取线性加速度z分量
            gx = float(row[4])  # 获取角速度x分量
            gy = float(row[5])  # 获取角速度y分量
            gz = float(row[6])  # 获取角速度z分量

            imu_msg = Imu()  # 创建一个新的IMU消息对象
            imu_msg.header.stamp.secs = int(timestep)  # 设置秒数部分
            imu_msg.header.stamp.nsecs = int((timestep % 1) * 1e9)  # 设置纳秒部分
            imu_msg.header.frame_id = "imu_link"  # 设置坐标系ID

            # 填充IMU消息的线性加速度和角速度
            imu_msg.linear_acceleration.x = ax
            imu_msg.linear_acceleration.y = ay
            imu_msg.linear_acceleration.z = az
            imu_msg.angular_velocity.x = gx
            imu_msg.angular_velocity.y = gy
            imu_msg.angular_velocity.z = gz
            
            if timestep == int(timestep):  # 如果时间戳是整数，则打印出来
                print(timestep)

            imu_msgs.append((timestep, imu_msg))  # 将时间戳和IMU消息添加到列表中

    return imu_msgs  # 返回IMU消息列表


def write_to_rosbag(imu_msgs, bag_file):
    """
    将IMU消息写入rosbag文件。

    参数:
        imu_msgs (List[Tuple[float, Imu]]): 要写入的IMU消息列表。
        bag_file (str): 输出的rosbag文件路径。
    """
    with rosbag.Bag(bag_file, "w") as bag:  # 打开rosbag文件以写入模式
        for timestamp, msg in imu_msgs:  # 遍历每个IMU消息
            bag.write("imu/data", msg, msg.header.stamp)  # 写入IMU消息及其时间戳


if __name__ == "__main__":
    csv_file = "imu_data_2.csv"  # 替换为你的CSV文件路径
    bag_file = "output_imu_data_2.bag"  # 输出rosbag文件的名称

    print("Starting conversion process...")  # 开始转换过程的提示信息
    imu_msgs = csv_to_imu_msgs(csv_file)  # 调用函数将CSV转换为IMU消息
    write_to_rosbag(imu_msgs, bag_file)  # 将IMU消息写入rosbag文件

    print(f"Rosbag file '{bag_file}' has been created successfully.")  # 转换完成后的提示信息
