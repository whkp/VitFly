"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import getpass
uname = getpass.getuser()

# 获取当前用户名
uname = getpass.getuser()

def dataloader(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None):
    """
    数据加载函数，用于加载图像和对应的元数据，并划分为训练集和验证集。

    参数:
        data_dir (str): 数据目录路径。
        val_split (float): 验证集比例。
        short (int): 数据集的子集大小，如果大于0，则只加载前short个文件夹。
        seed (int): 随机种子，用于打乱文件夹顺序。
        train_val_dirs (tuple): 包含训练和验证文件夹列表的元组。

    返回:
        tuple: 包含训练集和验证集的数据，以及其他辅助信息。
    """

    # 定义裁剪后的图像高度和宽度
    cropHeight = 60
    cropWidth = 90

    # 如果提供了train_val_dirs，则使用指定的训练和验证文件夹，并计算验证集比例
    if train_val_dirs is not None:
        traj_folders = train_val_dirs[0] + train_val_dirs[1]  # 合并训练和验证文件夹列表
        val_split = len(train_val_dirs[1]) / len(traj_folders)  # 计算验证集比例
    else:
        # 否则，从data_dir中获取所有子文件夹，并按字母顺序排序
        traj_folders = sorted(glob.glob(opj(data_dir, '*')))
        # 设置随机种子以确保结果可复现
        random.seed(seed)
        # 随机打乱文件夹顺序
        random.shuffle(traj_folders)

    # 如果short参数大于0，则只取前short个文件夹
    if short > 0:
        assert short <= len(traj_folders), f"short={short} 大于文件夹总数={len(traj_folders)}"
        traj_folders = traj_folders[:short]

    # 初始化用于存储所需速度、图像、元数据和当前四元数的列表
    desired_vels = []
    traj_ims_full = []
    traj_meta_full = []
    curr_quats = []    

    # 开始加载数据，记录开始时间
    start_dataloading = time.time()

    # 初始化跳过的图像和文件夹计数器
    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    # 遍历所有轨迹文件夹
    for i, traj_folder in enumerate(traj_folders):
        # 每加载10%的文件夹，打印一次加载进度
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] 正在加载文件夹 {os.path.basename(traj_folder)}, 文件夹编号 {i+1}/{len(traj_folders)}, 已用时间 {time.time()-start_dataloading:.2f}s')
        
        # 获取当前文件夹中所有PNG图像文件，按字母顺序排序
        im_files = sorted(glob.glob(opj(traj_folder, '*.png')))

        # 检查文件夹是否为空
        if len(im_files) == 0:
            print(f'[DATALOADER] 文件夹 {os.path.basename(traj_folder)} 中没有图像，跳过')
            continue

        # 定义元数据CSV文件名
        csv_file = 'data.csv'
        # 读取CSV文件，跳过第一行（假设是标题），并指定数据类型为float64
        traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
        # 将最后一列转换为int32布尔类型
        traj_meta[:,-1] = np.int32(np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype="bool")[1:,-1])

        # 检查轨迹中是否有碰撞，如果有则跳过该文件夹（此部分被注释掉，可以根据需要启用）
        # if traj_meta[:,-1].sum() > 0:
        #     print(f'[DATALOADER] 轨迹 {os.path.basename(traj_folder)} 中有碰撞，跳过')
        #     collisionFolders += 1
        #     collisionImages += int(len(traj_meta[:,0]))
        #     continue

        # 检查元数据中是否有NaN值
        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] 轨迹 {os.path.basename(traj_folder)} 中有NaN值，跳过')
            traj_meta = traj_meta[:,:-1]  # 移除最后一列

        # 读取PNG图像文件，并将像素值缩放到0到1的范围
        traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0

        # 检查图像数量和元数据条目数是否匹配
        if traj_ims.shape[0] != traj_meta.shape[0]:
            # 通常最后一张图像可能没有对应的元数据条目，检查这种情况
            last_im_timestamp = os.path.basename(im_files[-1])[:-4]
            if float(last_im_timestamp) > traj_meta[-1, 1]:
                traj_ims = traj_ims[:-1]  # 移除最后一张多余的图像
                print(f'[DATALOADER] 轨迹 {os.path.basename(traj_folder)} 末尾发现多余图像，已移除')
            # 再次检查是否匹配
            if traj_ims.shape[0] != traj_meta.shape[0]:
                print(f'[DATALOADER] 轨迹 {os.path.basename(traj_folder)} 中图像数量与元数据条目数仍不匹配，跳过')
                skippedFolders += 1
                skippedImages += int(len(traj_meta[:,0]))
                continue

        # 对所有图像进行裁剪和调整大小
        temp = [cv2.resize(img, (cropWidth, cropHeight)) for img in traj_ims]
        
        traj_ims = np.array(temp)

        # 遍历每个元数据条目，提取所需速度和四元数
        for ii in range(traj_meta.shape[0]):
            desired_vels.append(traj_meta[ii, 2])  # 第3列为所需速度
            q = traj_meta[ii, 3:7]  # 第4到7列为四元数
            rmat = q  # 这里可以根据需要转换为旋转矩阵
            curr_quats.append(rmat)

        try:
            traj_ims_full.append(traj_ims)  # 添加图像到总列表
            traj_meta_full.append(traj_meta)  # 添加元数据到总列表
        except:
            print(f'[DATALOADER] 图像形状 {traj_ims.shape}')
            print(f"[DATALOADER] 可能存在空图像，文件夹 {os.path.basename(traj_folder)}")

    # 打印跳过的文件夹和图像数量
    print("跳过的文件夹数: %d, 跳过的图像数: %d"%(skippedFolders, skippedImages))
    print("碰撞的文件夹数: %d, 碰撞的图像数: %d"%(collisionFolders, collisionImages))

    print("[ANALYZER] 正在分析数据....")
    # 计算每条轨迹的长度
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    print("轨迹长度: {}".format(traj_lengths))
    # 将所有图像和元数据合并为一个大数组
    traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    desired_vels = np.array(desired_vels)
    curr_quats = np.array(curr_quats)

    # 计算并标准化ctbr（可能是某种特征）的均值和标准差
    # 行: ct, brx, bry, brz
    # 列: 均值, 标准差
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta_full[:, 16]), np.std(traj_meta_full[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta_full[:, 17]), np.std(traj_meta_full[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta_full[:, 18]), np.std(traj_meta_full[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta_full[:, 19]), np.std(traj_meta_full[:, 19])

    # 对ctbr进行标准化处理
    traj_meta_full[:, 16] = (traj_meta_full[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta_full[:, 17] = (traj_meta_full[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta_full[:, 18] = (traj_meta_full[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta_full[:, 19] = (traj_meta_full[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    # 提取标准化后的ctbr
    curr_ctbr = traj_meta_full[:, 16:20]

    # 再次计算并标准化单个文件夹的ctbr
    # 行: ct, brx, bry, brz
    # 列: 均值, 标准差
    stats_ctbr = np.zeros((4, 2))
    stats_ctbr[0, :] = np.mean(traj_meta[:, 16]), np.std(traj_meta[:, 16])
    stats_ctbr[1, :] = np.mean(traj_meta[:, 17]), np.std(traj_meta[:, 17])
    stats_ctbr[2, :] = np.mean(traj_meta[:, 18]), np.std(traj_meta[:, 18])
    stats_ctbr[3, :] = np.mean(traj_meta[:, 19]), np.std(traj_meta[:, 19])

    # 对单个轨迹的ctbr进行标准化处理
    traj_meta[:, 16] = (traj_meta[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1])
    traj_meta[:, 17] = (traj_meta[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1])
    traj_meta[:, 18] = (traj_meta[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1])
    traj_meta[:, 19] = (traj_meta[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1])

    # 根据val_split参数划分训练集和验证集
    num_val_trajs = int(val_split * len(traj_lengths))  # 验证集轨迹数量
    val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)  # 验证集在合并后的数据中的索引
    traj_meta_val = traj_meta_full[:val_idx]  # 验证集的元数据
    traj_meta_train = traj_meta_full[val_idx:]  # 训练集的元数据
    traj_ims_val = traj_ims_full[:val_idx]  # 验证集的图像
    traj_ims_train = traj_ims_full[val_idx:]  # 训练集的图像
    traj_lengths_val = traj_lengths[:num_val_trajs]  # 验证集的轨迹长度
    traj_lengths_train = traj_lengths[num_val_trajs:]  # 训练集的轨迹长度
    desired_vels_val = desired_vels[:val_idx]  # 验证集的所需速度
    desired_vels_train = desired_vels[val_idx:]  # 训练集的所需速度
    # curr_vels_val = curr_vels[:val_idx]  # 当前速度（注释掉）
    # curr_vels_train = curr_vels[val_idx:]
    curr_quats_val = curr_quats[:val_idx]  # 验证集的当前四元数
    curr_quats_train = curr_quats[val_idx:]  # 训练集的当前四元数
    curr_ctbr_val = curr_ctbr[:val_idx]  # 验证集的ctbr
    curr_ctbr_train = curr_ctbr[val_idx:]  # 训练集的ctbr

    # 注意，我们返回is_png=1标志，因为它指示旧数据集与新数据集，这决定了如何解析元数据
    # 我们还返回训练集和验证集的轨迹文件夹名称，以便可以保存并稍后用于生成特定的评估图表
    return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train), \
           (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val), \
           1, \
           (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs])

def parse_meta_str(meta_str):
    """
    解析元数据字符串的函数。

    参数:
        meta_str (str): 元数据字符串。

    返回:
        torch.Tensor: 解析后的元数据张量。
    """
    # 创建一个与meta_str相同形状的零张量
    meta = torch.zeros_like(meta_str)

    # 这里应添加具体的解析逻辑

    return meta

def preload(items, device='cpu'):
    """
    预加载数据，将numpy数组转换为PyTorch张量，并移动到指定设备。

    参数:
        items (list): 包含numpy数组的列表。
        device (str): 目标设备，默认为'cpu'。

    返回:
        list: 包含PyTorch张量的列表。
    """
    # 遍历每个项目，将其转换为PyTorch张量并移动到指定设备
    return [torch.from_numpy(item).to(device).float() for item in items]