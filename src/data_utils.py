# -*- coding: utf-8 -*-
"""
数据预处理模块

将 I/Q 两路时域复信号转换为适合 CNN 输入的二维表示，提供三种特征：
    方式1 (iq_channels): I/Q 双通道时域序列  → shape (2, 1024)
    方式2 (constellation): 星座图灰度图       → shape (1, 128, 128)
    方式3 (stft): STFT 时频幅度图             → shape (1, 64, 64)

默认使用方式1，可通过 feature_type 参数切换。
"""

import numpy as np
from scipy import signal as scipy_signal


# ==================== 特征提取函数 ====================

def extract_iq_channels(iq_data):
    """
    方式1：I/Q 双通道时域序列（默认）

    直接返回原始 I/Q 两路数据，shape 为 (N, 2, 1024)。
    这是最直接的特征表示，保留了完整的时域信息。
    """
    return iq_data.astype(np.float32)


def extract_constellation(iq_data, img_size=128, spread=2.0):
    """
    方式2：星座图灰度图

    原理：将 I/Q 采样点作为二维平面上的散点，
          累加到 128×128 的灰度图上，形成星座图图像。
          CNN 可以从星座图的几何形状中学习调制特征。

    参数:
        iq_data: (N, 2, L) I/Q 数据
        img_size: 输出图像边长（像素）
        spread:  坐标归一化范围（±spread 对应图像边界）
    返回:
        (N, 1, img_size, img_size) 灰度图
    """
    N = iq_data.shape[0]
    images = np.zeros((N, 1, img_size, img_size), dtype=np.float32)

    for n in range(N):
        I = iq_data[n, 0]
        Q = iq_data[n, 1]

        # 归一化到 [-spread, spread] → 映射到 [0, img_size-1]
        I_norm = np.clip((I + spread) / (2 * spread) * (img_size - 1),
                         0, img_size - 1)
        Q_norm = np.clip((Q + spread) / (2 * spread) * (img_size - 1),
                         0, img_size - 1)

        # 累加散点到图像（用直方图方式）
        hist, _, _ = np.histogram2d(
            I_norm, Q_norm,
            bins=[img_size, img_size],
            range=[[0, img_size - 1], [0, img_size - 1]]
        )
        # 归一化
        if hist.max() > 0:
            hist = hist / hist.max()
        images[n, 0] = hist

    return images


def extract_stft(iq_data, img_size=64, nperseg=32, noverlap=16):
    """
    方式3：STFT 时频图

    原理：短时傅里叶变换 (Short-Time Fourier Transform) 将时域信号
          转换为时频二维表示，能同时展现信号的时间和频率特征。
          不同调制方式在时频图上有不同的纹理模式。

    参数:
        iq_data:  (N, 2, L) I/Q 数据
        img_size: 输出图像边长
        nperseg:  STFT 每段长度
        noverlap: 段间重叠
    返回:
        (N, 1, img_size, img_size) 时频幅度图
    """
    N = iq_data.shape[0]
    images = np.zeros((N, 1, img_size, img_size), dtype=np.float32)

    for n in range(N):
        # 合并为复信号
        sig = iq_data[n, 0] + 1j * iq_data[n, 1]

        # STFT
        f, t, Zxx = scipy_signal.stft(
            sig, nperseg=nperseg, noverlap=noverlap
        )
        magnitude = np.abs(Zxx)

        # 调整大小到 img_size × img_size（用最近邻/面积插值）
        if magnitude.shape[0] != img_size or magnitude.shape[1] != img_size:
            # 简单的缩放：裁剪或填充
            mag_resized = np.zeros((img_size, img_size))
            h = min(magnitude.shape[0], img_size)
            w = min(magnitude.shape[1], img_size)
            mag_resized[:h, :w] = magnitude[:h, :w]
            magnitude = mag_resized

        # 对数尺度 + 归一化
        magnitude = np.log1p(magnitude)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        images[n, 0] = magnitude

    return images


# ==================== 特征提取分发器 ====================

def extract_features(iq_data, feature_type='iq_channels'):
    """
    特征提取统一接口

    参数:
        iq_data:      (N, 2, L) 原始 I/Q 数据
        feature_type: 'iq_channels' | 'constellation' | 'stft'
    返回:
        提取后的特征数组
    """
    if feature_type == 'iq_channels':
        return extract_iq_channels(iq_data)
    elif feature_type == 'constellation':
        return extract_constellation(iq_data)
    elif feature_type == 'stft':
        return extract_stft(iq_data)
    else:
        raise ValueError(f"未知特征类型: {feature_type}")


# ==================== 数据归一化 ====================

def normalize_per_sample(X):
    """
    逐样本归一化：每个样本独立做零均值、单位方差归一化

    适用于 I/Q 双通道时域数据。
    """
    N = X.shape[0]
    for n in range(N):
        mean = X[n].mean()
        std = X[n].std()
        if std > 1e-8:
            X[n] = (X[n] - mean) / std
    return X


# ==================== PyTorch Dataset ====================

def create_dataloader(X, y, batch_size=256, shuffle=True):
    """
    创建 PyTorch DataLoader

    参数:
        X:          特征数组 (numpy)
        y:          标签数组 (numpy)
        batch_size: 批大小
        shuffle:    是否打乱
    返回:
        torch.utils.data.DataLoader
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return dataloader


def get_input_shape(feature_type, total_samples=1024):
    """
    根据特征类型返回模型输入形状信息

    返回:
        (channels, signal_length) 用于 1D 模型
        或 (channels, height, width) 用于 2D 模型
    """
    if feature_type == 'iq_channels':
        return (2, total_samples)
    elif feature_type == 'constellation':
        return (1, 128, 128)
    elif feature_type == 'stft':
        return (1, 64, 64)
    else:
        raise ValueError(f"未知特征类型: {feature_type}")
