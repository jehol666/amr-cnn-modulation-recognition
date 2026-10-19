# -*- coding: utf-8 -*-
"""
CNN 模型定义 —— 基于 PyTorch

包含两种模型：
    1. CNN1D: 4 个 1D 卷积块 + 2 个全连接层（主模型）
    2. ResNet1D: 带残差连接的 1D ResNet 变体（对比模型）

主模型结构（输入 2×1024）：
    Conv1d(2→64,  k=7) → BN → ReLU → MaxPool(2)   → 512
    Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)  → 256
    Conv1d(128→256,k=5) → BN → ReLU → MaxPool(2)  → 128
    Conv1d(256→256,k=3) → BN → ReLU → MaxPool(2)  → 64
    Flatten → FC(256*64→256) → ReLU → Dropout(0.5)
            → FC(256→128) → ReLU → Dropout(0.5)
            → FC(128→10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 主模型：CNN1D ====================

class Conv1DBlock(nn.Module):
    """
    1D 卷积块：Conv1d + BatchNorm + ReLU + MaxPool

    这是 CNN 的基本构建单元。BatchNorm 加速收敛并提升泛化能力，
    MaxPool 逐步降低序列长度以减少计算量并扩大感受野。
    """

    def __init__(self, in_ch, out_ch, kernel_size, pool_size=2):
        super().__init__()
        # padding='same' 保持卷积后长度不变（由 MaxPool 负责降维）
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNN1D(nn.Module):
    """
    一维卷积神经网络（主模型）

    4 个卷积块提取时域特征，2 个全连接层完成分类。
    输入: (batch, 2, 1024)  I/Q 双通道时域信号
    输出: (batch, 10)       10 类调制类型的 logits
    """

    def __init__(self, in_channels=2, num_classes=10, dropout=0.5):
        super().__init__()

        # 4 个 1D 卷积块
        self.block1 = Conv1DBlock(in_channels, 64, kernel_size=7)
        self.block2 = Conv1DBlock(64, 128, kernel_size=5)
        self.block3 = Conv1DBlock(128, 256, kernel_size=5)
        self.block4 = Conv1DBlock(256, 256, kernel_size=3)

        # 计算展平后的特征维度
        # 1024 → 512 → 256 → 128 → 64 (经过 4 次 MaxPool(2))
        self.flattened_dim = 256 * 64

        # 2 个全连接层
        self.fc1 = nn.Linear(self.flattened_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积块提取特征
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接分类
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ==================== 对比模型：ResNet1D ====================

class ResidualBlock1D(nn.Module):
    """
    1D 残差块：两层卷积 + 跳跃连接

    原理：残差连接 (x + F(x)) 使梯度更容易回传，允许训练更深的网络。
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # 残差连接
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    带 3 个残差块的 1D ResNet 变体（对比模型）

    结构: Conv → [ResBlock×2] → Pool → (重复 3 次) → FC
    输入: (batch, 2, 1024)
    输出: (batch, 10)
    """

    def __init__(self, in_channels=2, num_classes=10, dropout=0.5):
        super().__init__()

        # 初始卷积
        self.conv_in = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn_in = nn.BatchNorm1d(32)

        # 3 个阶段，每阶段 2 个残差块 + 下采样
        self.stage1 = self._make_stage(32, 64, num_blocks=2)
        self.stage2 = self._make_stage(64, 128, num_blocks=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2)

        # 全局平均池化 + 分类头
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _make_stage(self, in_ch, out_ch, num_blocks):
        """构建一个阶段：通道变换卷积 + 若干残差块 + MaxPool"""
        layers = []
        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
        layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU())
        for _ in range(num_blocks):
            layers.append(ResidualBlock1D(out_ch))
        layers.append(nn.MaxPool1d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)           # 全局平均池化 → (batch, 256, 1)
        x = x.view(x.size(0), -1)  # → (batch, 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 2D CNN（用于星座图/时频图） ====================

class CNN2D(nn.Module):
    """
    二维卷积网络（用于星座图或时频图输入）

    输入: (batch, 1, H, W)
    输出: (batch, num_classes)
    """

    def __init__(self, in_channels=1, num_classes=10, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================== 模型工厂函数 ====================

def create_model(model_name='cnn1d', in_channels=2, num_classes=10,
                 dropout=0.5, **kwargs):
    """
    模型工厂函数 —— 根据名称创建对应模型

    参数:
        model_name: 'cnn1d' | 'resnet1d' | 'cnn2d'
        in_channels: 输入通道数
        num_classes: 分类数
        dropout:     Dropout 概率
    返回:
        nn.Module 模型实例
    """
    name = model_name.lower()
    if name == 'cnn1d':
        return CNN1D(in_channels, num_classes, dropout)
    elif name == 'resnet1d':
        return ResNet1D(in_channels, num_classes, dropout)
    elif name == 'cnn2d':
        return CNN2D(in_channels, num_classes, dropout)
    else:
        raise ValueError(f"未知模型: {model_name}，可选: cnn1d, resnet1d, cnn2d")


def get_model_for_feature(feature_type, model_name='cnn1d', num_classes=10):
    """
    根据特征类型自动选择合适的模型

    iq_channels → 1D 模型 (cnn1d / resnet1d)
    constellation / stft → 2D 模型 (cnn2d)
    """
    if feature_type == 'iq_channels':
        return create_model(model_name, in_channels=2, num_classes=num_classes)
    elif feature_type in ('constellation', 'stft'):
        return create_model('cnn2d', in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"未知特征类型: {feature_type}")
