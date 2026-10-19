# -*- coding: utf-8 -*-
"""
训练主程序

功能：
    - 训练 CNN 模型，支持 1D (I/Q) 和 2D (星座图/时频图) 输入
    - 每 epoch 打印训练损失和准确率
    - 保存最佳模型到 results/best_model.pth
    - 保存训练曲线到 results/training_curve.png
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免显示窗口
import matplotlib.pyplot as plt

from .cnn_model import get_model_for_feature
from .data_utils import create_dataloader, normalize_per_sample


def train_model(X_train, y_train, X_val, y_val,
                feature_type='iq_channels',
                model_name='cnn1d',
                num_classes=10,
                epochs=30,
                batch_size=256,
                lr=0.001,
                dropout=0.5,
                results_dir='results',
                device=None,
                verbose=True):
    """
    训练模型主函数

    参数:
        X_train, y_train: 训练数据和标签
        X_val, y_val:     验证数据和标签
        feature_type:     特征类型 ('iq_channels' / 'constellation' / 'stft')
        model_name:       模型名称 ('cnn1d' / 'resnet1d')
        num_classes:      分类数
        epochs:           训练轮数
        batch_size:       批大小
        lr:               学习率
        dropout:          Dropout 概率
        results_dir:      结果保存目录
        device:           计算设备
        verbose:          是否打印训练信息
    返回:
        model:            训练后的最佳模型
        history:          训练历史 (loss, acc 曲线)
    """
    # ---------- 设备选择 ----------
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"  训练设备: {device}")

    os.makedirs(results_dir, exist_ok=True)

    # ---------- 数据归一化 ----------
    X_train = normalize_per_sample(X_train.copy())
    X_val = normalize_per_sample(X_val.copy())

    # ---------- 创建 DataLoader ----------
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)

    # ---------- 创建模型 ----------
    model = get_model_for_feature(feature_type, model_name, num_classes)
    model = model.to(device)
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型: {model_name}, 参数量: {total_params:,}")

    # ---------- 损失函数 & 优化器 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------- 训练循环 ----------
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    best_model_path = os.path.join(results_dir, 'best_model.pth')

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == batch_y).sum().item()
            running_total += batch_X.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_X.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # --- 记录历史 ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- 保存最佳模型 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'feature_type': feature_type,
                'num_classes': num_classes,
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }, best_model_path)

        # --- 打印进度 ---
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")

    # ---------- 保存训练曲线 ----------
    _plot_training_curve(history, results_dir, epochs)

    if verbose:
        total_time = time.time() - start_time
        print(f"  训练完成！总耗时: {total_time:.1f}s, 最佳验证准确率: {best_val_acc:.4f}")

    # 加载最佳模型权重
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history


def _plot_training_curve(history, results_dir, epochs):
    """绘制并保存训练曲线（Loss 和 Accuracy）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train / Val Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train / Val Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'training_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
