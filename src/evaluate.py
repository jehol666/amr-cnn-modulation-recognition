# -*- coding: utf-8 -*-
"""
评估模块

功能：
    1) 输出总体准确率、每类 precision / recall / F1
    2) 绘制 10×10 混淆矩阵 → results/confusion_matrix.png
    3) 绘制 SNR-准确率曲线 → results/snr_accuracy.png
    4) 保存分类报告 → results/classification_report.txt
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from .cnn_model import get_model_for_feature
from .data_utils import normalize_per_sample


def evaluate_model(model, X_test, y_test, snr_test,
                   modulations, feature_type='iq_channels',
                   results_dir='results', device=None, verbose=True):
    """
    模型评估主函数

    参数:
        model:        训练好的模型
        X_test:       测试数据
        y_test:       测试标签
        snr_test:     测试样本对应的 SNR 值
        modulations:  调制类型名称列表
        feature_type: 特征类型
        results_dir:  结果保存目录
        device:       计算设备
        verbose:      是否打印信息
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(results_dir, exist_ok=True)

    # ---------- 数据归一化 ----------
    X_test = normalize_per_sample(X_test.copy())

    # ---------- 模型预测 ----------
    model = model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        # 分批预测，避免内存溢出
        batch_size = 512
        N = X_test.shape[0]
        for i in range(0, N, batch_size):
            batch = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = y_test

    # ---------- 1) 总体准确率 + 分类报告 ----------
    overall_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=modulations, digits=4
    )

    if verbose:
        print(f"\n  总体准确率: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        print(f"\n  分类报告:\n{report}")

    # 保存分类报告
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("基于 CNN 的自动调制识别 (AMR) 分类报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总体准确率: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n\n")
        f.write("每类 Precision / Recall / F1-score:\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")

    # ---------- 2) 混淆矩阵 ----------
    _plot_confusion_matrix(y_true, y_pred, modulations, results_dir)

    # ---------- 3) SNR-准确率曲线 ----------
    _plot_snr_accuracy(y_true, y_pred, snr_test, modulations, results_dir)

    return overall_acc, report


def _plot_confusion_matrix(y_true, y_pred, modulations, results_dir):
    """
    绘制混淆矩阵热力图

    混淆矩阵的行是真实标签，列是预测标签。
    对角线越亮表示识别越准确，非对角线亮点表示易混淆的调制对。
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置坐标轴
    tick_marks = np.arange(len(modulations))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(modulations, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(modulations)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # 在格子中标注数值
    thresh = 0.5
    for i in range(len(modulations)):
        for j in range(len(modulations)):
            val = cm_norm[i, j]
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center',
                    color='white' if val > thresh else 'black',
                    fontsize=8)

    plt.tight_layout()
    path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_snr_accuracy(y_true, y_pred, snr_test, modulations, results_dir):
    """
    绘制 SNR-准确率曲线

    每种调制一条线 + 总体一条线。
    横轴为 SNR (dB)，纵轴为识别准确率。
    该曲线能直观展示模型在不同信噪比下的识别能力。
    """
    snr_values = np.unique(snr_test)
    snr_values.sort()

    fig, ax = plt.subplots(figsize=(12, 7))

    # 每种调制的 SNR-准确率曲线
    for idx, mod in enumerate(modulations):
        accs = []
        for snr in snr_values:
            mask = (snr_test == snr) & (y_true == idx)
            if mask.sum() > 0:
                acc = (y_pred[mask] == idx).sum() / mask.sum()
            else:
                acc = 0.0
            accs.append(acc)
        ax.plot(snr_values, accs, marker='o', markersize=4, linewidth=1.5,
                label=mod, alpha=0.8)

    # 总体 SNR-准确率曲线（粗黑线）
    overall_accs = []
    for snr in snr_values:
        mask = (snr_test == snr)
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
        else:
            acc = 0.0
        overall_accs.append(acc)
    ax.plot(snr_values, overall_accs, 'k-', marker='s', markersize=6,
            linewidth=3, label='Overall', zorder=10)

    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('SNR-Accuracy Curve', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(snr_values)
    ax.legend(loc='lower right', ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'snr_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()


def load_best_model(model_path, feature_type='iq_channels',
                    model_name='cnn1d', num_classes=10, device=None):
    """加载保存的最佳模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', model_name)
    feature_type = checkpoint.get('feature_type', feature_type)
    num_classes = checkpoint.get('num_classes', num_classes)

    model = get_model_for_feature(feature_type, model_name, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, feature_type, model_name
