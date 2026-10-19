# -*- coding: utf-8 -*-
"""
一键运行入口：生成数据 → 构建数据集 → 训练 → 评估 → 出图

用法:
    # 快速验证模式（1~2 分钟跑通流程）
    python main.py --quick

    # 完整模式（完整数据集，耗时较长）
    python main.py

    # 自定义参数
    python main.py --modulations BPSK QPSK 16QAM --snr-range -10 10 2 --epochs 20

    # 使用星座图特征
    python main.py --feature-type constellation --quick

    # 使用 ResNet1D 对比模型
    python main.py --model resnet1d --quick
"""

import argparse
import os
import sys
import time
import numpy as np

# 确保能导入 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_generator import SignalGenerator, MODULATIONS
from src.data_utils import extract_features
from src.train import train_model
from src.evaluate import evaluate_model


def set_seed(seed=42):
    """设置全局随机种子，保证可复现"""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='基于 CNN 的无线电信号自动调制识别 (AMR)'
    )
    parser.add_argument(
        '--modulations', nargs='+', default=None,
        help=f'调制类型列表（默认全部 10 种: {MODULATIONS}）'
    )
    parser.add_argument(
        '--snr-range', nargs=3, type=float, default=[-20, 20, 2],
        metavar=('MIN', 'MAX', 'STEP'),
        help='SNR 范围: 最小 最大 步长 (默认 -20 20 2)'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='训练轮数 (默认 30)'
    )
    parser.add_argument(
        '--feature-type', type=str, default='iq_channels',
        choices=['iq_channels', 'constellation', 'stft'],
        help='特征类型 (默认 iq_channels)'
    )
    parser.add_argument(
        '--model', type=str, default='cnn1d',
        choices=['cnn1d', 'resnet1d'],
        help='模型类型 (默认 cnn1d)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='批大小 (默认 256)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='学习率 (默认 0.001)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速验证模式：少量样本、少量 epoch，1~2 分钟跑通流程'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认 42)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='结果输出目录 (默认 results)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ==================== 参数配置 ====================
    # 调制类型
    modulations = args.modulations if args.modulations else MODULATIONS
    # 验证调制类型合法
    for m in modulations:
        if m not in MODULATIONS:
            print(f"错误: 未知调制类型 '{m}'，可选: {MODULATIONS}")
            sys.exit(1)

    # SNR 列表
    snr_min, snr_max, snr_step = args.snr_range
    snr_list = list(np.arange(snr_min, snr_max + snr_step / 2, snr_step))

    # 快速模式参数覆盖
    if args.quick:
        num_train = 20    # 每种调制每 SNR 的训练样本数
        num_test = 10     # 每种调制每 SNR 的测试样本数
        epochs = min(args.epochs, 5)
        batch_size = min(args.batch_size, 64)
        mode_name = "快速验证"
    else:
        num_train = 1000  # 完整模式: 1000 训练样本
        num_test = 200    # 完整模式: 200 测试样本
        epochs = args.epochs
        batch_size = args.batch_size
        mode_name = "完整"

    num_classes = len(modulations)

    # ==================== 打印配置信息 ====================
    print("=" * 70)
    print("  基于 CNN 的无线电通信信号自动调制识别 (AMR)")
    print("=" * 70)
    print(f"  运行模式:     {mode_name}")
    print(f"  调制类型:     {modulations} ({num_classes} 种)")
    print(f"  SNR 范围:     {snr_list[0]:.0f} ~ {snr_list[-1]:.0f} dB, "
          f"步长 {snr_step:.0f} dB ({len(snr_list)} 个点)")
    print(f"  训练样本:     {num_train} × {num_classes} 调制 × {len(snr_list)} SNR "
          f"= {num_train * num_classes * len(snr_list)} 个")
    print(f"  测试样本:     {num_test} × {num_classes} 调制 × {len(snr_list)} SNR "
          f"= {num_test * num_classes * len(snr_list)} 个")
    print(f"  特征类型:     {args.feature_type}")
    print(f"  模型:         {args.model}")
    print(f"  训练轮数:     {epochs}")
    print(f"  批大小:       {batch_size}")
    print(f"  学习率:       {args.lr}")
    print(f"  随机种子:     {args.seed}")
    print("=" * 70)

    # ==================== 设置随机种子 ====================
    set_seed(args.seed)

    # ==================== Step 1: 生成数据 ====================
    print("\n[Step 1/4] 生成调制信号数据集...")
    t0 = time.time()
    gen = SignalGenerator(seed=args.seed)

    # 训练集
    print(f"  生成训练集 ({num_train} 样本/类/SNR)...")
    X_train_iq, y_train, snr_train = gen.generate_dataset(
        modulations, snr_list, num_train
    )
    # 测试集（用不同种子确保不重叠）
    gen_test = SignalGenerator(seed=args.seed + 1000)
    print(f"  生成测试集 ({num_test} 样本/类/SNR)...")
    X_test_iq, y_test, snr_test = gen_test.generate_dataset(
        modulations, snr_list, num_test
    )

    print(f"  数据生成完成，耗时 {time.time()-t0:.1f}s")
    print(f"  训练集: X{X_train_iq.shape}, y{y_train.shape}")
    print(f"  测试集: X{X_test_iq.shape}, y{y_test.shape}")

    # ==================== Step 2: 特征提取 ====================
    print(f"\n[Step 2/4] 特征提取 ({args.feature_type})...")
    t0 = time.time()
    X_train = extract_features(X_train_iq, args.feature_type)
    X_test = extract_features(X_test_iq, args.feature_type)
    print(f"  特征提取完成，耗时 {time.time()-t0:.1f}s")
    print(f"  训练特征: {X_train.shape}")
    print(f"  测试特征: {X_test.shape}")

    # ==================== Step 3: 训练模型 ====================
    print(f"\n[Step 3/4] 训练 CNN 模型...")
    model, history = train_model(
        X_train, y_train, X_test, y_test,
        feature_type=args.feature_type,
        model_name=args.model,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr,
        results_dir=args.results_dir,
    )

    # ==================== Step 4: 评估模型 ====================
    print(f"\n[Step 4/4] 评估模型...")
    overall_acc, report = evaluate_model(
        model, X_test, y_test, snr_test,
        modulations=modulations,
        feature_type=args.feature_type,
        results_dir=args.results_dir,
    )

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("  全部完成，结果在 results/ 目录")
    print("=" * 70)
    print(f"  results/best_model.pth          - 最佳模型权重")
    print(f"  results/training_curve.png      - 训练曲线")
    print(f"  results/confusion_matrix.png    - 混淆矩阵")
    print(f"  results/snr_accuracy.png        - SNR-准确率曲线")
    print(f"  results/classification_report.txt - 分类报告")
    print("=" * 70)


if __name__ == '__main__':
    main()
