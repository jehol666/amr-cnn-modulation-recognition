# 基于 CNN 的无线电通信信号自动调制识别 (AMR)

## 研究背景

自动调制识别 (Automatic Modulation Recognition, AMR) 是软件定义无线电 (SDR) 和认知无线电 (CR) 中的关键技术。在频谱感知、频谱监测、非合作通信等场景中，接收端需要在未知调制方式的前提下，仅从接收信号中自动判别发射端所采用的调制类型。

传统 AMR 方法主要基于似然比检验或统计特征提取，前者需要先验知识且计算复杂，后者依赖人工设计特征、泛化能力有限。近年来，深度学习特别是卷积神经网络 (CNN) 在 AMR 领域展现出显著优势：CNN 能直接从原始 I/Q 信号中自动学习层次化特征，无需人工特征工程，且对信道损伤具有较好的鲁棒性。

本项目构建了一个完整的 AMR 仿真研究流程，涵盖信号生成、数据预处理、CNN 模型训练与评估，支持 10 种常见调制方式的识别。

## 项目结构

```
amr_cnn/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖
├── main.py                   # 一键运行入口
├── src/
│   ├── __init__.py
│   ├── signal_generator.py   # 信号生成器（10 种调制，纯 numpy）
│   ├── data_utils.py         # 数据预处理（I/Q 双通道 / 星座图 / STFT 时频图）
│   ├── cnn_model.py          # CNN 模型定义（CNN1D + ResNet1D + CNN2D）
│   ├── train.py              # 训练主程序
│   └── evaluate.py           # 评估、混淆矩阵、SNR-准确率曲线
└── results/                  # 输出目录（模型权重、图片、报告）
```

## 数据集说明

### 调制类型 (10 种)

| 标签 | 调制方式 | 类别 | 说明 |
|------|----------|------|------|
| 0 | BPSK | 数字相位调制 | 二进制相移键控 |
| 1 | QPSK | 数字相位调制 | 正交相移键控 |
| 2 | 8PSK | 数字相位调制 | 8 相相移键控 |
| 3 | 16QAM | 数字正交幅度调制 | 16 阶 QAM |
| 4 | 64QAM | 数字正交幅度调制 | 64 阶 QAM |
| 5 | GMSK | 数字连续相位调制 | 高斯最小频移键控 |
| 6 | AM-DSB | 模拟幅度调制 | 双边带调幅 |
| 7 | AM-SSB | 模拟幅度调制 | 单边带调幅 |
| 8 | FM | 模拟频率调制 | 调频 |
| 9 | PAM4 | 数字脉冲幅度调制 | 4 阶 PAM |

### 信号参数

- **符号数**: 每样本 128 个符号
- **采样率**: 每符号 8 个采样点 → 每样本 1024 个采样点
- **脉冲成型**: 根升余弦 (RRC) 滤波器，滚降系数 0.35，跨度 6 个符号周期
- **GMSK**: 高斯滤波器，BT = 0.3，调制指数 h = 0.5

### 信道效应

每个信号样本经过以下真实信道损伤：
1. **AWGN**: 加性高斯白噪声，由 SNR 控制噪声功率
2. **随机相位偏移**: [0, 2π) 均匀分布
3. **载微频率偏移**: ±5ppm 随机频偏
4. **时间偏移**: ±2 个采样点的随机分数延迟（线性插值实现）

### SNR 范围

-20 dB ~ 20 dB，步长 2 dB，共 21 个 SNR 点

### 数据集规模

| 模式 | 每种调制每 SNR | 总训练样本 | 总测试样本 |
|------|---------------|-----------|-----------|
| 完整模式 | 1000 训练 + 200 测试 | 210,000 | 42,000 |
| 快速模式 | 20 训练 + 10 测试 | 4,200 | 2,100 |

## 模型结构

### 主模型: CNN1D

输入: `(batch, 2, 1024)` — I/Q 双通道一维时域信号

```
Conv1d(2→64,  k=7) → BN → ReLU → MaxPool(2)   → 512
Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)  → 256
Conv1d(128→256,k=5) → BN → ReLU → MaxPool(2)  → 128
Conv1d(256→256,k=3) → BN → ReLU → MaxPool(2)  → 64
Flatten(256×64=16384)
FC(16384→256) → ReLU → Dropout(0.5)
FC(256→128)   → ReLU → Dropout(0.5)
FC(128→10)    → Softmax
```

- 优化器: Adam, lr = 0.001
- 损失函数: CrossEntropyLoss
- Dropout: 0.5

### 对比模型: ResNet1D

带 3 个阶段（每阶段 2 个残差块）的 1D ResNet 变体，使用全局平均池化替代大尺寸全连接层，参数量更少。

### 2D 模型: CNN2D

用于星座图或 STFT 时频图输入，4 层 Conv2D + 全局平均池化 + 分类头。

## 运行方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速验证模式 (1~2 分钟)

```bash
python main.py --quick
```

少量样本、5 个 epoch，用于验证流程能跑通。

### 3. 完整训练模式

```bash
python main.py
```

完整数据集，30 个 epoch。预期耗时约 30~60 分钟（取决于硬件，GPU 更快）。

### 4. 自定义参数

```bash
# 指定调制类型
python main.py --modulations BPSK QPSK 16QAM 64QAM

# 自定义 SNR 范围
python main.py --snr-range -10 10 2

# 使用星座图特征
python main.py --feature-type constellation

# 使用 STFT 时频图特征
python main.py --feature-type stft

# 使用 ResNet1D 对比模型
python main.py --model resnet1d

# 调整训练参数
python main.py --epochs 50 --batch-size 512 --lr 0.0005
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--modulations` | 全部 10 种 | 调制类型列表 |
| `--snr-range` | -20 20 2 | SNR 范围 (最小 最大 步长) |
| `--epochs` | 30 | 训练轮数 |
| `--feature-type` | iq_channels | 特征类型: iq_channels / constellation / stft |
| `--model` | cnn1d | 模型: cnn1d / resnet1d |
| `--batch-size` | 256 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--quick` | False | 快速验证模式 |
| `--seed` | 42 | 随机种子 |

## 输出结果

运行完成后，`results/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳模型权重 (PyTorch) |
| `training_curve.png` | 训练/验证的 Loss 和 Accuracy 曲线 |
| `confusion_matrix.png` | 10×10 归一化混淆矩阵热力图 |
| `snr_accuracy.png` | SNR-准确率曲线（每种调制一条线 + 总体） |
| `classification_report.txt` | 每类 Precision / Recall / F1 分类报告 |

## 预期结果

### 混淆矩阵

- 高 SNR (>10 dB) 下，各调制类型识别率接近 100%
- 低 SNR (<-10 dB) 下，M-PSK 和 M-QAM 之间可能出现混淆（如 QPSK ↔ BPSK）
- AM-DSB / AM-SSB / FM 等模拟调制与数字调制区分度高

### SNR-准确率曲线

- 总体准确率随 SNR 单调上升
- 低阶调制 (BPSK, QPSK) 在低 SNR 下表现优于高阶调制 (64QAM)
- 10 dB 以上总体准确率可达 90%+ (完整模式)

### 快速模式 vs 完整模式

| 指标 | 快速模式 | 完整模式 |
|------|---------|---------|
| 耗时 | 1~2 分钟 | 30~60 分钟 |
| 训练样本 | 4,200 | 210,000 |
| 预期准确率 | 50~70% | 85~95% |
| 用途 | 验证流程 | 正式实验 |

## 技术特点

1. **信号生成纯 numpy 实现**: 不依赖任何通信库 (如 CommPy)，所有调制和信道效应从底层实现
2. **三种特征表示**: I/Q 双通道时域、星座图、STFT 时频图，可对比不同输入对识别性能的影响
3. **真实信道仿真**: 包含 AWGN、相位偏移、频偏、时间偏移四种信道损伤
4. **可复现性**: 固定随机种子，训练/测试数据用不同种子生成确保不重叠
5. **模型工厂模式**: 支持 CNN1D、ResNet1D、CNN2D 三种模型灵活切换

## 依赖

- numpy >= 1.21
- scipy >= 1.7
- torch >= 1.10
- scikit-learn >= 1.0
- matplotlib >= 3.4
