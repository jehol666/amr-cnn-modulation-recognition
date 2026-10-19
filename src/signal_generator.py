# -*- coding: utf-8 -*-
"""
信号生成器 —— 纯 numpy 实现，不依赖任何通信库

支持 10 种调制方式：
    BPSK, QPSK, 8PSK, 16QAM, 64QAM, GMSK, AM-DSB, AM-SSB, FM, PAM4

参数说明：
    每样本 128 个符号，每符号 8 个采样点 → 每样本 1024 个采样点
    RRC 脉冲成型，滚降系数 0.35
    信道效应：AWGN + 随机相位偏移 + 轻微频偏 + 时间偏移
"""

import numpy as np


# ==================== 全局常量 ====================

# 10 种调制类型（顺序固定，对应标签 0~9）
MODULATIONS = [
    'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
    'GMSK', 'AM-DSB', 'AM-SSB', 'FM', 'PAM4'
]

# 调制类型 → 标签索引
MOD_TO_LABEL = {m: i for i, m in enumerate(MODULATIONS)}

# 默认参数
DEFAULT_NUM_SYMBOLS = 128       # 每样本符号数
DEFAULT_SPS = 8                 # 每符号采样点数 (samples per symbol)
DEFAULT_RRC_ALPHA = 0.35        # RRC 滚降系数
DEFAULT_RRC_SPAN = 6            # RRC 滤波器跨度（符号周期数）


# ==================== RRC 脉冲成型滤波器 ====================

def design_rrc_filter(sps, alpha, span):
    """
    设计根升余弦 (Root Raised Cosine) 脉冲成型滤波器

    原理：RRC 滤波器在发送端和接收端各用一个，两者级联得到升余弦响应，
          满足 Nyquist 无码间串扰准则。滚降系数 alpha 控制带宽过渡带宽度。

    参数:
        sps:   每符号采样点数
        alpha: 滚降系数 (0~1)
        span:  滤波器跨度（以符号周期为单位，单边）

    返回:
        归一化 RRC 滤波器系数 (numpy 数组)
    """
    N = span * sps
    t = np.arange(-N, N + 1, dtype=float) / sps  # 以符号周期为单位的时间轴
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0.0:
            # t=0 处的极限值
            h[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-9 and alpha != 0:
            # t=±T/(4α) 处的极限值（避免分母为零）
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) \
                + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den

    # 能量归一化，确保滤波后信号功率稳定
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ==================== 高斯滤波器（用于 GMSK） ====================

def design_gaussian_filter(sps, bt, span):
    """
    设计高斯低通滤波器（用于 GMSK 的脉冲成型）

    原理：GMSK 先用高斯滤波器对矩形 NRZ 脉冲做平滑，再积分相位，
          BT 值（带宽-时间乘积）控制高斯脉冲的宽度，典型值 0.3。

    参数:
        sps:  每符号采样点数
        bt:   带宽-时间乘积
        span: 滤波器跨度（符号周期数，单边）
    """
    N = span * sps
    t = np.arange(-N, N + 1, dtype=float) / sps
    # 高斯脉冲
    h = np.sqrt(2 * np.pi) / np.sqrt(np.log(2.0)) * bt * \
        np.exp(-2 * (np.pi * bt * t) ** 2 / np.log(2.0))
    h /= np.sum(h)  # 归一化（面积=1）
    return h


# ==================== 信号生成器主类 ====================

class SignalGenerator:
    """
    无线电调制信号生成器

    用法:
        gen = SignalGenerator()
        iq, label = gen.generate_sample('QPSK', snr_db=10)
        X, y, snrs = gen.generate_dataset(['BPSK','QPSK'], [0,10], num_per_class=100)
    """

    def __init__(self, num_symbols=DEFAULT_NUM_SYMBOLS,
                 sps=DEFAULT_SPS, rrc_alpha=DEFAULT_RRC_ALPHA,
                 rrc_span=DEFAULT_RRC_SPAN, seed=None):
        self.num_symbols = num_symbols
        self.sps = sps
        self.total_samples = num_symbols * sps  # 1024
        self.rrc_alpha = rrc_alpha
        self.rrc_span = rrc_span

        # 预设计滤波器（复用，提高效率）
        self.rrc = design_rrc_filter(sps, rrc_alpha, rrc_span)
        self.gaussian = design_gaussian_filter(sps, bt=0.3, span=2)

        # 随机数生成器
        self.rng = np.random.RandomState(seed)

    # ---------- 各调制方式的符号生成 ----------

    def _gen_bpsk(self):
        """BPSK: 二进制相移键控，符号 ∈ {-1, +1}"""
        bits = self.rng.randint(0, 2, self.num_symbols)
        return 2.0 * bits - 1.0  # 映射到 ±1

    def _gen_qpsk(self):
        """QPSK: 正交相移键控，4 个相位"""
        bits = self.rng.randint(0, 4, self.num_symbols)
        return np.exp(1j * np.pi * (2 * bits + 1) / 4.0)

    def _gen_8psk(self):
        """8PSK: 8 相相移键控"""
        bits = self.rng.randint(0, 8, self.num_symbols)
        return np.exp(1j * 2 * np.pi * bits / 8.0)

    def _gen_16qam(self):
        """16QAM: 16 阶正交幅度调制，4×4 星座图"""
        bits = self.rng.randint(0, 16, self.num_symbols)
        re = 2 * (bits // 4) - 3   # -3, -1, 1, 3
        im = 2 * (bits % 4) - 3    # -3, -1, 1, 3
        sym = re + 1j * im
        return sym / np.sqrt(np.mean(np.abs(sym) ** 2))  # 功率归一化

    def _gen_64qam(self):
        """64QAM: 64 阶正交幅度调制，8×8 星座图"""
        bits = self.rng.randint(0, 64, self.num_symbols)
        re = 2 * (bits // 8) - 7   # -7, -5, ..., 7
        im = 2 * (bits % 8) - 7
        sym = re + 1j * im
        return sym / np.sqrt(np.mean(np.abs(sym) ** 2))

    def _gen_pam4(self):
        """PAM4: 4 阶脉冲幅度调制，符号 ∈ {-3, -1, +1, +3}"""
        bits = self.rng.randint(0, 4, self.num_symbols)
        sym = 2 * bits - 3.0  # -3, -1, 1, 3
        return sym / np.sqrt(np.mean(np.abs(sym) ** 2))

    # ---------- 数字调制：上采样 + RRC 成型 ----------

    def _apply_rrc(self, symbols):
        """
        对符号序列上采样后做 RRC 脉冲成型

        原理：先将符号序列零插值上采样到 sps 倍速率，
              再与 RRC 滤波器卷积完成脉冲成型。
        """
        # 零插值上采样
        upsampled = np.zeros(self.total_samples, dtype=complex)
        upsampled[::self.sps] = symbols
        # 卷积成型
        shaped = np.convolve(upsampled, self.rrc, mode='same')
        return shaped

    # ---------- GMSK 调制 ----------

    def _gen_gmsk(self):
        """
        GMSK: 高斯最小频移键控（连续相位调制）

        原理：
            1. 生成随机比特 → NRZ 编码 (±1)
            2. 高斯滤波器平滑 NRZ 脉冲
            3. 积分得到瞬时相位
            4. s(t) = exp(j * 相位(t))
        """
        bits = self.rng.randint(0, 2, self.num_symbols)
        nrz = 2.0 * bits - 1.0  # ±1

        # 上采样
        upsampled = np.zeros(self.total_samples, dtype=float)
        upsampled[::self.sps] = nrz

        # 高斯滤波
        filtered = np.convolve(upsampled, self.gaussian, mode='same')

        # 积分相位（h=0.5 是 MSK 的调制指数）
        h_mod = 0.5
        phase = np.cumsum(filtered) * h_mod * np.pi / self.sps
        # 相位连续化
        phase -= phase[0]

        signal = np.exp(1j * phase)
        return signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    # ---------- 模拟调制 ----------

    def _gen_am_dsb(self):
        """
        AM-DSB: 双边带幅度调制
        基带等效: s(t) = 1 + m * x(t)
        其中 x(t) 为随机消息信号，m 为调制指数
        """
        # 随机消息信号（低通滤波后的随机信号）
        msg = self.rng.randn(self.total_samples)
        # 简单低通滤波（移动平均）使信号平滑
        kernel = np.ones(self.sps) / self.sps
        msg = np.convolve(msg, kernel, mode='same')
        msg /= np.std(msg)  # 归一化

        m = 0.8  # 调制指数
        signal = 1.0 + m * msg
        return signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    def _gen_am_ssb(self):
        """
        AM-SSB: 单边带幅度调制
        原理: s(t) = x(t) + j * Hilbert{x(t)}  (上边带)
        Hilbert 变换用 FFT 实现
        """
        msg = self.rng.randn(self.total_samples)
        kernel = np.ones(self.sps) / self.sps
        msg = np.convolve(msg, kernel, mode='same')
        msg /= np.std(msg)

        # FFT 实现 Hilbert 变换
        N = len(msg)
        fft_msg = np.fft.fft(msg)
        h_freq = np.zeros(N)
        h_freq[0] = 1.0
        if N % 2 == 0:
            h_freq[1:N // 2] = 2.0
            h_freq[N // 2] = 1.0
        else:
            h_freq[1:(N + 1) // 2] = 2.0
        analytic = np.fft.ifft(fft_msg * h_freq)

        signal = analytic  # 解析信号即为 SSB
        return signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    def _gen_fm(self):
        """
        FM: 频率调制
        原理: s(t) = exp(j * 2π * kf * ∫x(τ)dτ)
        """
        msg = self.rng.randn(self.total_samples)
        kernel = np.ones(self.sps) / self.sps
        msg = np.convolve(msg, kernel, mode='same')
        msg /= np.std(msg)

        kf = 0.5  # 频率偏移常数
        # 积分
        integral = np.cumsum(msg) / self.sps
        signal = np.exp(1j * 2 * np.pi * kf * integral)
        return signal / np.sqrt(np.mean(np.abs(signal) ** 2))

    # ---------- 调制分发 ----------

    def _modulate(self, mod_type):
        """根据调制类型分发到对应的调制函数"""
        if mod_type == 'BPSK':
            return self._apply_rrc(self._gen_bpsk())
        elif mod_type == 'QPSK':
            return self._apply_rrc(self._gen_qpsk())
        elif mod_type == '8PSK':
            return self._apply_rrc(self._gen_8psk())
        elif mod_type == '16QAM':
            return self._apply_rrc(self._gen_16qam())
        elif mod_type == '64QAM':
            return self._apply_rrc(self._gen_64qam())
        elif mod_type == 'PAM4':
            return self._apply_rrc(self._gen_pam4())
        elif mod_type == 'GMSK':
            return self._gen_gmsk()
        elif mod_type == 'AM-DSB':
            return self._gen_am_dsb()
        elif mod_type == 'AM-SSB':
            return self._gen_am_ssb()
        elif mod_type == 'FM':
            return self._gen_fm()
        else:
            raise ValueError(f"未知调制类型: {mod_type}")

    # ---------- 信道效应 ----------

    def add_channel_effects(self, signal, snr_db):
        """
        添加真实信道效应：
            1. AWGN 高斯白噪声（由 SNR 控制）
            2. 随机相位偏移 [0, 2π)
            3. 轻微频率偏移 (±5ppm)
            4. 时间偏移（分数倍采样延迟，用线性插值实现）

        参数:
            signal: 复基带信号
            snr_db: 信噪比 (dB)
        返回:
            受信道污染后的复信号
        """
        N = len(signal)

        # --- 1. 随机相位偏移 ---
        phase_offset = self.rng.uniform(0, 2 * np.pi)
        signal = signal * np.exp(1j * phase_offset)

        # --- 2. 轻微频率偏移 ---
        # 归一化频偏：±5ppm 相对于采样率
        freq_offset = self.rng.uniform(-5e-5, 5e-5)
        t = np.arange(N)
        signal = signal * np.exp(1j * 2 * np.pi * freq_offset * t)

        # --- 3. 时间偏移（分数延迟）---
        time_offset = self.rng.uniform(-2, 2)  # ±2 个采样点
        t_orig = np.arange(N)
        t_shifted = t_orig - time_offset
        # 线性插值实现分数延迟
        re = np.interp(t_shifted, t_orig, signal.real)
        im = np.interp(t_shifted, t_orig, signal.imag)
        signal = re + 1j * im

        # --- 4. AWGN 加性高斯白噪声 ---
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            self.rng.randn(N) + 1j * self.rng.randn(N)
        )
        signal = signal + noise

        return signal

    # ---------- 单样本生成 ----------

    def generate_sample(self, mod_type, snr_db):
        """
        生成一个完整的调制信号样本（含信道效应）

        返回:
            iq: (2, total_samples) 数组，[0]路为 I，[1]路为 Q
            label: 调制类型标签（整数）
        """
        signal = self._modulate(mod_type)
        signal = self.add_channel_effects(signal, snr_db)
        iq = np.stack([signal.real, signal.imag], axis=0)  # shape: (2, 1024)
        label = MOD_TO_LABEL[mod_type]
        return iq, label

    # ---------- 数据集生成 ----------

    def generate_dataset(self, mod_types, snr_list, num_per_class,
                         return_complex=False):
        """
        批量生成数据集

        参数:
            mod_types:     调制类型列表
            snr_list:      SNR 列表 (dB)
            num_per_class: 每种调制在每个 SNR 下生成的样本数
            return_complex: 是否返回复信号（默认返回 I/Q 两路）

        返回:
            X:     (N, 2, total_samples) I/Q 数据
            y:     (N,) 标签
            snrs:  (N,) 对应的 SNR 值
        """
        all_iq = []
        all_labels = []
        all_snrs = []

        for mod_type in mod_types:
            label = MOD_TO_LABEL[mod_type]
            for snr in snr_list:
                for _ in range(num_per_class):
                    signal = self._modulate(mod_type)
                    signal = self.add_channel_effects(signal, snr)
                    if return_complex:
                        all_iq.append(signal)
                    else:
                        iq = np.stack([signal.real, signal.imag], axis=0)
                        all_iq.append(iq)
                    all_labels.append(label)
                    all_snrs.append(snr)

        X = np.array(all_iq, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        snrs = np.array(all_snrs, dtype=np.float32)

        return X, y, snrs
