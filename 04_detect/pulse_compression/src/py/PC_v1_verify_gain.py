"""增益验证：理论信噪比增益 vs RD 图实测（多 seed 平均）

来源：04_detect/pulse_compression/doc/PC_v1_math.md 第 4 节、PC_v1_res.md 第 2 节
运行：python 04_detect/pulse_compression/src/py/PC_v1_verify_gain.py
用途：独立复现 res §2 的增益账——把噪声平均掉，核对 G = Ns·Nc·(mean w / rms w)^2 与
      落格损失，再与 rdmap 的实测峰值/噪声比对照。参数与 PC_v1.py 严格一致。
"""

import numpy as np

# ---------------- 参数（与 PC_v1.py 严格一致） ----------------
c = 3e8
fc = 10e9
lam = c / fc
B = 20e6
Tp = 5e-6
K = B / Tp
fr = 15e3
Tr = 1.0 / fr
oversample = 1.2
fs = oversample * B
Ts = 1.0 / fs
Nc = 256
Nd = 256
t_start = Tp
t_end = Tr
Ns = int(np.floor(Tp * fs))
Nw = int(np.floor((t_end - t_start) * fs))
Nr = 1024
N_fft = 2048
sigma = 1.0

targets = np.array([
    [2000.0, 40.0, -20.0, 0.8],
    [4000.0, -50.0, -20.0, 2.1],
])
amp = sigma * 10.0 ** (targets[:, 2] / 20.0)

W = np.hanning(Nc + 1)[:-1].astype(np.float32)
sum_w = float(W.sum())
sum_w2 = float((W ** 2).sum())

# ---------------- 确定性回波（与 PC_v1.py 相同） ----------------
t_fast = t_start + np.arange(Nw) * Ts
s_tx_bb = np.exp(1j * np.pi * K * (np.arange(Ns) * Ts - Tp / 2.0) ** 2).astype(np.complex64)
H = np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64)

m_idx = np.arange(Nc)[None, :, None]
t_ = t_fast[None, None, :]
tau_m = 2.0 * (targets[:, 0][:, None, None] + targets[:, 1][:, None, None] * m_idx * Tr) / c
gate = (t_ >= tau_m) & (t_ < tau_m + Tp)
echo = np.exp(1j * np.pi * K * (t_ - tau_m - Tp / 2.0) ** 2) * np.exp(-1j * 2.0 * np.pi * fc * tau_m)
rx_sig = (amp[:, None, None] * np.exp(1j * targets[:, 3])[:, None, None] * gate * echo).sum(axis=0)


def rdmap(seed):
    rng = np.random.default_rng(seed)
    rxbb = (rx_sig + (sigma / np.sqrt(2.0)) * (rng.standard_normal((Nc, Nw))
            + 1j * rng.standard_normal((Nc, Nw)))).astype(np.complex64)
    pad = np.zeros((Nc, N_fft), dtype=np.complex64)
    pad[:, :Nw] = rxbb
    pc = np.fft.ifft(np.fft.fft(pad, axis=1) * H[None, :], axis=1).astype(np.complex64)[:, 0:Nr]
    rd = np.ascontiguousarray(pc.T)
    np.multiply(rd, W[None, :], out=rd)
    return np.fft.fftshift(np.fft.fft(rd, n=Nd, axis=1), axes=1).astype(np.complex64)


# 目标所在的距离行与多普勒 bin（最近格）
range_cell = c / (2.0 * fs)
j = [int(round((t[0] - c / 2.0 * t_start) / range_cell)) for t in targets]
k_unshift = [int(round(-2.0 * t[1] / (lam * fr) * Nd)) for t in targets]       # 真值所在格
k_shift = [k + Nd // 2 for k in k_unshift]
offset = [-2.0 * t[1] / (lam * fr) * Nd - k for t, k in zip(targets, k_unshift)]  # 离最近格的距离


def hann_resp(d):
    n = np.arange(Nc)
    return abs((W * np.exp(-2j * np.pi * d * n / Nc)).sum()) / sum_w


scallop = [hann_resp(d) for d in offset]

# ---------------- 理论值 ----------------
snr_in = amp ** 2 / sigma ** 2                       # A^2 / sigma^2
gain_pc = float(Ns)
gain_win = sum_w ** 2 / sum_w2                       # Nc 点相干积累的增益
noise_cell = sigma ** 2 * Ns * sum_w2                # RD 图每格噪声功率

print("theory")
print("  input SNR A^2/sigma^2              %.4f  (%.3f dB)" % (snr_in[0], 10 * np.log10(snr_in[0])))
print("  pulse compression gain  Ns         %.1f  (%.3f dB)" % (gain_pc, 10 * np.log10(gain_pc)))
print("  windowed FFT gain (sum w)^2/sum w^2 %.4f  (%.3f dB)" % (gain_win, 10 * np.log10(gain_win)))
print("  total gain                         %.2f  (%.3f dB, on-bin)"
      % (gain_pc * gain_win, 10 * np.log10(gain_pc * gain_win)))
print("  RD noise power per cell            %.1f" % noise_cell)
print("  target 1 doppler offset            %.4f bin -> scallop %.4f (%.3f dB)"
      % (offset[0], scallop[0], 20 * np.log10(scallop[0])))
print("  target 2 doppler offset            %.4f bin -> scallop %.4f (%.3f dB)"
      % (offset[1], scallop[1], 20 * np.log10(scallop[1])))

# ---------------- 蒙特卡洛 ----------------
NSEED = 100
peak_pow = np.zeros((NSEED, 2))
noise_pow = np.zeros(NSEED)
for s in range(NSEED):
    m = np.abs(rdmap(s)) ** 2
    noise_pow[s] = m.mean()
    peak_pow[s, 0] = m[j[0], k_shift[0]]
    peak_pow[s, 1] = m[j[1], k_shift[1]]

print()
print("measured over %d noise seeds (mean)" % NSEED)
print("  RD noise power per cell            %.1f" % noise_pow.mean())
for i in range(2):
    pred_amp = amp[i] * Ns * sum_w * scallop[i]
    pred_pow = pred_amp ** 2
    meas_pow = peak_pow[:, i].mean()
    ratio = meas_pow / noise_pow.mean()
    pred_ratio = pred_pow / noise_cell
    print("  target %d  predicted peak power %.4e   measured %.4e   (%+.3f dB)"
          % (i + 1, pred_pow, meas_pow, 10 * np.log10(meas_pow / pred_pow)))
    print("            predicted SNR %.2f (%.3f dB)   measured SNR %.2f (%.3f dB)"
          % (pred_ratio, 10 * np.log10(pred_ratio), ratio, 10 * np.log10(ratio)))
