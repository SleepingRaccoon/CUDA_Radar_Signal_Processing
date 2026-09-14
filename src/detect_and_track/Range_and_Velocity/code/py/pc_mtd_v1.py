"""
脉冲压缩 + MTD（NumPy 版，CPU 基准）

来源：../../doc/PC_and_MTD_v01.md 第 3、4、6、7 节
运行：python src/detect_and_track/Range_and_Velocity/code/py/pc_mtd_v1.py
本程序承担两件事：生成仿真回波基带，以及做 CPU 端的脉压 + MTD
输出：output/py/pc_mtd_v1_rxbb.npy（仿真回波基带 [Nc][Nw] complex64）
      output/py/pc_mtd_v1_rdm.npy（距离-多普勒图 [Nr][Nd] complex64）与四张图
布局：rd 布局 [Nr][Nd]，r 在前 d 在后
计时：REPS 次取平均，只统计算法时间（内存访存 + 计算，两者在 CPU 上交织，合成一个数）；
      磁盘读写与回波生成一律在计时区外；LFM 频谱是雷达系统常驻量，预先算好，不计时
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPS = 100

OUT = Path(__file__).resolve().parents[2] / "output" / "py"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "pc_mtd_v1"
RXBB = OUT / (TAG + "_rxbb.npy")     # 仿真回波基带
RDM = OUT / (TAG + "_rdm.npy")       # 距离-多普勒图

# IEEE 风格绘图：衬线字体，数学符号用 Computer Modern，刻度内向，高 dpi
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ---------------- 系统参数 ----------------
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
Ns = int(np.floor(Tp * fs))                 # 120
Nw = int(np.floor((t_end - t_start) * fs))  # 1480
Nr = 1024
N_fft = 2048                                # 距离维 FFT 点数，只要 >= Nw
sigma = 1.0                                 # 噪声标准差，每采样点方差 sigma^2 = 1
seed = 42

# ---------------- 场景 ----------------
# 每个目标：[距离 m, 速度 m/s（远离为正）, 回波幅度 dB, 初相 rad]
# 回波幅度 A = sigma * 10^(amp/20)
targets = np.array([
    [2000.0, 40.0, -20.0, 0.8],
    [4000.0, -50.0, -20.0, 2.1],
])
amp = sigma * 10.0 ** (targets[:, 2] / 20.0)

# ---------------- 派生参数 ----------------
range_res = c / (2.0 * B)            # 距离分辨率
range_cell = c / (2.0 * fs)          # 距离单元格
vel_res = lam * fr / (2.0 * Nc)      # 速度分辨率（真实脉冲数决定）
vel_cell = lam * fr / (2.0 * Nd)     # 速度单元格（FFT 点数决定）
R_ua = c / 2.0 * (t_end - Tp)        # 回波完整落在窗内的最大距离
R_min = c / 2.0 * t_start            # 窗起点对应的最小距离
v_ua = lam * fr / 4.0                # 最大不模糊速度

r_axis = c / 2.0 * t_start + np.arange(Nr) * range_cell
v_axis = -(lam / 2.0) * (np.arange(-(Nd // 2), Nd // 2) * fr / Nd)

# ---------------- 打印系统参数与派生参数 ----------------
print("=" * 62)
print("pulse compression + MTD  (NumPy)")
print("=" * 62)
print("system parameters")
for k, v in [
    ("c", "%.3e m/s" % c),
    ("fc", "%.3f GHz" % (fc / 1e9)),
    ("lambda", "%.4f m" % lam),
    ("B", "%.3f MHz" % (B / 1e6)),
    ("Tp", "%.3f us" % (Tp * 1e6)),
    ("K", "%.3e Hz/s" % K),
    ("fr (PRF)", "%.3f kHz" % (fr / 1e3)),
    ("Tr (PRI)", "%.3f us" % (Tr * 1e6)),
    ("fs", "%.3f MHz (%.2f x B)" % (fs / 1e6, oversample)),
    ("Ts", "%.3f ns" % (Ts * 1e9)),
    ("t_start, t_end", "%.3f, %.3f us" % (t_start * 1e6, t_end * 1e6)),
    ("Ns (pulse samples)", "%d" % Ns),
    ("Nw (window samples)", "%d" % Nw),
    ("Nr (kept range cells)", "%d" % Nr),
    ("N_fft (range FFT)", "%d" % N_fft),
    ("Nc (chirps)", "%d" % Nc),
    ("Nd (Doppler bins)", "%d" % Nd),
    ("sigma (noise std)", "%.3f" % sigma),
    ("seed", "%d" % seed),
    ("window", "hann"),
    ("dtype", "complex64"),
]:
    print("  %-34s %s" % (k, v))

print("derived parameters")
for k, v in [
    ("range resolution  c/2B", "%.3f m" % range_res),
    ("range cell  c/2fs", "%.3f m" % range_cell),
    ("velocity resolution  lam*fr/2Nc", "%.3f m/s" % vel_res),
    ("velocity cell  lam*fr/2Nd", "%.3f m/s" % vel_cell),
    ("max unambiguous range R_ua", "%.1f m" % R_ua),
    ("measurable range (full pulse)", "%.1f .. %.1f m" % (R_min, R_ua)),
    ("max unambiguous velocity v_ua", "%.1f m/s" % v_ua),
    ("range axis", "%.1f .. %.1f m" % (r_axis[0], r_axis[-1])),
    ("velocity axis", "%.1f .. %.1f m/s" % (v_axis[0], v_axis[-1])),
]:
    print("  %-34s %s" % (k, v))

print("scene")
for i, t in enumerate(targets):
    print("  target %d   R = %8.1f m   v = %+7.2f m/s   amp = %+5.1f dB   A = %.5f" % (i + 1, t[0], t[1], t[2], amp[i]))

# ---------------- 生成仿真回波基带（不计时） ----------------
rng = np.random.default_rng(seed)
t_fast = t_start + np.arange(Nw) * Ts

# 发射复基带：中心在 Tp/2 的 LFM（相位用 float64 算，落到 complex64）
s_tx_bb = np.exp(1j * np.pi * K * (np.arange(Ns) * Ts - Tp / 2.0) ** 2).astype(np.complex64)

# 每个目标一次广播算完 [Nt][Nc][Nw]；相位 fc*tau 是大数，必须用 float64
m_idx = np.arange(Nc)[None, :, None]
t_ = t_fast[None, None, :]
tau_m = 2.0 * (targets[:, 0][:, None, None] + targets[:, 1][:, None, None] * m_idx * Tr) / c
gate = (t_ >= tau_m) & (t_ < tau_m + Tp)
echo = np.exp(1j * np.pi * K * (t_ - tau_m - Tp / 2.0) ** 2) * np.exp(-1j * 2.0 * np.pi * fc * tau_m)
rx = (amp[:, None, None] * np.exp(1j * targets[:, 3])[:, None, None] * gate * echo).sum(axis=0)

# 复高斯白噪声，每个采样点方差 sigma^2 = 1；采集数据落到 complex64
rxbb = (rx + (sigma / np.sqrt(2.0)) * (rng.standard_normal((Nc, Nw)) + 1j * rng.standard_normal((Nc, Nw)))).astype(np.complex64)
np.save(RXBB, rxbb)
print("saved: %s   %s   %s" % (RXBB, rxbb.shape, rxbb.dtype))

# ---------------- 雷达系统常驻量：LFM 频谱与加窗（预先算好，不计时） ----------------
# H = conj(FFT(s_tx_bb)) 为相关形式，峰值索引直接对应回波起始
H = np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64)   # [N_fft]
# 慢时间加窗：周期汉宁窗，长度等于脉冲数
W = np.hanning(Nc + 1)[:-1].astype(np.float32)                 # [Nc]


def run_once(rxbb):
    # 补零到 N_fft
    raw_pad = np.zeros((Nc, N_fft), dtype=np.complex64)
    raw_pad[:, :Nw] = rxbb
    # 距离维匹配滤波，截取前 Nr 个距离单元 -> [Nc][Nr]
    pc = np.fft.ifft(np.fft.fft(raw_pad, axis=1) * H[None, :], axis=1).astype(np.complex64)[:, 0:Nr]
    # 加窗 + 转置成 rd 布局 -> [Nr][Nc]
    rd = np.ascontiguousarray((pc * W[:, None]).T)
    # 慢时间 FFT + fftshift -> [Nr][Nd]
    return np.fft.fftshift(np.fft.fft(rd, n=Nd, axis=1), axes=1).astype(np.complex64)


# ---------------- 第一次执行：这一次的结果就是输出 ----------------
rdm = run_once(rxbb)
np.save(RDM, rdm)

mag = np.abs(rdm)
print("self-check  (peak vs ground truth)")
for i, t in enumerate(targets):
    r_win = (r_axis >= t[0] - 5.0 * range_cell) & (r_axis <= t[0] + 5.0 * range_cell)
    sub = mag[r_win, :]
    ir, iv = np.unravel_index(int(np.argmax(sub)), sub.shape)
    print("  target %d   true R = %8.1f m  v = %+7.2f m/s   measured R = %8.1f m  v = %+7.2f m/s"
          % (i + 1, t[0], t[1], r_axis[r_win][ir], v_axis[iv]))

# ---------------- 四张图（rd 布局，横轴速度、纵轴距离） ----------------
db = np.clip(20.0 * np.log10(mag / mag.max() + 1e-12), -60.0, 0.0)
ext = [v_axis[0], v_axis[-1], r_axis[0] / 1e3, r_axis[-1] / 1e3]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(mag, origin="lower", aspect="auto", extent=ext, cmap="viridis", interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$")
fig.savefig(OUT / (TAG + "_rdm_2d.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(db, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=-60.0, vmax=0.0, interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$ (dB)")
fig.savefig(OUT / (TAG + "_rdm_2d_dB.png"))
plt.close(fig)

for tag, z, zlabel, zlim in (("", mag, r"$|s_{\mathrm{RD}}|$", None), ("_dB", db, r"$|s_{\mathrm{RD}}|$ (dB)", (-60.0, 0.0))):
    fig = plt.figure(figsize=(4.2, 3.4))
    ax = fig.add_subplot(projection="3d")
    Vs, Rs = np.meshgrid(v_axis, r_axis / 1e3)
    ax.plot_surface(Vs, Rs, z, rstride=4, cstride=1, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel(r"$R$ (km)")
    ax.set_zlabel(zlabel)
    if zlim is not None:
        ax.set_zlim(*zlim)
    fig.savefig(OUT / (TAG + "_rdm_3d%s.png" % tag))
    plt.close(fig)

# ---------------- 计时：REPS 次取平均，只统计算法时间 ----------------
# 输入数据已在内存里（上面的 rxbb），磁盘读写与回波生成都在计时区之外
run_once(rxbb)                       # 热身
t_algo = 0.0
for _ in range(REPS):
    t0 = time.perf_counter()
    run_once(rxbb)
    t_algo += time.perf_counter() - t0
t_algo /= REPS

print("timing  (mean of %d runs)" % REPS)
print("  %-34s %10.3f ms" % ("algorithm time (memory + compute)", t_algo * 1e3))

with open(OUT / (TAG + "_timing.json"), "w") as f:
    json.dump({
        "tag": TAG,
        "reps": REPS,
        "algo_ms": t_algo * 1e3,
    }, f, indent=2)

print("saved: %s" % RDM)
print("saved: %s" % (OUT / (TAG + "_timing.json")))
print("saved: 4 figures in %s" % OUT)
