"""
脉冲压缩 + MTD 仿真（NumPy 版，高度向量化，MTI 可开关）

来源：../../doc/脉冲压缩与MTD-v01.md 第 3、4、5、6 节
运行：python src/detect_and_track/pulse_compression_and_MTD/code/py/numpy_MTD_v0.py [MTI]
      MTI = 0（默认）只做脉压 + MTD，产物前缀 numpy_MTD_v0_
      MTI = 1 先做双对消再脉压输出加窗做 MTD，产物前缀 numpy_MTI_v0_
输出：src/detect_and_track/pulse_compression_and_MTD/output/py/numpy_<MTD|MTI>_v0_*
精度：采样数据与整条处理链都用 32 位（complex64）；只有回波的相位生成用 float64，
      因为 fc·τ 量级到 1e5~1e6 rad，float32 存不下那个差
布局：缓冲区名带内存布局，drmap 表示 d 在前 r 在后（[Nd][Nr]），rdmap 反之（[Nr][Nd]）
计时：REPS 次取平均；IO 时间 = 落盘，计算时间 = 脉压 + 加窗 + MTD；
      回波生成不计入时间（各实现统一这个口径）；另存 timing json 供 CuPy 版算加速比
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MTI 开关：0 = 不做 MTI；1 = 双对消（可被命令行参数覆盖）
MTI = 0
if len(sys.argv) > 1:
    MTI = int(sys.argv[1])
if MTI not in (0, 1):
    raise ValueError("MTI must be 0 or 1")

OUT = Path(__file__).resolve().parents[2] / "output" / "py"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "numpy_MTD_v0" if MTI == 0 else "numpy_MTI_v0"
REPS = 10                    # 计时重复次数，取平均

# IEEE 风格绘图：衬线字体，数学符号用 Computer Modern，刻度内向；dpi 拉高保证细节
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

# ---------------- 雷达参数 ----------------
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
N_fft = int(2 ** np.ceil(np.log2(Nw)))   # 只要 >= Nw；写法二才需要 >= Ns+Nw-1
window_name = "hamming"
sigma = 1.0                  # 强制规定：噪声标准差为 1，即每采样点方差 sigma^2 = 1
seed = 42

# ---------------- 场景 ----------------
# 每个目标：[距离 m, 速度 m/s（远离为正）, 回波幅度 dB, 初相 rad]
# 幅度 A = sigma * 10^(amp/20)
targets = np.array([
    [2000.0, 40.0, -20.0, 0.8],
    [4000.0, -50.0, -20.0, 2.1],
])
amp = sigma * 10.0 ** (targets[:, 2] / 20.0)

# ---------------- 派生量 ----------------
range_res = c / (2.0 * B)            # 距离分辨率
range_cell = c / (2.0 * fs)          # 距离单元格
vel_res = lam * fr / (2.0 * Nc)      # 速度分辨率（真实脉冲数决定）
vel_cell = lam * fr / (2.0 * Nd)     # 速度单元格（FFT 点数决定）
R_ua = c / 2.0 * (t_end - Tp)        # 回波完整落在窗内（延时 <= t_end-Tp）的最大距离
R_min = c / 2.0 * t_start            # 窗起点对应的最小距离
v_ua = lam * fr / 4.0                # 最大不模糊速度

r_axis = c / 2.0 * t_start + np.arange(Nr) * range_cell
v_axis = -(lam / 2.0) * (np.arange(-(Nd // 2), Nd // 2) * fr / Nd)


def make_window(name, M):
    # 加窗接口：默认 hamming，可切 rect/hann/blackman；返回 float32，与 complex64 数据相乘不升精度
    if name == "rect":
        w = np.ones(M)
    elif name == "hann":
        w = np.hanning(M + 1)[:-1]
    elif name == "hamming":
        w = np.hamming(M + 1)[:-1]
    elif name == "blackman":
        w = np.blackman(M + 1)[:-1]
    else:
        raise ValueError("unknown window: " + name)
    return w.astype(np.float32)


# ---------------- 参数打印 ----------------
print("=" * 62)
print("pulse compression + MTD  (NumPy, MTI = %d)" % MTI)
print("=" * 62)
print("radar parameters")
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
    ("sigma (noise std, fixed)", "%.3f" % sigma),
    ("MTI", "none" if MTI == 0 else "double canceller"),
    ("window", "%s (M=%d)" % (window_name, Nc - 2 * MTI)),
    ("dtype", "complex64"),
]:
    print("  %-34s %s" % (k, v))

print("derived metrics")
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

print("scene  (A = sigma * 10^(amp/20), sigma = %.2f)" % sigma)
for i, t in enumerate(targets):
    print("  target %d   R = %8.1f m   v = %+7.2f m/s   amp = %+5.1f dB   A = %.5f" % (i + 1, t[0], t[1], t[2], amp[i]))

# ---------------- 仿真：回波生成（向量化；按约定不计入时间） ----------------
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

# 复高斯白噪声，每个采样点方差 sigma^2 = 1；采样数据落到 complex64
raw = (rx + (sigma / np.sqrt(2.0)) * (rng.standard_normal((Nc, Nw)) + 1j * rng.standard_normal((Nc, Nw)))).astype(np.complex64)


def run_once():
    # 一遍完整处理链：脉压 +（可选 MTI）+ 加窗 + MTD；返回 drmap [Nd][Nr]
    H = np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64)
    raw_pad = np.zeros((Nc, N_fft), dtype=np.complex64)
    raw_pad[:, :Nw] = raw
    # H = conj(FFT(s_tx_bb)) 是相关形式，峰值索引直接对应回波起始，故从 0 截取
    pc_full = np.fft.ifft(np.fft.fft(raw_pad, axis=1) * H[None, :], axis=1).astype(np.complex64)
    pc = pc_full[:, 0:Nr]          # [Nc][Nr]

    if MTI == 0:
        x_mti = pc
    else:
        # 双对消：线性（非循环）对消，丢掉前两行，输出 Nc-2 行
        x_mti = pc[2:, :] - 2.0 * pc[1:-1, :] + pc[:-2, :]
    M = x_mti.shape[0]
    pc_win = x_mti * make_window(window_name, M)[:, None]                                   # [M][Nr]
    drmap = np.fft.fftshift(np.fft.fft(pc_win, n=Nd, axis=0), axes=0).astype(np.complex64)  # [Nd][Nr]
    return drmap


# ---------------- 计时：热身一次丢掉，再重复 REPS 次取平均 ----------------
run_once()
t_calc = 0.0
t_io = 0.0
drmap = None
for _ in range(REPS):
    t0 = time.perf_counter()
    drmap = run_once()
    t_calc += time.perf_counter() - t0
    t1 = time.perf_counter()
    np.save(OUT / (TAG + "_raw.npy"), raw)
    np.save(OUT / (TAG + "_drmap.npy"), drmap)
    t_io += time.perf_counter() - t1
t_calc /= REPS
t_io /= REPS
t_total = t_calc + t_io

# ---------------- 自检：峰值位置与真值对比 ----------------
mag = np.abs(drmap)
print("self-check  (peak of drmap vs ground truth)")
for i, t in enumerate(targets):
    r_win = (r_axis >= t[0] - 5.0 * range_cell) & (r_axis <= t[0] + 5.0 * range_cell)
    sub = mag[:, r_win]
    iv, ir = np.unravel_index(int(np.argmax(sub)), sub.shape)
    print("  target %d   true R = %8.1f m  v = %+7.2f m/s   measured R = %8.1f m  v = %+7.2f m/s"
          % (i + 1, t[0], t[1], r_axis[r_win][ir], v_axis[iv]))

# ---------------- 计时结果 ----------------
print("timing  (mean of %d runs)" % REPS)
for k, v in [
    ("io (save npy)", t_io * 1e3),
    ("compute (range MF + window + MTD)", t_calc * 1e3),
    ("total (= io + compute)", t_total * 1e3),
]:
    print("  %-34s %10.3f ms" % (k, v))

with open(OUT / (TAG + "_timing.json"), "w") as f:
    json.dump({
        "tag": TAG,
        "reps": REPS,
        "io_ms": t_io * 1e3,
        "compute_ms": t_calc * 1e3,
        "total_ms": t_total * 1e3,
    }, f, indent=2)
print("saved: %s" % (OUT / (TAG + "_timing.json")))

# ---------------- 四张图（drmap 的图，画 [Nd][Nr] 的转置使横轴为速度） ----------------
db = np.clip(20.0 * np.log10(mag / mag.max() + 1e-12), -60.0, 0.0)
ext = [v_axis[0], v_axis[-1], r_axis[0] / 1e3, r_axis[-1] / 1e3]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(mag.T, origin="lower", aspect="auto", extent=ext, cmap="viridis", interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$")
fig.savefig(OUT / (TAG + "_drmap_2d_linear.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(db.T, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=-60.0, vmax=0.0, interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$ (dB)")
fig.savefig(OUT / (TAG + "_drmap_2d_db.png"))
plt.close(fig)

for tag, z, zlabel, zlim in (("linear", mag.T, r"$|s_{\mathrm{RD}}|$", None), ("db", db.T, r"$|s_{\mathrm{RD}}|$ (dB)", (-60.0, 0.0))):
    fig = plt.figure(figsize=(4.2, 3.4))
    ax = fig.add_subplot(projection="3d")
    Vs, Rs = np.meshgrid(v_axis, r_axis / 1e3)
    ax.plot_surface(Vs, Rs, z, rstride=4, cstride=1, cmap="viridis", linewidth=0, antialiased=False)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel(r"$R$ (km)")
    ax.set_zlabel(zlabel)
    if zlim is not None:
        ax.set_zlim(*zlim)
    fig.savefig(OUT / (TAG + "_drmap_3d_%s.png" % tag))
    plt.close(fig)

print("saved: 4 figures in %s" % OUT)
