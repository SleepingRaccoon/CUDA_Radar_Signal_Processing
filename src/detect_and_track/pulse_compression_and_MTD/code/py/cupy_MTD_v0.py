"""
脉冲压缩 + MTD 仿真（CuPy 版，按 CUDA 的步骤组织）

来源：../../doc/脉冲压缩与MTD-v01.md 第 3、4、6、7 节
说明：输入直接读 NumPy 版的产物，不再自己造数据；流程按 CUDA 版的数据流逐步写，
      每步一个显式操作、中间缓冲显式命名，方便之后一对一换成 CUDA kernel
运行：python src/detect_and_track/pulse_compression_and_MTD/code/py/cupy_MTD_v0.py
输出：src/detect_and_track/pulse_compression_and_MTD/output/py/cupy_MTD_v0_*
精度：全程 complex64，与 NumPy 版口径一致
布局：d_win 与 d_rdmap 是 rd 布局（r 在前，[Nr][...]）；存盘前转成 drmap（d 在前，[Nd][Nr]）与 NumPy 版一致
计时：REPS 次取平均，IO 时间 = 读 npy + 主存到显存 + 显存回主存 + 落盘，计算时间 = 中间九步；
      加速比由 NumPy 版留下的 <numpy_tag>_timing.json 算出
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp

OUT = Path(__file__).resolve().parents[2] / "output" / "py"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "cupy_MTD_v0"
INPUT_TAG = "numpy_MTD_v0"
REPS = 10                     # 计时重复次数，取平均

RAW_IN = OUT / (INPUT_TAG + "_raw.npy")
DRMAP_REF = OUT / (INPUT_TAG + "_drmap.npy")
TIMING_IN = OUT / (INPUT_TAG + "_timing.json")

# 绘图风格与 NumPy 版一致：衬线字体、Computer Modern 数学、刻度内向、高 dpi
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

# ---------------- 雷达参数（与 NumPy 版一致，只用于生成参考波形与坐标轴） ----------------
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

# 用于误差对照的目标真值（与 NumPy 版同一组）
targets = np.array([
    [2000.0, 40.0, -20.0, 0.8],
    [4000.0, -50.0, -20.0, 2.1],
])

range_cell = c / (2.0 * fs)
r_axis = c / 2.0 * t_start + np.arange(Nr) * range_cell
v_axis = -(lam / 2.0) * (np.arange(-(Nd // 2), Nd // 2) * fr / Nd)


def make_window(name, M):
    # 与 NumPy 版一致：返回 float32
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


# ---------------- 读 NumPy 版的仿真数据与参照结果 ----------------
drmap_numpy = np.load(DRMAP_REF)                               # [Nd][Nr]
if drmap_numpy.shape != (Nd, Nr):
    raise ValueError("drmap_numpy shape %s != (%d, %d)" % (drmap_numpy.shape, Nd, Nr))

# ---------------- 设备端预计算（对应 CUDA 的"准备"阶段） ----------------
s_tx_bb = np.exp(1j * np.pi * K * (np.arange(Ns) * Ts - Tp / 2.0) ** 2).astype(np.complex64)
d_H = cp.asarray(np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64))   # [N_fft]
d_W = cp.asarray(make_window(window_name, Nc))                              # [Nc]

# ---------------- 参数打印 ----------------
print("=" * 62)
print("pulse compression + MTD  (CuPy)")
print("=" * 62)
print("radar parameters")
for k, v in [
    ("Nc (chirps)", "%d" % Nc),
    ("Nw (window samples)", "%d" % Nw),
    ("Nr (kept range cells)", "%d" % Nr),
    ("N_fft (range FFT)", "%d" % N_fft),
    ("Nd (Doppler bins)", "%d" % Nd),
    ("Ns (pulse samples)", "%d" % Ns),
    ("window", "%s" % window_name),
    ("dtype", "complex64"),
    ("input", str(RAW_IN)),
    ("reference", str(DRMAP_REF)),
]:
    print("  %-34s %s" % (k, v))


def sync():
    cp.cuda.Stream.null.synchronize()


def compute(d_raw):
    # 计算段：补零 + 距离维匹配滤波 + 加窗转置 + 慢时间 FFT + fftshift；返回 [Nr][Nd]
    d_pad = cp.zeros((Nc, N_fft), dtype=cp.complex64)        # [Nc][N_fft]
    d_pad[:, :Nw] = d_raw
    d_spec = cp.fft.fft(d_pad, axis=1)                       # [Nc][N_fft]，快时间频域
    d_spec = d_spec * d_H[None, :]                           # 乘 H（逐点乘）
    d_pc_full = cp.fft.ifft(d_spec, axis=1)                  # [Nc][N_fft]；cupy 的 ifft 自带 1/N
    d_pc = cp.ascontiguousarray(d_pc_full[:, :Nr])           # [Nc][Nr]
    d_win = cp.ascontiguousarray((d_pc * d_W[:, None]).T)    # [Nr][Nc]，转成 rd 布局
    d_rdmap = cp.fft.fft(d_win, n=Nd, axis=1)                # [Nr][Nd]，rd 布局
    return cp.fft.fftshift(d_rdmap, axes=1)                  # [Nr][Nd]


# ---------------- 计时：热身一次丢掉，再重复 REPS 次取平均 ----------------
compute(cp.asarray(np.load(RAW_IN)))
sync()

t_load = 0.0
t_calc = 0.0
t_store = 0.0
drmap_cupy = None
for _ in range(REPS):
    t0 = time.perf_counter()
    raw = np.load(RAW_IN)                                    # 读 npy
    d_raw = cp.asarray(raw)                                  # 主存到显存
    sync()
    t1 = time.perf_counter()

    d_rdmap = compute(d_raw)                                 # 中间九步
    sync()
    t2 = time.perf_counter()

    drmap_cupy = np.ascontiguousarray(cp.asnumpy(d_rdmap).T)  # [Nd][Nr]，显存回主存并转 dr 布局
    np.save(OUT / (TAG + "_drmap.npy"), drmap_cupy)          # 落盘
    t3 = time.perf_counter()

    t_load += t1 - t0
    t_calc += t2 - t1
    t_store += t3 - t2

t_load /= REPS
t_calc /= REPS
t_store /= REPS
t_io = t_load + t_store
t_total = t_io + t_calc

# ---------------- 与 NumPy 版的误差（在 drmap 上比，统一 [Nd][Nr] 布局） ----------------
err_max = float(np.abs(drmap_cupy - drmap_numpy).max())
ref_max = float(np.abs(drmap_numpy).max())

# ---------------- 计时结果与加速比 ----------------
print("timing  (CuPy, mean of %d runs)" % REPS)
for k, v in [
    ("io in (load npy + host to device)", t_load * 1e3),
    ("compute (pad + MF + window + MTD)", t_calc * 1e3),
    ("io out (device to host + save npy)", t_store * 1e3),
    ("io total", t_io * 1e3),
    ("total (= io + compute)", t_total * 1e3),
]:
    print("  %-34s %10.3f ms" % (k, v))

if TIMING_IN.exists():
    with open(TIMING_IN) as f:
        ref = json.load(f)
    print("speedup vs NumPy  (%s)" % TIMING_IN.name)
    for k in ("total", "compute", "io"):
        a = ref[k + "_ms"]
        b = t_total * 1e3 if k == "total" else (t_calc * 1e3 if k == "compute" else t_io * 1e3)
        print("  %-34s %10.3f / %8.3f ms = %7.1f x" % (k, a, b, a / b))
else:
    print("speedup vs NumPy  (missing %s, run numpy_MTD_v0.py first)" % TIMING_IN.name)

print("error on drmap [Nd][Nr]  (CuPy vs NumPy)")
print("  %-34s %.6e" % ("max absolute error", err_max))
print("  %-34s %.6e" % ("max relative error", err_max / ref_max))

print("self-check  (peak near each target, CuPy vs NumPy)")
for i, t in enumerate(targets):
    r_win = (r_axis >= t[0] - 5.0 * range_cell) & (r_axis <= t[0] + 5.0 * range_cell)
    out = []
    for name, m in (("NumPy", np.abs(drmap_numpy)), ("CuPy", np.abs(drmap_cupy))):
        sub = m[:, r_win]
        iv, ir = np.unravel_index(int(np.argmax(sub)), sub.shape)
        out.append("%s R = %8.1f m v = %+7.2f m/s" % (name, r_axis[r_win][ir], v_axis[iv]))
    print("  target %d   %s   |   %s" % (i + 1, out[0], out[1]))

print("saved: %s" % (OUT / (TAG + "_drmap.npy")))

# ---------------- 四张图（drmap 的图，画 [Nd][Nr] 的转置使横轴为速度） ----------------
mag = np.abs(drmap_cupy)
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
