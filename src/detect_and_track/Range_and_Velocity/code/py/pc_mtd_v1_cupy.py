"""
脉冲压缩 + MTD（CuPy 版，GPU）

来源：../../doc/PC_and_MTD_v01.md 第 3、4、6、7 节
运行：python src/detect_and_track/Range_and_Velocity/code/py/pc_mtd_v1_cupy.py
本程序是 CUDA 版的结构草图：每一步一个显式操作，与 cuda 版的一个 kernel 或一次
cufft 调用一一对应，不做合并；FFT 走底层 cupy.cuda.cufft，与 cuFFT 行为完全一致
输入：output/py/pc_mtd_v1_rxbb.npy（pc_mtd_v1.py 生成的仿真回波基带，[Nc][Nw] complex64）
输出：output/py/pc_mtd_v1_cupy_rdm.npy（距离-多普勒图 [Nr][Nd] complex64）与四张图
布局：rd 布局 [Nr][Nd]，与 CPU 版一致
计时：REPS 次取平均，算法时间 = 通信（主机内存 ↔ 显存）+ 计算（九步，含显存访存）；
      磁盘读写在计时区外；每段前后都先同步再读时钟；LFM 频谱是常驻量，预先算好，不计时
参数与 pc_mtd_v1.py 严格一致，改一处必须两边一起改
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
from cupy.cuda import cufft

REPS = 100

OUT = Path(__file__).resolve().parents[2] / "output" / "py"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "pc_mtd_v1_cupy"
REF_TAG = "pc_mtd_v1"
RXBB = OUT / (REF_TAG + "_rxbb.npy")
RDM_REF = OUT / (REF_TAG + "_rdm.npy")
RDM = OUT / (TAG + "_rdm.npy")
TIMING_REF = OUT / (REF_TAG + "_timing.json")

# IEEE 风格绘图：与 CPU 版一致
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

# ---------------- 系统参数（与 pc_mtd_v1.py 严格一致） ----------------
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

# ---------------- 目标真值（只用于自校验，不参与计算） ----------------
targets = np.array([
    [2000.0, 40.0],
    [4000.0, -50.0],
])

# ---------------- 派生参数 ----------------
range_res = c / (2.0 * B)
range_cell = c / (2.0 * fs)
vel_res = lam * fr / (2.0 * Nc)
vel_cell = lam * fr / (2.0 * Nd)
R_ua = c / 2.0 * (t_end - Tp)
R_min = c / 2.0 * t_start
v_ua = lam * fr / 4.0

r_axis = c / 2.0 * t_start + np.arange(Nr) * range_cell
v_axis = -(lam / 2.0) * (np.arange(-(Nd // 2), Nd // 2) * fr / Nd)

# ---------------- 打印系统参数与派生参数 ----------------
print("=" * 62)
print("pulse compression + MTD  (CuPy)")
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
    ("window", "hann"),
    ("dtype", "complex64"),
    ("input", str(RXBB)),
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

# ---------------- 常驻量准备（不计时） ----------------
# LFM 频谱：主机侧用 numpy 算好再拷到设备，等价于 cuda 版的准备阶段
s_tx_bb = np.exp(1j * np.pi * K * (np.arange(Ns) * Ts - Tp / 2.0) ** 2).astype(np.complex64)
d_H = cp.asarray(np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64))   # [N_fft]
# 慢时间加窗：周期汉宁窗
d_W = cp.asarray(np.hanning(Nc + 1)[:-1].astype(np.float32))                # [Nc]
SCALE = np.float32(1.0 / N_fft)                                            # 逆变换要乘的 1/N
# 两个批量 FFT 计划：距离维 batch=Nc 点 N_fft，慢时间 batch=Nr 点 Nd
plan_range = cufft.Plan1d(N_fft, cufft.CUFFT_C2C, Nc)
plan_dopp = cufft.Plan1d(Nd, cufft.CUFFT_C2C, Nr)

rdm_cpu = np.load(RDM_REF)                                                 # [Nr][Nd]，参照


def sync():
    cp.cuda.Stream.null.synchronize()


def run_once(d_rxbb):
    # 1 补零到 N_fft：对应 cuda 版的 memset + pad_kernel
    d_pad = cp.zeros((Nc, N_fft), dtype=cp.complex64)
    d_pad[:, :Nw] = d_rxbb

    # 2 距离维正变换：对应 cufftExecC2C FORWARD（batch=Nc）
    d_spec = cp.empty((Nc, N_fft), dtype=cp.complex64)
    cufft.execC2C(plan_range.handle, d_pad.data.ptr, d_spec.data.ptr, cufft.CUFFT_FORWARD)

    # 3 乘 H：对应 mul_H_kernel
    d_spec *= d_H[None, :]

    # 4 距离维逆变换：对应 cufftExecC2C INVERSE（cuFFT 不除 N）
    d_pc_full = cp.empty((Nc, N_fft), dtype=cp.complex64)
    cufft.execC2C(plan_range.handle, d_spec.data.ptr, d_pc_full.data.ptr, cufft.CUFFT_INVERSE)

    # 5 乘 1/N_fft 并截取前 Nr 个距离单元：对应 scale_extract_kernel
    d_pc = cp.empty((Nc, Nr), dtype=cp.complex64)
    cp.multiply(d_pc_full[:, :Nr], SCALE, out=d_pc)

    # 6 加窗 + 转置成 rd 布局：对应 window_transpose_kernel
    d_win = cp.ascontiguousarray((d_pc * d_W[:, None]).T)      # [Nr][Nc]

    # 7 慢时间补零到 Nd：对应 memset + pad_kernel
    d_win_pad = cp.zeros((Nr, Nd), dtype=cp.complex64)
    d_win_pad[:, :Nc] = d_win

    # 8 慢时间正变换：对应 cufftExecC2C FORWARD（batch=Nr）
    d_rdm = cp.empty((Nr, Nd), dtype=cp.complex64)
    cufft.execC2C(plan_dopp.handle, d_win_pad.data.ptr, d_rdm.data.ptr, cufft.CUFFT_FORWARD)

    # 9 fftshift：对应 fftshift_kernel
    return cp.fft.fftshift(d_rdm, axes=1)


# ---------------- 第一次执行：这一次的结果就是输出 ----------------
rxbb = np.load(RXBB)                     # 磁盘读，计时区外
d_rxbb = cp.asarray(rxbb)
sync()
d_rdm = run_once(d_rxbb)
sync()
rdm = cp.asnumpy(d_rdm)
np.save(RDM, rdm)                        # 磁盘写，计时区外

mag = np.abs(rdm)
print("self-check  (peak vs ground truth)")
for i, t in enumerate(targets):
    r_win = (r_axis >= t[0] - 5.0 * range_cell) & (r_axis <= t[0] + 5.0 * range_cell)
    sub = mag[r_win, :]
    ir, iv = np.unravel_index(int(np.argmax(sub)), sub.shape)
    print("  target %d   true R = %8.1f m  v = %+7.2f m/s   measured R = %8.1f m  v = %+7.2f m/s"
          % (i + 1, t[0], t[1], r_axis[r_win][ir], v_axis[iv]))

print("error on rdm [Nr][Nd]  (CuPy vs CPU)")
print("  %-34s %.6e" % ("max absolute error", float(np.abs(rdm - rdm_cpu).max())))
print("  %-34s %.6e" % ("max relative error", float(np.abs(rdm - rdm_cpu).max() / np.abs(rdm_cpu).max())))

# ---------------- 四张图（rd 布局，与 CPU 版一致） ----------------
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

# ---------------- 计时：REPS 次取平均 ----------------
# 算法时间 = 通信（主机内存 ↔ 显存）+ 计算（九步，含显存访存）
# 输入数据已在内存里（上面的 rxbb），磁盘读写都在计时区之外
# GPU 是异步的，每段前后都先同步再读时钟，否则读到的是"排队完成"而不是"算完"
d_rxbb = cp.asarray(rxbb)                # 热身（含首次 H2D）
sync()
run_once(d_rxbb)
sync()

t_comm = 0.0
t_calc = 0.0
for _ in range(REPS):
    t0 = time.perf_counter()
    d_rxbb = cp.asarray(rxbb)            # 主存到显存
    sync()
    t1 = time.perf_counter()

    d_rdm = run_once(d_rxbb)             # 中间九步
    sync()
    t2 = time.perf_counter()

    rdm = cp.asnumpy(d_rdm)              # 显存回主存
    sync()
    t3 = time.perf_counter()

    t_comm += (t1 - t0) + (t3 - t2)
    t_calc += t2 - t1

t_comm /= REPS
t_calc /= REPS
t_algo = t_comm + t_calc

print("timing  (CuPy, mean of %d runs)" % REPS)
for k, v in [
    ("io (host to device + device to host)", t_comm * 1e3),
    ("compute (9 steps, kernel + cufft)", t_calc * 1e3),
    ("algorithm time (= io + compute)", t_algo * 1e3),
]:
    print("  %-34s %10.3f ms" % (k, v))

with open(OUT / (TAG + "_timing.json"), "w") as f:
    json.dump({
        "tag": TAG,
        "reps": REPS,
        "io_ms": t_comm * 1e3,
        "compute_ms": t_calc * 1e3,
        "algo_ms": t_algo * 1e3,
    }, f, indent=2)
print("saved: %s" % (OUT / (TAG + "_timing.json")))

if TIMING_REF.exists():
    with open(TIMING_REF) as f:
        ref = json.load(f)
    a = ref["algo_ms"]
    print("speedup vs CPU  (%s)" % TIMING_REF.name)
    print("  %-34s %10.3f / %8.3f ms = %7.1f x" % ("total time", a, t_algo * 1e3, a / (t_algo * 1e3)))
    print("  %-34s %10.3f / %8.3f ms = %7.1f x" % ("compute time", a, t_calc * 1e3, a / (t_calc * 1e3)))
else:
    print("speedup vs CPU  (missing %s, run pc_mtd_v1.py first)" % TIMING_REF.name)

print("saved: %s" % RDM)
print("saved: 4 figures in %s" % OUT)
