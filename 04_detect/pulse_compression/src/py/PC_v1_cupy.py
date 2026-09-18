"""
脉冲压缩 + MTD（CuPy 版，GPU 库基线）

来源：04_detect/pulse_compression/doc/PC_v1_math.md（原理）、PC_v1_eng.md（约定与计时口径）
运行：python 04_detect/pulse_compression/src/py/PC_v1_cupy.py
定位：GPU 库基线。目标是用现成库（cp.fft + CuPy 算子）给出这个算法在 GPU 上能达到的
      最快时间，不刻意与 CUDA 版保持步骤一一对应，也不使用 RawKernel 或手写核——
      把加窗+转置的访存遍数从 4 遍压到 2 遍的融合核是 PC_v1.cu 的活，
      两个版本的差值就是"手写核值多少"。
输入：04_detect/pulse_compression/output/py/PC_v1_rxbb.npy（PC_v1.py 生成的回波基带 [Nc][Nw] complex64）
      04_detect/pulse_compression/output/py/PC_v1_rdmap.npy（CPU 参照，[Nr][Nd] complex64）
      04_detect/pulse_compression/output/py/PC_v1_timing.json（CPU 计时参照）
输出：04_detect/pulse_compression/output/py/PC_v1_cupy_rdmap.npy（距离-多普勒图 [Nr][Nd] complex64）
      04_detect/pulse_compression/output/py/PC_v1_cupy_timing.json 与四张图
布局：rd = [Nr][Nd]（r 在外、d 在内），与 CPU 版一致；dr = [Nd][Nr]
计时：REPS 次取平均。算法时间不含 IO：读原始回波、写 RD 矩阵、绘图、写 json 都在计时区外，
      常驻量（LFM 频谱、Hann 窗、pinned 缓冲、补零缓冲）只准备一次。
      GPU 分两项：copy = 显存↔内存拷贝（H2D + D2H），compute = 库函数与 cuFFT，
      算法总时间 = copy + compute。两段都用 CUDA event 计时，每轮末尾同步一次再读。
      参数与 PC_v1.py 严格一致，改一处必须两边一起改。

本版取代了原来的九步版（九步版刻意与 CUDA 版一一对应，事后判断没有必要，见决策记录）。
相对九步版的改动（都是库层面的，不涉及手写核）：
  1. 九步合并成 5 步：距离维 FFT / 乘 H / 距离维 IFFT+截取 / 加窗+转置 / 慢时间 FFT+fftshift；
     FFT 走 cp.fft 高级接口，1/N 归一化由 ifft 负责，不再手写缩放。
  2. Nd == Nc 时不补零（原实现每轮 memset 2.1 MB + 满拷贝 2.1 MB，纯浪费）。
  3. H2D 走 pinned + memcpy2DAsync，直接写进补零缓冲的前 Nw 列，省掉设备侧的 pad 拷贝；
     D2H 走 pinned + copy_to_host_async。
  4. 加窗与转置的顺序：实测两种顺序在 GPU 上等价（0.3104 对 0.3102 ms），CPU 上差约 5%，
     本版按可读性选"先乘后转"（与 CPU 版同形）。最初测到的 10% 差异是热身不足的假象。
  5. 补零缓冲只分配一次，尾部保持 0，每轮只覆盖前 Nw 列。
  6. 计时前先热身够 0.3 s：GPU 是 DVFS 的（空闲 210 MHz、满载 1500-2100 MHz），
     短工作负载会让数字严重偏慢；统计量改用中位数，并打印 GPU 时钟以便判断可比性。
"""

import ctypes
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp

REPS = 100

OUT = Path(__file__).resolve().parents[2] / "output" / "py"
OUT.mkdir(parents=True, exist_ok=True)
TAG = "PC_v1_cupy"
REF_TAG = "PC_v1"
RXBB = OUT / (REF_TAG + "_rxbb.npy")
RDMAP_REF = OUT / (REF_TAG + "_rdmap.npy")
RDMAP = OUT / (TAG + "_rdmap.npy")
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


def gpu_clocks():
    """GPU 型号/驱动/当前时钟：数字可比性直接依赖这些（项目建议第 6 节的锁频纪律）。"""
    fields = "name,driver_version,clocks.current.graphics,clocks.max.graphics"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        name, driver, cur, mx = [v.strip() for v in out.split(",")]
        return "%s, driver %s, clock %s / max %s" % (name, driver, cur, mx)
    except Exception as e:
        return "nvidia-smi unavailable (%s)" % e


# ---------------- 系统参数（与 PC_v1.py 严格一致） ----------------
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

# ---------------- 慢时间加窗与信噪比增益（与 CPU 版同一套账） ----------------
W = np.hanning(Nc + 1)[:-1].astype(np.float32)           # 周期汉宁窗 [Nc]
win_coh = float(W.sum() / Nc)                            # 相干增益（幅度），Hann = 0.5
win_gain = float(W.sum() ** 2 / (Nc * np.sum(W ** 2)))   # 处理增益相对无窗，Hann = 2/3
gain_pc = float(Ns)                                      # 脉压：相干积累 Ns 个采样
gain_fft = float(Nc)                                     # MTD：Nc 点相干积累
gain_total = gain_pc * gain_fft * win_gain               # 总信噪比增益

ITEM = np.dtype(np.complex64).itemsize                   # 8 字节

# ---------------- 打印系统参数与派生参数 ----------------
print("=" * 62)
print("pulse compression + MTD  (CuPy, library baseline)")
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
    ("host buffers", "pinned (H2D + D2H async)"),
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

print("SNR gain  (per-sample input SNR -> RD peak)")
for k, g in [
    ("pulse compression  Ns", gain_pc),
    ("MTD (Doppler FFT)  Nc", gain_fft),
    ("window (Hann)  (sum w)^2/(N sum w^2)", win_gain),
    ("total = Ns * Nc * window", gain_total),
]:
    print("  %-38s %10.4f   %9.3f dB" % (k, g, 10.0 * np.log10(g)))
print("  %-38s %10.4f   %9.3f dB   amplitude, not in the SNR budget"
      % ("(window coherent gain  sum w/N)", win_coh, 20.0 * np.log10(win_coh)))

# ---------------- 常驻量与设备缓冲（不计时） ----------------
# 匹配滤波频响：主机侧算好再上传，与 PC_v1.py 的 H 逐位一致（相关形式，峰值索引即回波起始）
s_tx_bb = np.exp(1j * np.pi * K * (np.arange(Ns) * Ts - Tp / 2.0) ** 2).astype(np.complex64)
d_H = cp.asarray(np.conj(np.fft.fft(s_tx_bb, N_fft)).astype(np.complex64))   # [N_fft]
d_W = cp.asarray(W)                                                         # [Nc]

# 补零缓冲：只分配一次，尾部永远为 0，每轮只覆盖前 Nw 列
d_pad = cp.zeros((Nc, N_fft), dtype=cp.complex64)


def pinned_like(shape, dtype):
    """分配一块 pinned 主机内存，并返回 (持有者, numpy 视图)。

    cp.cuda.PinnedMemory 只暴露 .ptr/.size，没有 buffer 协议，所以 memoryview() 和
    np.frombuffer(mem) 都会报 "a bytes-like object is required"；先用 ctypes 在同一个
    地址上造一个 bytes-like 对象，再交给 np.frombuffer，就得到零拷贝的 ndarray 视图。
    返回值里的 mem 必须活着，否则视图指向的内存会被释放。
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.PinnedMemory(nbytes, cp.cuda.runtime.hostAllocPortable)
    buf = (ctypes.c_char * nbytes).from_address(mem.ptr)
    return mem, np.frombuffer(buf, dtype=dtype).reshape(shape)


mem_in, h_rxbb = pinned_like((Nc, Nw), np.complex64)
mem_out, h_out = pinned_like((Nr, Nd), np.complex64)

# ---------------- 待测数据（磁盘读，计时区外） ----------------
rxbb = np.load(RXBB)
np.copyto(h_rxbb, rxbb)                     # pinned 缓冲只填一次，每轮内容不变
rdmap_cpu = np.load(RDMAP_REF)              # [Nr][Nd]，CPU 参照


def copy_in():
    """H2D：pinned -> 显存，直接写进补零缓冲的前 Nw 列（行间距 N_fft）。

    异步拷贝显式挂在"当前流"上：同一条流里的后续核函数天然排在它后面执行，
    所以不需要额外同步——这正是用 cudaMemcpyAsync 而不是 cudaMemcpy 的意义所在。
    如果这里写死 stream=0（legacy 默认流）而后面又在别的流上跑核函数，就会真的抢跑。
    """
    cp.cuda.runtime.memcpy2DAsync(d_pad.data.ptr, N_fft * ITEM,
                                  mem_in.ptr, Nw * ITEM,
                                  Nw * ITEM, Nc,
                                  cp.cuda.runtime.memcpyHostToDevice,
                                  cp.cuda.get_current_stream().ptr)


def run_compute():
    """脉压 + MTD 主体：[Nc][Nw] -> [Nr][Nd]，五步库调用。"""
    d_spec = cp.fft.fft(d_pad, axis=1)                      # 距离维正变换
    d_spec *= d_H                                           # 乘匹配滤波频响（原地）
    d_pc = cp.fft.ifft(d_spec, axis=1)[:, :Nr]              # 逆变换 + 截取（ifft 自带 1/N）
    d_win = cp.ascontiguousarray((d_pc * d_W[:, None]).T)   # 加窗 + 转置 -> [Nr][Nc]
    drd = cp.fft.fft(d_win, n=Nd, axis=1)                   # 慢时间 FFT（Nd == Nc 时无需补零）
    return cp.fft.fftshift(drd, axes=1)                     # fftshift -> [Nr][Nd]


# ---------------- 第一次执行：这一次的结果就是输出 ----------------
copy_in()
d_rdmap = run_compute()
d_rdmap.data.copy_to_host_async(mem_out.ptr, d_rdmap.nbytes)
cp.cuda.Stream.null.synchronize()
# h_out 只是 pinned 暂存区，计时区里每轮都会被覆盖，所以在这里拷出一份正常持有内存的结果
rdmap = np.array(h_out)
np.save(RDMAP, rdmap)

mag = np.abs(rdmap)
print("self-check  (peak vs ground truth)")
for i, t in enumerate(targets):
    r_win = (r_axis >= t[0] - 5.0 * range_cell) & (r_axis <= t[0] + 5.0 * range_cell)
    sub = mag[r_win, :]
    ir, iv = np.unravel_index(int(np.argmax(sub)), sub.shape)
    print("  target %d   true R = %8.1f m  v = %+7.2f m/s   measured R = %8.1f m  v = %+7.2f m/s"
          % (i + 1, t[0], t[1], r_axis[r_win][ir], v_axis[iv]))

print("error on rdmap [Nr][Nd]  (CuPy vs CPU)")
print("  %-34s %.6e" % ("max absolute error", float(np.abs(rdmap - rdmap_cpu).max())))
print("  %-34s %.6e" % ("max relative error", float(np.abs(rdmap - rdmap_cpu).max() / np.abs(rdmap_cpu).max())))

# ---------------- 四张图（rd 布局，与 CPU 版一致） ----------------
db = np.clip(20.0 * np.log10(mag / mag.max() + 1e-12), -60.0, 0.0)
ext = [v_axis[0], v_axis[-1], r_axis[0] / 1e3, r_axis[-1] / 1e3]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(mag, origin="lower", aspect="auto", extent=ext, cmap="viridis", interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$")
fig.savefig(OUT / (TAG + "_rdmap_2d.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
im = ax.imshow(db, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=-60.0, vmax=0.0, interpolation="nearest")
ax.set_xlabel(r"$v$ (m/s)")
ax.set_ylabel(r"$R$ (km)")
fig.colorbar(im, ax=ax).set_label(r"$|s_{\mathrm{RD}}|$ (dB)")
fig.savefig(OUT / (TAG + "_rdmap_2d_dB.png"))
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
    fig.savefig(OUT / (TAG + "_rdmap_3d%s.png" % tag))
    plt.close(fig)

# ---------------- 计时：REPS 次取中位数 ----------------
# 算法时间 = copy（显存↔内存拷贝）+ compute（库函数与 cuFFT）
# 磁盘读写、绘图、写 json 都在计时区之外；输入每轮内容不变，与 CPU 版口径一致
# 每段用 CUDA event 计时，每轮末尾同步一次再读时间，避免读到"排队完成"
ev = [cp.cuda.Event() for _ in range(4)]


def elapsed_ms(start, end):
    """两个 CUDA event 之间的毫秒数（CuPy 的 Event 不提供该接口，走 runtime）。"""
    return cp.cuda.runtime.eventElapsedTime(start.ptr, end.ptr)


def timed_round():
    ev[0].record()
    copy_in()
    ev[1].record()
    d_out = run_compute()
    ev[2].record()
    d_out.data.copy_to_host_async(mem_out.ptr, d_out.nbytes)
    ev[3].record()
    ev[3].synchronize()
    return (elapsed_ms(ev[0], ev[1]),       # H2D
            elapsed_ms(ev[1], ev[2]),       # compute
            elapsed_ms(ev[2], ev[3]))       # D2H


print("environment")
print("  %-34s %s" % ("gpu / driver / clocks (idle)", gpu_clocks()))
print("  %-34s cupy %s, cuda runtime %d"
      % ("software", cp.__version__, cp.cuda.runtime.runtimeGetVersion()))


def warm_up(seconds=0.3):
    """计时前先把设备持续占住 seconds 秒。

    注意这不是"预热 JIT / cuFFT plan"那种热身（那只在进程内有效一两次），而是 DVFS：
    空闲时钟只有满频的十分之一（实测 210 / 2100 MHz），短工作负载会让读到的数字成倍偏慢。
    0.3 s 是按这张卡实测定的，换卡或换电源计划要重新标定。
    """
    n = 0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        timed_round()
        n += 1
    print("warm up  %d rounds in %.2f s" % (n, seconds))
    print("  %-34s %s" % ("gpu clocks after warm up", gpu_clocks()))


warm_up(0.3)

samples = np.asarray([timed_round() for _ in range(REPS)])   # [REPS][3]，单位 ms
t_h2d, t_calc, t_d2h = samples.T
t_copy = t_h2d + t_d2h
t_algo = t_copy + t_calc


def median(a):
    return float(np.median(a))


print("timing  (CuPy, median of %d runs)" % REPS)
for k, a in [
    ("copy (host to device + device to host)", t_copy),
    ("compute (library calls + cufft)", t_calc),
    ("algorithm time (= copy + compute)", t_algo),
]:
    print("  %-34s %10.3f ms   (min %.3f  max %.3f)" % (k, median(a), a.min(), a.max()))
print("  %-34s %s" % ("gpu clocks after timing", gpu_clocks()))

with open(OUT / (TAG + "_timing.json"), "w") as f:
    json.dump({
        "tag": TAG,
        "reps": REPS,
        "copy_ms": median(t_copy),
        "copy_min_ms": float(t_copy.min()),
        "copy_max_ms": float(t_copy.max()),
        "compute_ms": median(t_calc),
        "compute_min_ms": float(t_calc.min()),
        "compute_max_ms": float(t_calc.max()),
        "algo_ms": median(t_algo),
        "algo_min_ms": float(t_algo.min()),
        "algo_max_ms": float(t_algo.max()),
        "gpu": gpu_clocks(),
    }, f, indent=2)
print("saved: %s" % (OUT / (TAG + "_timing.json")))

if TIMING_REF.exists():
    with open(TIMING_REF) as f:
        ref = json.load(f)
    a = ref["algo_ms"]
    print("speedup vs CPU  (%s)" % TIMING_REF.name)
    print("  %-34s %10.3f / %8.3f ms = %7.1f x" % ("total time", a, median(t_algo), a / median(t_algo)))
    print("  %-34s %10.3f / %8.3f ms = %7.1f x" % ("compute time", a, median(t_calc), a / median(t_calc)))
else:
    print("speedup vs CPU  (missing %s, run PC_v1.py first)" % TIMING_REF.name)

print("saved: %s" % RDMAP)
print("saved: 4 figures in %s" % OUT)
