# PC_v1 工程设计手册

三个实现（NumPy / CuPy / CUDA）共用的数据契约、代码约定与运行方式。
原理见 `PC_v1_math.md`，实测与决策记录见 `PC_v1_res.md`。

# 1 输入与符号

输入是 \(N_c\times N_w\) 行主序原始数据矩阵 `raw[Nc][Nw]`——一行是一个脉冲的接收窗采样，
列是距离维快时间——以及数字 LFM `s_tx_bb` 的离散频谱（雷达系统常驻量）。

| 符号 | 含义 |
| --- | --- |
| \(N_s\) | 数字 LFM 的采样点数，`Ns = floor(Tp*fs)` |
| \(N_w\) | 采样窗内的点数，`Nw = floor((t_end-t_start)*fs)` |
| \(N_c\) | 实际发射的 chirp 数 |
| \(N_d\) | 慢时间维实际做 FFT 的点数 |
| \(N_r\) | 距离维实际保留的点数 |
| \(N_{fft}\) | 距离维 FFT 点数，`>= Nw`，取 2 的幂 |

# 2 布局约定

行主序，前闭后开。

| 记号 | 布局 | 说明 |
| --- | --- | --- |
| dr | `[Nd][Nr]` | 多普勒在外、距离在内，采样点的自然顺序 |
| rd | `[Nr][Nd]` | 距离在外、多普勒在内，**本模块统一输出** |

最终产物一律 `rdmap[Nr][Nd]`。距离维压完的 `pc` 保持 `[Nc][Nr]`，转置成 `[Nr][Nc]` 之后沿行做
慢时间 FFT，输出天然就是 rd 布局，不需要再补一次转置。

# 3 距离维匹配滤波的两种写法

频响与截取方式必须成对使用，否则产生 \((N_s-1)T_s\) 的固定距离偏移（本参数下 743.8 m）。

| 写法 | 频响 | 等价卷积核 | 峰值索引 | 截取 |
| --- | --- | --- | --- | --- |
| 相关（本模块采用） | `H = conj(FFT(s_tx_bb))` | 循环反转共轭核，时间原点在 0 | 回波起始 \(r_0\) | `pc[:, 0:Nr]` |
| 卷积 | `H = FFT(conj(flip(s_tx_bb)))` | 线性反转共轭核，原点在 \(N_s-1\) | \(r_0+N_s-1\) | `pc[:, Ns-1:Ns-1+Nr]` |

约束：`Nfft >= Nw` 即可，混叠只会落在被丢弃的区间，不必补到 `Ns+Nw-1`；
`Nr <= Nw - Ns` 保证回波完整落在接收窗内。

# 4 坐标轴

\[
r_j=\frac{c}{2}t_{start}+j\frac{c}{2f_s},\qquad
f_k=\frac{k}{N_d}f_r,\qquad
v_k=-\frac{\lambda}{2}f_k,\qquad
k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1
\]

# 5 处理链

```
raw[Nc][Nw] → 补零到 Nfft → 距离维 FFT → 乘 H → 距离维 IFFT → 截取前 Nr
            → 慢时间加窗(Hann) → 转置成 rd → 慢时间 FFT(Nd) → fftshift → rdmap[Nr][Nd]
```

# 6 计时口径（三版必须一致）

- **不含 IO**：读原始回波、写 RD 矩阵、绘图、写 json 都在计时区外；常驻量（LFM 频谱、
  Hann 窗、FFT plan、pinned 缓冲）只准备一次。把磁盘读写计进去，数字会被文件系统噪声主导。
- **CPU**：只有一个总时间，语义上等于「计算 + 访存」，NumPy 层面两者交织，不拆。
- **GPU**：`copy`（H2D + D2H）+ `compute`（核函数与 cuFFT）= `algo`。
- **统计量**：`REPS = 100`，报**中位数**并记 min/max，json 存全部。单次或平均值不足以下结论。
- **必须热身**：GPU 是 DVFS 的，空闲时显存时钟只有满频的 1/15，短负载会读到 3.5 倍偏慢的数。
  计时前连续跑够 0.3 s，并打印 GPU 型号、驱动与时钟。
- 三版必须同口径、同机器、同一时间窗内测，比率才允许写进 `res`。

# 7 代码约定

- 产物名一律 `{程序名}_{后缀}`，程序名取源文件主干：`PC_v1_*`、`PC_v1_cupy_*`、`PC_v1_cu_*`。
- 每个程序都必须打印：系统参数、派生参数、信噪比增益、峰值自检、与参照的误差、计时、环境块。
- `dtype` 一律 complex64；三个实现的参数逐项一致，改一处必须同时改三处。
- 分工：NumPy 版负责 CPU 逻辑正确性（可读优先）；CuPy 版只负责 GPU 上的初步正确性
  （与 CPU 对齐、跑通、给出量级参考），**不做调优、不出现 RawKernel**；性能调优一律归 CUDA 版。
- 因此 CuPy 版允许保留已知的低效写法：发现的优化点登记在 `res` 里作为 CUDA 版的任务，
  不回头改 CuPy。
- 布局、截取、坐标轴、计时口径只在本文件维护，代码里引用本文件，不复述。

# 8 运行

```
python 04_detect/pulse_compression/src/py/PC_v1.py        # CPU 参照，产出 rxbb / rdmap / timing
python 04_detect/pulse_compression/src/py/PC_v1_cupy.py   # GPU 库基线，读上一行的 rxbb 与 rdmap
python 04_detect/pulse_compression/src/py/PC_v1_verify_gain.py   # 独立复现 res §2 的增益账
# CUDA 版：src/cpp/PC_v1.cu（待写）
```

依赖：numpy、matplotlib、cupy。环境：CUDA 13.2 / driver 616.92 / RTX 3050 Laptop 4 GB（sm_86）。

# 9 产物

| 文件 | 内容 |
| --- | --- |
| `PC_v1_rxbb.npy` | 仿真回波基带 `[Nc][Nw]` complex64，CPU 版产出 |
| `PC_v1_rdmap.npy` | CPU 参照 RD 图 `[Nr][Nd]` complex64 |
| `PC_v1_cupy_rdmap.npy` | CuPy 版 RD 图 |
| `PC_v1{,_cupy}_timing.json` | copy / compute / algo 的中位数与 min/max |
| `PC_v1{,_cupy}_rdmap_{2d,2d_dB,3d,3d_dB}.png` | 四张图 |
