# 4 动目标检测（MTD）

脉冲压缩后的回波近似为 \(T_p e^{-j2\pi f_c\tau}\operatorname{sinc}\bigl(B(t-\tau)\bigr)\)。不考虑高速目标的距离走动时，它在时延 \(\tau\) 处压成一条亮线，核心信息集中在相位项 \(e^{-j2\pi f_c\tau}\) 上。

设目标初距 \(R_0\)、径向速度 \(v\)，脉冲重复间隔 \(T_r\)（即 \(T_r=PRT=1/f_r=1/PRF\)，\(f_r\) 为脉冲重复频率）。第 \(m\) 个脉冲对应 \(t_m=mT_r\)，\(m=0,1,\dots,N_c-1\)，\(N_c\) 为实际发射的 chirp 数，则
\[
R_m=R_0+vmT_r,
\qquad
\tau_m=\frac{2R_m}{c}=\frac{2R_0}{c}+\frac{2vmT_r}{c}.
\]

代入相位项：
\[
    e^{-j2\pi f_c\tau_m}
    =
    e^{-j2\pi f_c\frac{2R_0}{c}}
    e^{-j2\pi \frac{2v}{\lambda}mT_r},
    \qquad
    \lambda=\frac{c}{f_c}.
\]

第一项与 \(m\) 无关。沿慢时间维看，回波是频率为
\[
f_d=-\frac{2v}{\lambda}
\]

的单频信号，这正是模拟多普勒频率。因此对慢时间维做傅里叶变换，峰值位置直接对应 \(f_d\)，测出频率也就测出了速度。

慢时间维采样间隔为 \(T_r\)，采样率为 \(1/T_r=f_r\)。由采样定理，无模糊多普勒范围为 \([-f_r/2,f_r/2]\)，对应的无模糊测速范围为
\[
\left[-\frac{\lambda f_r}{4},\frac{\lambda f_r}{4}\right],
\]

最大不模糊速度为
\[
v_{\max}=\frac{\lambda f_r}{4}.
\]

相干积累时间 \(T_{CPI}=N_c T_r\)，故多普勒频率分辨率与测速分辨率为
\[
\Delta f_d=\frac{1}{T_{CPI}}=\frac{f_r}{N_c},
\qquad
\Delta v=\frac{\lambda}{2}\Delta f_d=\frac{\lambda f_r}{2N_c}.
\]

需要严格区分模拟多普勒频率与数字多普勒频率：前者 \(f_d=-2v/\lambda\) 是连续的物理量，后者是 \(f_d\) 以 \(f_r\) 归一化的结果，即 \(f'_d=f_d/f_r\in[-1/2,1/2)\)。FFT 输出索引 \(k=0,1,\dots,N_d-1\)，\(N_d\) 为 FFT 点数，`fftshift` 后 \(k=-N_d/2,\dots,N_d/2-1\)，对应数字频率 \(k/N_d\)，因此模拟频率轴与速度轴为
\[
f_k = \frac{k}{N_d} f_r,
\qquad
v_k = -\frac{\lambda}{2} f_k = -\frac{\lambda}{2} \frac{k}{N_d} f_r,
\qquad
k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1.
\]

工程上通常取 \(N_c=N_d=2^n\)。\(N_c\) 是实际发射的 chirp 数，决定物理相干积累时间与真实分辨率 \(\Delta v=\lambda f_r/(2N_c)\)；\(N_d\) 是 FFT 点数，\(N_d=N_c\) 时两者一致，\(N_d>N_c\)（补零）时频谱被插值、读数更密，但物理分辨率不变，真实分辨率始终由 \(N_c\) 决定。

由于只有 \(N_c\) 个脉冲，相当于对慢时间维加了矩形窗截断，直接 FFT 会产生 sinc 泄露，第一旁瓣仅比主瓣低约 \(-13.2\) dB，强目标的旁瓣可能掩盖弱目标，因此通常需要加窗。矩形窗主瓣最窄、旁瓣最高；Hamming 窗旁瓣约 \(-42\) dB，Hann 窗更低，Blackman 窗极低但主瓣最宽。加窗的代价有两条：主瓣展宽使可分辨间隔变大（Hamming 窗的 3 dB 主瓣宽约为矩形窗的 \(1.3\) 倍），以及等效噪声带宽大于 \(1\) 带来信噪比损失（Hamming 约 \(1.34\) dB，Hann 约 \(1.76\) dB，Blackman 约 \(2.38\) dB，矩形窗为 \(0\)）。窗的选择是主瓣分辨率与旁瓣抑制的折衷，需分辨邻近强/弱目标时通常取 Hamming 或 Taylor 窗。

FFT 之后需要 shift（如 `fftshift`）：常规 FFT 输出的 \(0\) 到 \(f_r\) 对应频率，零频在数组首位、正负多普勒分居两端，读数和判向都不直观；shift 后零频移到中心，频率轴变成 \([-f_r/2,f_r/2)\)，即 \((-N_d/2, N_d/2-1)/N_d \times f_r\)，与物理意义一致，便于判向与读数。