# PC_v1 数学原理

脉压 + MTD 的原理推导，三个实现（NumPy / CuPy / CUDA）共享。
代码约定见 `PC_v1_eng.md`，实测与决策记录见 `PC_v1_res.md`。

# 1 驻定相位原理

振荡积分 \(I=\int a(t)e^{j\varphi(t)}dt\)。若 \(a\) 缓变、\(\varphi\) 在区间内有唯一非退化驻点
\(\varphi'(t_0)=0,\ \varphi''(t_0)\neq 0\)，则主要贡献集中在驻点附近：

\[
I\approx a(t_0)e^{j\varphi(t_0)}\sqrt{\frac{2\pi}{|\varphi''(t_0)|}}
\exp\left(j\frac{\pi}{4}\operatorname{sgn}\varphi''(t_0)\right)
\]

推导要点：非驻点区快速振荡相消；驻点附近把 \(\varphi\) 展开到二次项、\(a\) 视为常数，剩下
Fresnel 积分 \(\int_0^\infty e^{j\pi y^2/2}dy=e^{j\pi/4}/\sqrt{2}\)，有限上限的尾部为 \(O(1/Y)\)。

# 2 匹配滤波

\(y=x+n\)，\(n\) 为白噪声（双边功率谱密度 \(N_0/2\)）。输出信噪比最大的频响与冲激响应：

\[
H(f)=kX^*(f)e^{-j2\pi ft_0},\qquad h(t)=kx^*(t_0-t),\qquad
\mathrm{SNR}_{\max}=\frac{2E}{N_0}
\]

证明用柯西–施瓦茨不等式加帕塞瓦尔定理，等号条件是两个函数共轭成比例。

# 3 脉冲压缩

发射复基带（中心在 \(T_p/2\)）：

\[
s_{TX\_BB}(t)=\operatorname{rect}\left(\frac{t-T_p/2}{T_p}\right)e^{j\pi K(t-T_p/2)^2},
\qquad B=|K|T_p
\]

去载频后的回波基带 \(s_{RX\_BB}(t)=e^{-j2\pi f_c\tau}s_{TX\_BB}(t-\tau)\)，\(\tau=2R/c\)。
取匹配滤波器 \(H(f)=S^*_{TX\_BB}(f)\)、判决时刻 \(t_0=0\)，并用中心化 LFM 的频谱

\[
S_0(f)\approx |K|^{-1/2}\operatorname{rect}\left(\frac{f}{B}\right)
e^{-j\pi f^2/K}e^{j(\pi/4)\operatorname{sgn}K}
\]

得到脉压输出

\[
y(t)=T_p\,e^{-j2\pi f_c\tau}\operatorname{sinc}\bigl(B(t-\tau)\bigr),
\qquad \operatorname{sinc}(x)=\frac{\sin \pi x}{\pi x}
\]

距离分辨率 \(\delta r=c/(2B)\)，脉压增益 \(G_p=BT_p\)，即时间带宽积。

注：频率从 \(0\) 扫到 \(B\) 的输出会多一个 \(e^{j\pi B(t-\tau)}\)，模为 1，只改相位、不移动峰值；
中心化 LFM 让这个因子消失，读数和物理量一一对应，所以采用中心化。

# 4 加窗与信噪比增益

慢时间维加窗 \(w[m]\)：信号相干累加、噪声非相干累加，两者走不同的求和，于是

\[
G_{win}=\frac{\left(\sum w\right)^2}{\sum w^2}
=N_c\left(\frac{\operatorname{mean}w}{\operatorname{rms}w}\right)^2,
\qquad
\mathrm{ENBW}=\frac{N_c\sum w^2}{\left(\sum w\right)^2}
\]

周期 Hann 窗的 \(\operatorname{mean}w=0.5\)、\(\operatorname{rms}w=0.612\)，处理增益 \(2/3=-1.76\) dB。
相干增益 \(\sum w/N_c=0.5\) 只是幅度缩放（峰值掉 6.02 dB，噪声底同时掉 4.26 dB），不进信噪比账。
窗的代价与旁瓣抑制是折衷：矩形窗第一旁瓣 \(-13.2\) dB，Hamming \(-42\) dB，代价是主瓣展宽与
信噪比损失（Hamming 1.34 dB、Hann 1.76 dB、Blackman 2.38 dB）。

与脉压合并成一个式子（脉压是 \(|h|\equiv 1\) 的退化情形，\((\sum|h|)^2/\sum|h|^2=N_s\)）：

\[
G=N_s\,G_{win}=N_sN_c\left(\frac{\operatorname{mean}w}{\operatorname{rms}w}\right)^2
\]

落格损失：目标不落在格子上时峰值还要乘 \(|W(\delta)|/\sum w\)，半格处矩形窗 \(-3.92\) dB、Hann \(-1.42\) dB。

# 5 MTD

第 \(m\) 个脉冲的时延 \(\tau_m=2(R_0+vmT_r)/c\)，代入载波相位：

\[
e^{-j2\pi f_c\tau_m}=e^{-j2\pi f_c\cdot 2R_0/c}e^{j2\pi f_d mT_r},
\qquad f_d=-\frac{2v}{\lambda}
\]

慢时间维是单频信号，FFT 峰值位置直接对应 \(f_d\)，测频即测速。无模糊速度
\(v_{\max}=\lambda f_r/4\)；真实速度分辨率 \(\delta v=\lambda f_r/(2N_c)\)，速度格子
\(\Delta v=\lambda f_r/(2N_d)\)。频率轴与速度轴

\[
f_k=\frac{k}{N_d}f_r,\qquad v_k=-\frac{\lambda}{2}f_k,\qquad
k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1
\]

**分辨率与格子必须分开说**：距离真实分辨率 \(\delta r=c/(2B)\) 由带宽决定，格子
\(\Delta r=c/(2f_s)\) 由采样率决定；速度真实分辨率由实际发射的 chirp 数 \(N_c\) 决定，格子由
FFT 点数 \(N_d\) 决定。补零只把格子加密，不改变分辨率。

# 6 MTI

单延迟线对消器 \(H(z)=1-z^{-1}\)，\(|H(e^{j\omega})|=2|\sin(\omega/2)|\)；双对消器
\(H(z)=(1-z^{-1})^2\)，\(|H|=4\sin^2(\omega/2)\)。凹口落在零频，用来抵掉地物杂波。

顺序取「脉压 → 时域对消 → 加窗 → MTD」：先对消可以避免强杂波旁瓣泄漏污染整个多普勒谱。
频域对消必须乘复响应 \(1-e^{-j\omega}\) 本身；只乘幅度等于丢掉相位，相干处理退化为非相干加权。

# 7 二维信号流程

| 步骤 | 域 | 操作 |
| --- | --- | --- |
| 1 | \((\tau,\eta)\) | 回波按慢时间排成二维 |
| 2 | \((f_\tau,\eta)\) | 沿 \(\tau\) 做 FFT |
| 3 | \((f_\tau,\eta)\) | 乘 \(H(f_\tau)\)，抛物线相位被抵消 |
| 4 | \((\tau,\eta)\) | 沿 \(f_\tau\) 做 IFFT，快时间压成 sinc |
| 5 | \((\tau,f_\eta)\) | 沿 \(\eta\) 做 FFT，得到距离–多普勒图 |

忽略距离走动时

\[
s_{pc}(\tau,\eta)\approx T_p\,e^{-j2\pi f_c\cdot 2R_0/c}e^{j2\pi f_d\eta}
\operatorname{sinc}\Bigl(B\Bigl(\tau-\frac{2R_0}{c}\Bigr)\Bigr)
\]

第 2、5 步可合并为一次二维正变换；第 4 步的逆变换不能并进去——慢时间必须停在频域，它本身就是多普勒。
