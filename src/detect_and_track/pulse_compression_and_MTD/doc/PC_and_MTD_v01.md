[toc]

# 1 驻定相位原理

## 1.1 内容

考虑如下形式的振荡积分：
\[
I = \int_{-\infty}^{+\infty} a(t) e^{j\varphi(t)} dt
\]

若满足以下条件：

1. \(a(t)\) 为实包络，在驻点附近变化缓慢；
2. \(\varphi(t)\) 为实相位函数；
3. 在积分区间内，\(\varphi(t)\) 存在唯一驻点 \(t_0\)，满足 \(\varphi'(t_0)=0\)；
4. 驻点非退化，即 \(\varphi''(t_0)\neq 0\)。

则当相位 \(\varphi(t)\) 变化足够快时，积分的主要贡献集中在驻点 \(t_0\) 附近，其近似值为：
\[
\boxed{
I \approx a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{|\varphi''(t_0)|}} \exp\left( j\frac{\pi}{4} \operatorname{sgn}(\varphi''(t_0)) \right)
}
\]
其中 \(\operatorname{sgn}(\cdot)\) 为符号函数。

## 1.2 证明

非驻点区域的积分因快速振荡而相互抵消，贡献可忽略，故取 \(\delta > 0\) 作局部化：
\[
I \approx \int_{t_0-\delta}^{t_0+\delta} a(t) e^{j\varphi(t)} dt
\]

在驻点处展开相位，并视包络为常数：
\[
\varphi(t) = \varphi(t_0) + \frac{1}{2}\varphi''(t_0)(t-t_0)^2 + O((t-t_0)^3), \qquad a(t) \approx a(t_0)
\]
忽略高阶项并提取常数项：
\[
I \approx a(t_0) e^{j\varphi(t_0)} \int_{t_0-\delta}^{t_0+\delta} \exp\left[ j\frac{1}{2}\varphi''(t_0)(t-t_0)^2 \right] dt
\]

设 \(\varphi''(t_0) > 0\)。令 \(\tau = t - t_0\)，再令
\[
\frac{1}{2}\varphi''(t_0)\tau^2 = \frac{\pi}{2}y^2
\quad\Rightarrow\quad
\tau = \sqrt{\frac{\pi}{\varphi''(t_0)}} y,
\qquad
Y = \sqrt{\frac{\varphi''(t_0)}{\pi}} \delta
\]
被积函数为偶函数，于是
\[
I \approx 2a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{\pi}{\varphi''(t_0)}} \int_{0}^{Y} e^{j\frac{\pi}{2}y^2} dy
\]

标准菲涅尔积分为
\[
\int_{0}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}}
\]
有限上限的尾部误差由分部积分估计：
\[
\int_{Y}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \int_{Y}^{\infty} \frac{1}{j\pi y} d\left( e^{j\frac{\pi}{2}y^2} \right) = \left[ \frac{e^{j\frac{\pi}{2}y^2}}{j\pi y} \right]_{Y}^{\infty} + \int_{Y}^{\infty} \frac{1}{j\pi y^2} e^{j\frac{\pi}{2}y^2} dy = O\left(\frac{1}{Y}\right)
\]
取 \(\delta\) 足够大使 \(Y\) 足够大，该项可忽略，积分上限可延拓到无穷：
\[
\int_{0}^{Y} e^{j\frac{\pi}{2}y^2} dy \approx \int_{0}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}}
\]

代回原式：
\[
I \approx 2a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{\pi}{\varphi''(t_0)}} \cdot \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}} = a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{\varphi''(t_0)}} e^{j\frac{\pi}{4}}
\]

\(\varphi''(t_0) < 0\) 时同样代换，指数变为 \(-j\pi/4\)。统一引入符号函数，最终单驻点驻定相位公式为：
\[
\boxed{
I \approx a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{|\varphi''(t_0)|}} \exp\left( j\frac{\pi}{4} \operatorname{sgn}(\varphi''(t_0)) \right)
}
\]

# 2 匹配滤波

## 2.1 内容

考虑信号模型
\[
y(t) = x(t) + n(t)
\]
其中 \(x(t)\) 为已知的确定性信号，\(n(t)\) 为加性白噪声，双边功率谱密度为 \(N_0/2\)。接收信号经冲激响应 \(h(t)\)（频响 \(H(f)\)）的 LTI 系统后输出
\[
y_o(t) = y(t) * h(t) = x_o(t) + n_o(t)
\]
设计目标是使判决时刻 \(t_0\) 的输出信噪比最大。最优频响、冲激响应与最大输出信噪比为
\[
\boxed{
H(f) = k X^*(f) e^{-j 2\pi f t_0}
}
\qquad
\boxed{
h(t) = k x^*(t_0 - t)
}
\qquad
\boxed{
\text{SNR}_{\max} = \frac{2E}{N_0}
}
\]
其中 \(k\) 为任意非零常数，\(X(f)\) 为 \(x(t)\) 的频谱，\(E = \int |x(t)|^2 dt = \int |X(f)|^2 df\) 为信号能量。

## 2.2 证明

输出信号在 \(t_0\) 时刻的取值与输出噪声平均功率为
\[
x_o(t_0) = \int_{-\infty}^{+\infty} H(f) X(f) e^{j 2\pi f t_0} df,
\qquad
P_{n_o} = \int_{-\infty}^{+\infty} |H(f)|^2 \frac{N_0}{2} df
\]
故输出信噪比为
\[
\text{SNR} = \frac{|x_o(t_0)|^2}{P_{n_o}} = \frac{\left| \int H(f) X(f) e^{j 2\pi f t_0} df \right|^2}{\frac{N_0}{2} \int |H(f)|^2 df}
\]

对分子应用柯西-施瓦茨不等式，并利用 \(|e^{j 2\pi f t_0}| = 1\)：
\[
\left| \int H(f) X(f) e^{j 2\pi f t_0} df \right|^2
\le \left( \int |H(f)|^2 df \right) \left( \int |X(f)|^2 df \right)
\]
代入信噪比表达式，再由帕塞瓦尔定理 \(E = \int |X(f)|^2 df\) 得
\[
\text{SNR} \le \frac{\int |X(f)|^2 df}{N_0 / 2} = \frac{2E}{N_0}
\]

取等条件为两函数成共轭比例
\[
H(f) = k \left( X(f) e^{j 2\pi f t_0} \right)^* = k X^*(f) e^{-j 2\pi f t_0}
\]
此时 \(\text{SNR} = \text{SNR}_{\max} = 2E/N_0\)。对 \(H(f)\) 取逆变换得时域冲激响应
\[
h(t) = \int_{-\infty}^{+\infty} H(f) e^{j 2\pi f t} df = k \left( \int_{-\infty}^{+\infty} X(f) e^{j 2\pi f (t_0 - t)} df \right)^* = k x^*(t_0 - t)
\]

# 3 脉冲压缩

矩形窗定义为前闭后开形式：
\[
\operatorname{rect}(t)=
\begin{cases}
1, & -\frac12 \le t < \frac12,\\
0, & \text{其他}.
\end{cases}
\]

发射带载波的射频复信号为
\[
s_{TX}(t)=e^{j2\pi f_c t}s_{TX\_BB}(t),
\]
其中发射复基带信号 \(s_{TX\_BB}\) 的中心在 \(T_p/2\)：
\[
s_{TX\_BB}(t)=\operatorname{rect}\left(\frac{t-T_p/2}{T_p}\right)e^{j\pi K(t-T_p/2)^2}.
\]
记中心在 \(0\) 的 LFM 为
\[
s_0(t)=\operatorname{rect}\left(\frac{t}{T_p}\right)e^{j\pi K t^2},
\]
则 \(s_{TX\_BB}\) 是 \(s_0\) 的时移：
\[
s_{TX\_BB}(t)=s_0(t-T_p/2).
\]
其瞬时频率 \(f_i(t)=K(t-T_p/2)\)，在 \(0\le t<T_p\) 内由 \(-KT_p/2\) 变到 \(KT_p/2\)，带宽
\[
B=|K|T_p.
\]

接收去载频后，基带回波是 \(s_0\) 平移 \(T_p/2+\tau\)，并乘上延时载波相位：
\[
s_{RX\_BB}(t)=e^{-j2\pi f_c\tau}s_{TX\_BB}(t-\tau)
=e^{-j2\pi f_c\tau}s_0(t-T_p/2-\tau),
\qquad
\tau=\frac{2R}{c}.
\]
其中 \(e^{-j2\pi f_c\tau}\) 为延时载波相位，是后续测速的信息来源；\(s_0(t-T_p/2-\tau)\) 为延时 LFM，是脉冲压缩的对象。

先求 \(s_0(t)\) 的频谱 \(S_0(f)\)，其相位为
\[
\varphi(t)=\pi K t^2-2\pi f t,
\qquad
\varphi'(t)=0 \Rightarrow t_s=\frac{f}{K},
\qquad
\varphi''(t_s)=2\pi K.
\]
当 \(|f|\le B/2\) 时 \(|t_s|\le T_p/2\)，驻点落在矩形窗内，由驻定相位原理得
\[
S_0(f)\approx
\frac{1}{\sqrt{|K|}}
\operatorname{rect}\left(\frac{f}{B}\right)
e^{-j\pi f^2/K}
e^{j\frac{\pi}{4}\operatorname{sgn}(K)}.
\]

其余信号的频谱均写成 \(S_0(f)\) 的形式。由时移性质
\[
S_{TX\_BB}(f)=e^{-j2\pi f T_p/2}S_0(f),
\]
回波再叠加时延 \(\tau\) 与去载频相位：
\[
S_{RX\_BB}(f)=e^{-j2\pi f_c\tau}e^{-j2\pi f(T_p/2+\tau)}S_0(f).
\]

匹配滤波器取
\[
H(f)=S_{TX\_BB}^*(f),
\]
判决时刻取 \(t_0=0\)，否则输出时延不再对应真实回波时延 \(\tau\)。匹配滤波输出频谱为
\[
Y(f)=S_{RX\_BB}(f)H(f)
=e^{-j2\pi f_c\tau}\,e^{-j2\pi f(T_p/2+\tau)}\,e^{j2\pi fT_p/2}\,|S_0(f)|^2
=e^{-j2\pi f_c\tau}|S_0(f)|^2e^{-j2\pi f\tau},
\]
两处 \(T_p/2\) 的相位相消，输出只保留载波相位与真实时延 \(\tau\)。

由频域近似矩形窗
\[
|S_0(f)|^2\approx
\frac{1}{|K|}
\operatorname{rect}\left(\frac{f}{B}\right),
\]
逆变换得
\[
y(t)=e^{-j2\pi f_c\tau}
\frac{1}{|K|}
\int_{-B/2}^{B/2}
e^{j2\pi f(t-\tau)}df
=e^{-j2\pi f_c\tau}
\frac{B}{|K|}
\operatorname{sinc}\bigl(B(t-\tau)\bigr)
=T_p e^{-j2\pi f_c\tau}
\operatorname{sinc}\bigl(B(t-\tau)\bigr),
\]
其中
\[
\operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x},
\]
并用到 \(B/|K|=T_p\)。

这里频率从 \(-B/2\) 到 \(B/2\) 对称变化，积分后不会出现额外的 \(e^{j\pi B(t-\tau)}\) 线性相位。若频率从 \(0\) 到 \(B\) 变化，则积分中心在 \(B/2\)，会引入 \(e^{j\pi B(t-\tau)}\) 因子，相当于附加时移，导致时延不准。因此必须采用中心化 LFM，即频率在 \(-B/2\) 到 \(B/2\) 内变化。

时域近似为 sinc 函数，主瓣宽度由第一零点间距决定：
\[
\Delta t=\frac{1}{B},
\qquad
\Delta R=\frac{c\Delta t}{2}=\frac{c}{2B}.
\]

设白噪声双边功率谱密度为 \(N_0/2\)，发射基带信号能量为
\[
E=\int |s_{TX\_BB}(t)|^2dt=T_p.
\]
匹配滤波输出峰值信噪比为
\[
SNR_{out}=\frac{2E}{N_0}=\frac{2T_p}{N_0}.
\]
输入信号平均功率为 \(P_s=E/T_p=1\)，输入噪声在信号带宽 \(B\) 内的功率为 \(P_n=(N_0/2)B\)，故输入信噪比为
\[
SNR_{in}=\frac{P_s}{P_n}=\frac{2}{N_0 B},
\]
于是脉冲压缩信噪比增益为
\[
G_p=\frac{SNR_{out}}{SNR_{in}}=BT_p,
\]
即时间带宽积，也即脉冲压缩比。

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

的单频信号，这正是模拟多普勒频率。因此对慢时间维做 FFT，峰值位置直接对应 \(f_d\)，测出频率也就测出了速度。

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

# 5 动目标显示（MTI）

杂波（地物、云雨等）功率远强于目标，且多普勒频率集中在零频附近；若直接做 MTD，强杂波的旁瓣会淹没慢速小目标。因此在 MTD 之前需要滤除零频附近的杂波，这就是 MTI。MTI 本质上是零频凹口滤波器。

单延迟线对消器的时域形式与 Z 变换为
\[
y(m)=x(m)-x(m-1),
\qquad
H(z)=1-z^{-1}.
\]
令 \(z=e^{j\omega}\)，\(\omega\) 为数字角频率，则频响与幅度响应为
\[
H(e^{j\omega})=1-e^{-j\omega}
=2j e^{-j\omega/2}\sin\left(\frac{\omega}{2}\right),
\qquad
|H(e^{j\omega})|=2\left|\sin\left(\frac{\omega}{2}\right)\right|.
\]
在 \(\omega=0\) 处为零，形成零频凹口。

双延迟线对消器的时域形式与 Z 变换为
\[
y(m)=x(m)-2x(m-1)+x(m-2),
\qquad
H(z)=(1-z^{-1})^2,
\]
频响幅度为
\[
|H(e^{j\omega})|=4\sin^2\left(\frac{\omega}{2}\right).
\]
双对消器的凹口更宽、更深，对消效果更好，代价是主瓣增益损失更大、对慢速目标的抑制更强。杂波带宽窄、目标速度较高时单对消即可；杂波频谱宽或对消深度要求高时采用双对消或更高阶对消器。

脉冲压缩之后，MTI 与 MTD 的先后顺序有两种常见方案：
\[
\text{脉冲压缩} \rightarrow \text{时域对消} \rightarrow \text{加窗} \rightarrow \text{MTD},
\]
\[
\text{脉冲压缩} \rightarrow \text{加窗} \rightarrow \text{MTD} \rightarrow \text{频域对消}.
\]
两种方案在数学上都是合理的，因为时域差分与频域相乘本质上等价。工程上一般选择第一种：先对消可以在进入 FFT 之前抑制强杂波，避免其旁瓣泄漏污染整个多普勒谱；若先做 FFT，杂波能量已通过旁瓣扩散到所有多普勒通道，此时再乘零频凹口只能抑制零频附近主瓣区域的杂波，已泄漏到其他通道的旁瓣无法消除，弱目标仍可能被淹没。

需要注意，频域对消所乘的应当是复数响应 \(H(e^{j\omega})=1-e^{-j\omega}\) 本身，而不是它的幅度 \(|H(e^{j\omega})|=2\left|\sin\left(\frac{\omega}{2}\right)\right|\) 或 \(4\sin^2\left(\frac{\omega}{2}\right)\) 这类实凹口：只乘幅度相当于丢掉相位，把相干处理退化为非相干加权，与 \(H(z)=1-z^{-1}\) 并不等价。

# 6 细节

按 C/C++、Python 的内存布局，遵守行主序原则。

原始数据矩阵为 \(\text{raw}[N_c][N_w]\)，每一行是一个脉冲的接收窗采样，列是距离维快时间。发射 LFM 为 \(s_{TX\_BB}[N_s]\)，\(N_s=\lfloor T_p f_s\rfloor\)。接收窗从 \(t_{\text{start}}\) 到 \(t_{\text{end}}\)，满足 \(t_{\text{start}}\ge T_p\)、\(t_{\text{end}}\le T_r\)，窗内点数 \(N_w=\lfloor (t_{\text{end}}-t_{\text{start}})f_s\rfloor\)。最终期望得到多普勒-距离图 \(\text{drmap}[N_d][N_r]\)，\(d\) 在前、\(r\) 在后，\(N_r\) 为保留的距离点数。

距离维匹配滤波在频域完成。取 FFT 长度
\[
N_{\text{fft}}\ge N_w,
\]
通常取下一个 2 的幂。对发射 LFM 补零后做 FFT 得到频响，这里有两条等价写法，卷积核与截取起点必须成对使用。

写法一（相关形式，本文档与代码采用）：
\[
H=\operatorname{conj}\left(\operatorname{FFT}\left(s_{TX\_BB}\right)\right).
\]
它等价于与循环反转共轭核 \(s^*_{TX\_BB}[(-n)\bmod N_{\text{fft}}]\) 做 \(N_{\text{fft}}\) 点循环卷积，核的时间原点落在索引 \(0\)，输出峰值索引即回波起始索引 \(r_0\)，因此截取
\[
\text{pc}=\text{pc\_full}[:,0:N_r].
\]

写法二（卷积形式）：
\[
H=\operatorname{FFT}\left(\operatorname{conj}\left(\operatorname{flip}\left(s_{TX\_BB}\right)\right)\right).
\]
它等价于与线性反转共轭核 \(s^*_{TX\_BB}[N_s-1-n]\) 做卷积，核的时间原点落在索引 \(N_s-1\)，峰值索引为 \(r_0+N_s-1\)，因此截取
\[
\text{pc}=\text{pc\_full}[:,N_s-1:N_s-1+N_r].
\]

两种写法输出的同一段逐点相同，只差 \(N_s-1\) 的索引平移；配套错用会产生 \((N_s-1)T_s\) 的固定距离偏移，本文参数下是 \(119\times6.25=743.8\) m。取 \(N_{\text{fft}}\ge N_w\) 即可：混叠只会落在写法二中被丢弃的前 \(N_s-1\) 个索引上，或落在写法一中回波不完整的那一段（\(j>N_w-N_s\)），都不影响保留区，因此不必补零到 \(N_s+N_w-1\)。

得到脉冲压缩矩阵 \(\text{pc}[N_c][N_r]\)，其中 \(N_r\le N_w-N_s\)（可测区上限，保证回波完整落在窗内）。距离轴为
\[
r_j=\frac{c}{2}t_{\text{start}}+j\Delta r,
\qquad
\Delta r=\frac{c}{2f_s},
\qquad
j=0,1,\dots,N_r-1,
\]
真实物理分辨率为
\[
\delta r=\frac{c}{2B}.
\]
过采样时 \(\Delta r<\delta r\)，一个分辨单元对应多个距离单元格。

若需 MTI，对 \(\text{pc}\) 沿慢时间维做对消，记对消结果为 \(\text{mti}\)。单对消为
\[
\text{mti}[m,:]=\text{pc}[m,:]-\text{pc}[m-1,:],
\qquad m=1,\dots,N_c-1,
\]
双对消为
\[
\text{mti}[m,:]=\text{pc}[m,:]-2\text{pc}[m-1,:]+\text{pc}[m-2,:],
\qquad m=2,\dots,N_c-1,
\]
边界行可置零或丢弃，通常直接保留有效行：单对消输出 \(N_c-1\) 行，双对消输出 \(N_c-2\) 行。对消后沿慢时间维乘窗函数 \(w[m]\)，如 Hamming、Hann 等。

慢时间维 MTD 对加窗后的矩阵沿行方向做 FFT，补零到 \(N_d\) 点：
\[
\text{drmap}[N_d][N_r]=\operatorname{fftshift}\left(\operatorname{FFT}_{N_d}\{\text{pc}_{\text{win}}\},\ \text{axis}=0\right).
\]
频率轴与速度轴为
\[
f_k=\frac{k}{N_d}f_r,
\qquad
v_k=-\frac{\lambda}{2}f_k=-\frac{\lambda}{2}\frac{k}{N_d}f_r,
\qquad
k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1.
\]
速度单元格大小与真实速度分辨率为
\[
\Delta v=\frac{\lambda}{2N_d T_r},
\qquad
\delta v=\frac{\lambda}{2N_c T_r}.
\]
若 \(N_c=N_d=2^n\)，则 \(\Delta v=\delta v\)；若 \(N_d>N_c\)，则 \(\Delta v<\delta v\)，一个分辨单元对应多个速度单元格。\(\Delta v\) 是格子大小，\(\delta v\) 是物理分辨率，二者必须区分。

伪代码如下。

```
# 输入：raw[Nc][Nw], s_tx_bb[Ns], 参数 Tp, fs, B, K, Nc, Nd, Nr, fr, lambda, c
Ns = floor(Tp * fs)
Nw = floor((t_end - t_start) * fs)
N_fft = nextpow2(Nw)          # 只要 >= Nw；写法二才需要 >= Ns+Nw-1

# 距离维匹配滤波（写法一：conj(FFT)，从 0 截取）
H = conj(fft(s_tx_bb, N_fft))
pc = zeros(Nc, Nr)
for m in range(Nc):
    Raw = fft(raw[m], N_fft)
    conv_full = ifft(Raw * H)
    pc[m] = conv_full[0 : Nr]

# 可选 MTI
if use_MTI:
    if mode == 'single':
        mti = zeros(Nc, Nr)
        for m in range(1, Nc):
            mti[m, :] = pc[m, :] - pc[m-1, :]
        # mti[0, :] 置零或丢弃
        pc_use = mti
    elif mode == 'double':
        mti = zeros(Nc, Nr)
        for m in range(2, Nc):
            mti[m, :] = pc[m, :] - 2*pc[m-1, :] + pc[m-2, :]
        # mti[0, :], mti[1, :] 置零或丢弃
        pc_use = mti
else:
    pc_use = pc

# 加窗
w = window(pc_use.shape[0])   # 如 Hamming
pc_win = pc_use * w[:, None]

# MTD
drmap = fftshift(fft(pc_win, n=Nd, axis=0), axes=0)   # [Nd][Nr]

# 坐标轴
r_axis = c/2 * t_start + arange(Nr) * c/(2*fs)
f_axis = arange(-Nd/2, Nd/2) / Nd * fr
v_axis = -lambda/2 * f_axis

# 绘图
plot_2d(abs(drmap), r_axis, v_axis)
plot_3d(abs(drmap), r_axis, v_axis)
```

三个版本中，Python CPU 版用 NumPy 批量 FFT，Python CuPy 版将数组迁至 GPU 用 CuPy 对应接口，CUDA C/C++ 版手写核函数做距离维频域匹配滤波与慢时间维 FFT，并用 Nsight 分析访存与占用率。原始数据 \(\text{raw}[N_c][N_w]\) 是输入，最终输出 \(\text{drmap}[N_d][N_r]\) 即多普勒-距离图。

# 7 测距测速的二维信号流程

模拟域推导，从回波基带 \(s_{RX\_BB}\) 开始。快时间 \(\tau\) 是相对本脉冲发射时刻的本地时间，慢时间 \(\eta=mT_r\)，\(m=0,\dots,N_c-1\)。只保留与测距测速有关的项，幅度与常数相位统一并入 \(A\)；发射基带 \(s_{TX\_BB}(\tau)=\operatorname{rect}\left(\frac{\tau-T_p/2}{T_p}\right)e^{j\pi K(\tau-T_p/2)^2}\) 见第 3 节。

每一步的"域"记作（快时间所处的域，慢时间所处的域），整条链路是：

| 步骤 | 域 | 信号 | 操作 |
| --- | --- | --- | --- |
| 1 | \((\tau,\eta)\) | \(s_{RX\_BB}(\tau,\eta)\) | 回波基带按慢时间排成二维 |
| 2 | \((f_\tau,\eta)\) | \(s_{RX\_BB}(f_\tau,\eta)\) | 沿 \(\tau\) 做 FFT |
| 3 | \((f_\tau,\eta)\) | \(H(f_\tau)s_{RX\_BB}(f_\tau,\eta)\) | 乘匹配滤波响应，抛物线相位被抵消 |
| 4 | \((\tau,\eta)\) | \(s_{pc}(\tau,\eta)\) | 沿 \(f_\tau\) 做 IFFT，快时间压成 sinc |
| 5 | \((\tau,f_\eta)\) | \(s_{pc}(\tau,f_\eta)\) | 沿 \(\eta\) 做 FFT，得到 RD 图 |

下面逐段写出。

## 7.1 \((\tau,\eta)\)：回波基带

目标距离 \(R(\eta)=R_0+v\eta\)，时延
\[
\tau(\eta)=\frac{2R(\eta)}{c}=\frac{2R_0}{c}+\frac{2v\eta}{c}.
\]
单脉冲回波基带（第 3 节）沿慢时间排开：
\[
s_{RX\_BB}(\tau,\eta)=e^{-j2\pi f_c\tau(\eta)}\operatorname{rect}\left(\frac{\tau-\tau(\eta)}{T_p}\right)e^{j\pi K(\tau-\tau(\eta))^2},
\]
其中
\[
e^{-j2\pi f_c\tau(\eta)}=e^{-j2\pi f_c\frac{2R_0}{c}}e^{j2\pi f_d\eta},
\qquad
f_d=-\frac{2v}{\lambda}.
\]
慢时间维只有 \(e^{j2\pi f_d\eta}\) 一个相位在变，测速信息全在这里；快时间维是 \(s_{TX\_BB}\) 被 \(\tau(\eta)\) 平移，\(\tau(\eta)\) 随 \(\eta\) 变化的那部分就是距离走动。

## 7.2 \((f_\tau,\eta)\)：快时间频域

沿 \(\tau\) 做傅里叶变换，用 3.2 节的 \(S_{TX\_BB}(f_\tau)\) 与时移性质：
\[
s_{RX\_BB}(f_\tau,\eta)=e^{-j2\pi f_c\tau(\eta)}\,S_{TX\_BB}(f_\tau)\,e^{-j2\pi f_\tau\tau(\eta)}.
\]
带内 \(S_{TX\_BB}(f_\tau)\) 幅度恒定、相位为抛物线（含时移因子 \(e^{-j2\pi f_\tau T_p/2}\)）；线性相位 \(-2\pi f_\tau\tau(\eta)\) 携带 \(\tau(\eta)\) 的全部信息。

## 7.3 \((f_\tau,\eta)\)：匹配滤波后

乘 \(H(f_\tau)=S_{TX\_BB}^*(f_\tau)\)，抛物线相位被共轭抵消，频域只剩常数幅度与线性相位：
\[
H(f_\tau)s_{RX\_BB}(f_\tau,\eta)=e^{-j2\pi f_c\tau(\eta)}\,|S_{TX\_BB}(f_\tau)|^2\,e^{-j2\pi f_\tau\tau(\eta)}.
\]

## 7.4 \((\tau,\eta)\)：快时间逆变换（脉压输出）

沿 \(f_\tau\) 做 IFFT。带内 \(|S_{TX\_BB}(f_\tau)|^2\approx1/|K|\)，于是
\[
s_{pc}(\tau,\eta)=\mathcal{F}_\tau^{-1}\bigl\{H(f_\tau)s_{RX\_BB}(f_\tau,\eta)\bigr\}
=T_p\,e^{-j2\pi f_c\tau(\eta)}\operatorname{sinc}\bigl(B(\tau-\tau(\eta))\bigr).
\]
快时间压成 sinc（第一零点间距 \(1/B\)，即 \(\Delta R=c/2B\)），慢时间相位原样保留。忽略距离走动时 \(\tau(\eta)\approx 2R_0/c\)，峰固定在 \(\tau=2R_0/c\)：
\[
s_{pc}(\tau,\eta)\approx T_p\,e^{-j2\pi f_c\frac{2R_0}{c}}e^{j2\pi f_d\eta}\operatorname{sinc}\left(B\left(\tau-\frac{2R_0}{c}\right)\right).
\]
走动量超过半个距离单元格时需校正（Keystone 变换或插值），不在此展开。

## 7.5 \((\tau,f_\eta)\)：慢时间频域（RD 图）

沿 \(\eta\) 做傅里叶变换：
\[
s_{pc}(\tau,f_\eta)=\int s_{pc}(\tau,\eta)e^{-j2\pi f_\eta\eta}d\eta
=T_p\,e^{-j2\pi f_c\frac{2R_0}{c}}\operatorname{sinc}\left(B\left(\tau-\frac{2R_0}{c}\right)\right)W(f_\eta-f_d).
\]
\(W\) 由有限相干积累时长 \(N_cT_r\) 决定（矩形截断时为 sinc，主瓣宽 \(1/(N_cT_r)\)，对应 \(\Delta v=\lambda/(2N_cT_r)\)）。距离在 \(\tau\)、速度在 \(f_\eta\)，这就是 RD 图。

## 7.6 二维变换视角

记 \(s_{RX\_BB}(f_\tau,f_\eta)=\mathcal{F}_\tau\mathcal{F}_\eta\{s_{RX\_BB}(\tau,\eta)\}\)（两个维度的变换次序可交换）。步骤 3 的乘法只在 \(f_\tau\) 维、步骤 5 的变换只沿 \(\eta\)，两者作用在不同维度、可以交换，于是
\[
s_{pc}(\tau,f_\eta)=\mathcal{F}_\tau^{-1}\bigl\{H(f_\tau)\,s_{RX\_BB}(f_\tau,f_\eta)\bigr\}.
\]
即"一次二维正变换 + 沿 \(f_\tau\) 乘 \(H\) + 一次沿快时间的逆变换"。最后一步不能用二维逆变换：慢时间必须停在频域（它本身就是多普勒），只有快时间要回到时域，所以二维 FFT 能替掉的是步骤 2 与步骤 5 那两次正变换，替不掉步骤 4 的逆变换。

离散化、补零、截取与坐标轴见第 6 节。
