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

由于相位 \(\varphi(t)\) 的快速变化，非驻点区域的积分因快速振荡而相互抵消，贡献可忽略。因此，积分的主要贡献集中在驻点 \(t_0\) 附近的邻域内。取 \(\delta > 0\)，原积分可近似为有限区间上的积分：
\[
I \approx \int_{t_0-\delta}^{t_0+\delta} a(t) e^{j\varphi(t)} dt
\]
此时，积分上下限从 \((-\infty, +\infty)\) 缩减为有限的 \([t_0-\delta, t_0+\delta]\)。

在驻点 \(t_0\) 处对相位进行泰勒展开。由于 \(\varphi'(t_0)=0\)，展开式中无一次项，同时包络 \(a(t)\) 在驻点附近近似为常数：
\[
\varphi(t) = \varphi(t_0) + \frac{1}{2}\varphi''(t_0)(t-t_0)^2 + O((t-t_0)^3), \qquad a(t) \approx a(t_0)
\]
忽略高阶项，代入局部化后的积分并提取常数项：
\[
I \approx a(t_0) e^{j\varphi(t_0)} \int_{t_0-\delta}^{t_0+\delta} \exp\left[ j\frac{1}{2}\varphi''(t_0)(t-t_0)^2 \right] dt
\]

设 \(\varphi''(t_0) > 0\)（若小于零，推导过程类似，仅相位符号不同）。令变量替换 \(\tau = t - t_0\)，积分区间变换为 \([-\delta, \delta]\)。为进一步化为标准菲涅尔积分，令：
\[
\frac{1}{2}\varphi''(t_0)\tau^2 = \frac{\pi}{2}y^2
\]
由此可得 \(\tau = \sqrt{\frac{\pi}{\varphi''(t_0)}} y\)，积分上下限随之变换。当 \(\tau = \pm\delta\) 时，新的积分上限 \(Y\) 和下限 \(-Y\) 为：
\[
Y = \sqrt{\frac{\varphi''(t_0)}{\pi}} \delta
\]
此时积分变为：
\[
I \approx a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{\pi}{\varphi''(t_0)}} \int_{-Y}^{Y} e^{j\frac{\pi}{2}y^2} dy
\]
被积函数为偶函数，可化简为：
\[
I \approx 2a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{\pi}{\varphi''(t_0)}} \int_{0}^{Y} e^{j\frac{\pi}{2}y^2} dy
\]
此时积分上限为有限值 \(Y\)，并非无穷大。

已知标准菲涅尔积分的无穷上限结果为：
\[
\int_{0}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}}
\]
为将有限上限 \(Y\) 替换为无穷大，需考虑尾部积分的误差：
\[
\int_{0}^{Y} e^{j\frac{\pi}{2}y^2} dy = \int_{0}^{\infty} e^{j\frac{\pi}{2}y^2} dy - \int_{Y}^{\infty} e^{j\frac{\pi}{2}y^2} dy
\]
对尾部积分进行分部积分：
\[
\int_{Y}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \int_{Y}^{\infty} \frac{1}{j\pi y} d\left( e^{j\frac{\pi}{2}y^2} \right) = \left[ \frac{e^{j\frac{\pi}{2}y^2}}{j\pi y} \right]_{Y}^{\infty} + \int_{Y}^{\infty} \frac{1}{j\pi y^2} e^{j\frac{\pi}{2}y^2} dy
\]
当 \(Y \to \infty\) 时，边界项在无穷远处趋于0，在 \(Y\) 处的量级为 \(O(1/Y)\)；第二项积分的量级同样为 \(O(1/Y)\)。因此：
\[
\int_{Y}^{\infty} e^{j\frac{\pi}{2}y^2} dy = O\left(\frac{1}{Y}\right)
\]
在驻定相位近似中，只要邻域 \(\delta\) 选得足够大，使得 \(Y\) 足够大，该尾部积分即可忽略不计。故可将积分上限近似延拓至无穷：
\[
\int_{0}^{Y} e^{j\frac{\pi}{2}y^2} dy \approx \int_{0}^{\infty} e^{j\frac{\pi}{2}y^2} dy = \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}}
\]

将上述结果代回原式：
\[
I \approx 2a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{\pi}{\varphi''(t_0)}} \cdot \frac{1}{\sqrt{2}} e^{j\frac{\pi}{4}} = a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{\varphi''(t_0)}} e^{j\frac{\pi}{4}}
\]

若 \(\varphi''(t_0) < 0\)，通过相同的变量代换步骤，可得积分结果为：
\[
I \approx a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{|\varphi''(t_0)|}} e^{-j\frac{\pi}{4}}
\]

统一引入符号函数 \(\operatorname{sgn}(\cdot)\)，最终单驻点驻定相位公式为：
\[
\boxed{
I \approx a(t_0) e^{j\varphi(t_0)} \sqrt{\frac{2\pi}{|\varphi''(t_0)|}} \exp\left( j\frac{\pi}{4} \operatorname{sgn}(\varphi''(t_0)) \right)
}
\]

# 2 匹配滤波

## 2.1 内容

考虑信号模型：
\[
y(t) = x(t) + n(t)
\]

其中 \(x(t)\) 为已知的确定性信号，\(n(t)\) 为加性噪声（通常假设为白噪声，双边功率谱密度为 \(N_0/2\)）。

接收信号 \(y(t)\) 经过一个冲激响应为 \(h(t)\)（频域响应为 \(H(f)\)）的 LTI 系统，输出为：
\[
y_o(t) = y(t) * h(t) = x_o(t) + n_o(t)
\]

其中 \(x_o(t)\) 为输出信号分量，\(n_o(t)\) 为输出噪声分量。

匹配滤波的目标是：设计 LTI 系统 \(H(f)\)，使得在某一判决时刻 \(t_0\)，输出信号瞬时功率与输出噪声平均功率之比（即输出信噪比）达到最大。

输出信噪比最大的 LTI 系统频率响应为：
\[
\boxed{
H(f) = k X^*(f) e^{-j 2\pi f t_0}
}
\]
对应的时域冲激响应为：
\[
\boxed{
h(t) = k x^*(t_0 - t)
}
\]
其中 \(k\) 为任意非零常数，\(X(f)\) 为 \(x(t)\) 的傅里叶变换，\(t_0\) 为判决时刻。此时的最大输出信噪比为：
\[
\boxed{
\text{SNR}_{\max} = \frac{2E}{N_0}
}
\]
其中 \(E = \int_{-\infty}^{+\infty} |x(t)|^2 dt = \int_{-\infty}^{+\infty} |X(f)|^2 df\) 为信号 \(x(t)\) 的能量。

## 2.2 证明

输出信号分量在 \(t_0\) 时刻的值为：
\[
x_o(t_0) = \int_{-\infty}^{+\infty} H(f) X(f) e^{j 2\pi f t_0} df
\]
输出噪声 \(n_o(t)\) 的功率谱密度为 \(P_{n_o}(f) = |H(f)|^2 \frac{N_0}{2}\)，其平均功率为：
\[
P_{n_o} = \int_{-\infty}^{+\infty} |H(f)|^2 \frac{N_0}{2} df
\]
因此，\(t_0\) 时刻的输出信噪比定义为：
\[
\text{SNR} = \frac{|x_o(t_0)|^2}{P_{n_o}} = \frac{\left| \int_{-\infty}^{+\infty} H(f) X(f) e^{j 2\pi f t_0} df \right|^2}{\frac{N_0}{2} \int_{-\infty}^{+\infty} |H(f)|^2 df}
\]

对分子应用柯西-施瓦茨不等式：
\[
\left| \int_{-\infty}^{+\infty} H(f) X(f) e^{j 2\pi f t_0} df \right|^2 \le \left( \int_{-\infty}^{+\infty} |H(f)|^2 df \right) \left( \int_{-\infty}^{+\infty} |X(f) e^{j 2\pi f t_0}|^2 df \right)
\]
由于 \(|e^{j 2\pi f t_0}| = 1\)，上式右边为：
\[
\left( \int_{-\infty}^{+\infty} |H(f)|^2 df \right) \left( \int_{-\infty}^{+\infty} |X(f)|^2 df \right)
\]
将不等式代入信噪比表达式：
\[
\text{SNR} \le \frac{\left( \int_{-\infty}^{+\infty} |H(f)|^2 df \right) \left( \int_{-\infty}^{+\infty} |X(f)|^2 df \right)}{\frac{N_0}{2} \int_{-\infty}^{+\infty} |H(f)|^2 df} = \frac{\int_{-\infty}^{+\infty} |X(f)|^2 df}{N_0 / 2}
\]
由帕塞瓦尔定理，信号能量为 \(E = \int_{-\infty}^{+\infty} |X(f)|^2 df = \int_{-\infty}^{+\infty} |x(t)|^2 dt\)，因此：
\[
\text{SNR} \le \frac{2E}{N_0}
\]

柯西-施瓦茨不等式取等号的条件是两函数成共轭比例关系，即：
\[
H(f) = k \left( X(f) e^{j 2\pi f t_0} \right)^* = k X^*(f) e^{-j 2\pi f t_0}
\]
其中 \(k\) 为任意常数。此时输出信噪比达到最大值：
\[
\text{SNR}_{\max} = \frac{2E}{N_0}
\]

对 \(H(f)\) 取逆傅里叶变换，得到对应的时域冲激响应：
\[
h(t) = \int_{-\infty}^{+\infty} H(f) e^{j 2\pi f t} df = k \int_{-\infty}^{+\infty} X^*(f) e^{-j 2\pi f t_0} e^{j 2\pi f t} df = k \left( \int_{-\infty}^{+\infty} X(f) e^{j 2\pi f (t_0 - t)} df \right)^* = k x^*(t_0 - t)
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
发射带有载波的射频复信号，忽略常数系数，写为
\[
s_{RF}(t)=e^{j2\pi f_c t}
\operatorname{rect}\left(\frac{t-T_p/2}{T_p}\right)
e^{j\pi K(t-T_p/2)^2},
\]
其中 \(T_p\) 为脉宽，\(K\) 为调频率。该 LFM 的瞬时频率为
\[
f_i(t)=\frac{1}{2\pi}\frac{d}{dt}\left[\pi K(t-T_p/2)^2\right]
=K(t-T_p/2),
\]
当 \(0\le t<T_p\) 时，\(f_i(t)\) 从 \(-KT_p/2\) 变到 \(KT_p/2\)。令
\[
B=|K|T_p,
\]
则频率覆盖 \(-B/2\) 到 \(B/2\)，矩形窗中心在 \(T_p/2\)。

接收去载频后，基带回波为
\[
s_{RX\_BB}(t)=e^{-j2\pi f_c\tau}
\operatorname{rect}\left(\frac{t-\tau-T_p/2}{T_p}\right)
e^{j\pi K(t-\tau-T_p/2)^2},
\qquad
\tau=\frac{2R}{c},
\]
其中 \(e^{-j2\pi f_c\tau}\) 为延时载波相位，\(e^{j\pi K(t-\tau-T_p/2)^2}\) 为延时 LFM 回波。

匹配滤波器取
\[
H(f)=S^*(f),
\]
其中 \(S(f)\) 为发射基带信号
\[
s_{TX\_BB}(t)=
\operatorname{rect}\left(\frac{t-T_p/2}{T_p}\right)
e^{j\pi K(t-T_p/2)^2}
\]
的频谱。判决时刻取 \(t_0=0\)，否则输出时延不再对应真实回波时延 \(\tau\)。

为求 \(S(f)\)，先定义中心化 LFM 基带信号
\[
s_0(t)=\operatorname{rect}\left(\frac{t}{T_p}\right)e^{j\pi K t^2},
\]
其中 \(\operatorname{rect}\left(\frac{t}{T_p}\right)\) 在 \(-T_p/2\le t<T_p/2\) 内为 \(1\)。其频谱
\[
S_0(f)=\int_{-\infty}^{+\infty}s_0(t)e^{-j2\pi f t}dt
\]
的相位为
\[
\varphi(t)=\pi K t^2-2\pi f t.
\]
驻点满足
\[
\varphi'(t)=2\pi Kt-2\pi f=0
\quad\Rightarrow\quad
t_s=\frac{f}{K},
\]
二阶导
\[
\varphi''(t_s)=2\pi K.
\]
当 \(|f|\le B/2\) 时，\(t_s\in[-T_p/2,T_p/2]\)，驻点落在矩形窗内，由驻定相位原理得
\[
S_0(f)\approx
\frac{1}{\sqrt{|K|}}
\operatorname{rect}\left(\frac{f}{B}\right)
e^{-j\pi f^2/K}
e^{j\frac{\pi}{4}\operatorname{sgn}(K)},
\]
其中 \(\operatorname{rect}\left(\frac{f}{B}\right)\) 在 \(-B/2\le f<B/2\) 内为 \(1\)。

发射基带信号 \(s_{TX\_BB}(t)\) 是 \(s_0(t)\) 的时移：
\[
s_{TX\_BB}(t)=s_0(t-T_p/2).
\]
由时移性质得
\[
S(f)=e^{-j2\pi f T_p/2}S_0(f)
=e^{-j\pi f T_p}S_0(f).
\]
于是
\[
|S(f)|^2\approx
\frac{1}{|K|}
\operatorname{rect}\left(\frac{f}{B}\right),
\]
频域近似为矩形窗。

回波频谱为
\[
S_{RX\_BB}(f)=e^{-j2\pi f_c\tau}S(f)e^{-j2\pi f\tau}.
\]
匹配滤波输出频谱
\[
Y(f)=S_{RX\_BB}(f)H(f)
=e^{-j2\pi f_c\tau}|S(f)|^2e^{-j2\pi f\tau}.
\]
逆变换得
\[
\begin{aligned}
y(t)
&=e^{-j2\pi f_c\tau}
\frac{1}{|K|}
\int_{-B/2}^{B/2}
e^{j2\pi f(t-\tau)}df\\
&=e^{-j2\pi f_c\tau}
\frac{1}{|K|}
\frac{\sin\left[\pi B(t-\tau)\right]}{\pi(t-\tau)}\\
&=e^{-j2\pi f_c\tau}
\frac{B}{|K|}
\operatorname{sinc}\bigl(B(t-\tau)\bigr)\\
&=T_p e^{-j2\pi f_c\tau}
\operatorname{sinc}\bigl(B(t-\tau)\bigr),
\end{aligned}
\]
其中
\[
\operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}.
\]
这里频率从 \(-B/2\) 到 \(B/2\) 对称变化，积分后不会出现额外的 \(e^{j\pi B(t-\tau)}\) 线性相位。若频率从 \(0\) 到 \(B\) 变化，则积分中心在 \(B/2\)，会引入 \(e^{j\pi B(t-\tau)}\) 因子，相当于附加时移，导致时延不准。因此必须采用中心化 LFM，即频率在 \(-B/2\) 到 \(B/2\) 内变化。

时域近似为 sinc 函数，主瓣宽度由第一零点间距决定：
\[
\Delta t=\frac{1}{B}.
\]
雷达距离分辨率为
\[
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
输入信号平均功率为
\[
P_s=\frac{E}{T_p}=1,
\]
输入噪声在信号带宽 \(B\) 内的功率为
\[
P_n=\frac{N_0}{2}B,
\]
故输入信噪比为
\[
SNR_{in}=\frac{P_s}{P_n}
=\frac{1}{\frac{N_0}{2}B}
=\frac{2}{N_0 B}.
\]
于是脉冲压缩信噪比增益为
\[
G_p=\frac{SNR_{out}}{SNR_{in}}
=\frac{\frac{2T_p}{N_0}}{\frac{2}{N_0 B}}
=BT_p,
\]
即时间带宽积，也即脉冲压缩比。

# 4 动目标检测（MTD）

脉冲压缩后的回波近似为
\[
T_p e^{-j2\pi f_c\tau} \operatorname{sinc}\bigl(B(t-\tau)\bigr)。
\]

在不考虑高速目标距离走动的情况下，脉冲压缩会在时延 \(\tau\) 处压成一条亮线。忽略其他可视为不变的常量，核心信息集中于相位项 \(e^{-j2\pi f_c\tau}\)。设目标初始距离为 \(R_0\)，径向速度为 \(v\)，脉冲重复间隔为 \(T_r\)（即 \(PRT = T_r=1/PRF=1/f_r\)），第 \(m\) 个发射脉冲对应的时间为 \(t_m=mT_r\)，其中 \(m=0,1,\dots,N_c-1\)，\(N_c\) 为实际发射的 chirp 数。此时目标距离为
\[
R_m=R_0+vmT_r，
\]

对应时延
\[
\tau_m=\frac{2R_m}{c}=\frac{2R_0}{c}+\frac{2vmT_r}{c}。
\]

将其代入相位项，得
\[
    e^{-j2\pi f_c\tau_m}
    =
    e^{-j2\pi f_c\frac{2R_0}{c}}
    e^{-j2\pi \frac{2v}{\lambda}mT_r}，
\]

其中 \(\lambda=c/f_c\)。忽略常数相位 \(e^{-j2\pi f_c\frac{2R_0}{c}}\)，沿慢时间维 \(m\) 看，回波成为一个频率为
\[
f_d=-\frac{2v}{\lambda}
\]
的单频信号，这正是模拟多普勒频率。因此对慢时间维做 FFT，频谱峰值位置直接对应 \(f_d\)，测出频率也就测出了速度，这就是 FFT 能实现测速的本质。

慢时间维的采样间隔为 \(T_r\)，即采样率为 \(PRF=1/T_r=f_r\)。由离散信号采样定理，无模糊多普勒频率范围为 \([-PRF/2,PRF/2]\)，对应的无模糊测速范围为
\[
\left[-\frac{\lambda PRF}{4},\frac{\lambda PRF}{4}\right]，
\]

最大不模糊速度为
\[
v_{\max}=\frac{\lambda PRF}{4}。
\]

相干积累时间 \(T_{CPI}=N_c T_r\)，因此多普勒频率分辨率为
\[
\Delta f_d=\frac{1}{T_{CPI}}=\frac{PRF}{N_c}，
\]

对应的测速分辨率为
\[
\Delta v=\frac{\lambda}{2}\Delta f_d=\frac{\lambda PRF}{2N_c}。
\]

这里必须严格区分模拟多普勒频率与数字多普勒频率。模拟多普勒频率 \(f_d=-2v/\lambda\) 是连续的物理量；而数字多普勒频率是将 \(f_d\) 以 \(f_r\) 为采样率进行归一化得到的，即 \(f'_d=f_d/f_r\)，其范围为 \([-1/2, 1/2)\)。FFT 输出的离散频率索引为 \(k=0,1,\dots,N_d-1\)，其中 \(N_d\) 是 FFT 的点数。经过 `fftshift` 后，索引变为 \(k=-N_d/2,\dots,N_d/2-1\)，对应的数字频率为 \(k/N_d\)。因此真实的模拟频率轴为
\[
f_k = \frac{k}{N_d} \times PRF, \qquad k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1。
\]
由 \(f_d=-2v/\lambda\) 可得速度轴，速度轴在频率轴的基础上乘以 \(-\lambda/2\)，即
\[
v_k = -\frac{\lambda}{2} f_k = -\frac{\lambda}{2} \frac{k}{N_d} PRF, \qquad k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1。
\]

实际工程中，通常 \(N_c\) 和 \(N_d\) 均取为 2 的幂，即 \(N_c=N_d=2^n\)。\(N_c\) 是实际发射的 chirp 数，它决定了物理相干积累时间和真实的物理分辨率，\(\Delta v=\lambda PRF/(2N_c)\) 是不可突破的物理极限。而 \(N_d\) 是 FFT 的点数，当 \(N_d=N_c\) 时，数字频率分辨率与物理分辨率一致；当 \(N_d>N_c\) 时（通常通过补零实现），频谱被插值，采样点变密，但这只能使频率读数更平滑，并不能提高真实的物理分辨率。因此用 \(k/N_d\) 可以测频率，但真实分辨率始终由 \(N_c\) 决定。

由于仅有有限的 \(N_c\) 个脉冲，相当于对慢时间维信号加了矩形窗截断，直接 FFT 会产生 sinc 泄露，其第一旁瓣仅比主瓣低约 \(-13.2\) dB，强目标的旁瓣可能掩盖弱目标。因此通常需要加窗以降低旁瓣。常用的窗包括矩形窗、Hamming 窗、Hann 窗、Blackman 窗等。矩形窗主瓣最窄，分辨率最好，但旁瓣最高；Hamming 窗旁瓣可降至 \(-42\) dB 左右，但主瓣会加宽；Hann 窗旁瓣更低，主瓣更宽；Blackman 窗旁瓣极低，但主瓣最宽。窗的选择取决于对主瓣分辨率和旁瓣抑制的折衷要求，若需分辨邻近强/弱目标，通常选择 Hamming 或 Taylor 窗以兼顾旁瓣抑制和主瓣展宽。

做完 FFT 后需要执行 shift 操作（如 `fftshift`）。常规 FFT 输出 \(k=0,1,\dots,N_d-1\) 对应的频率从 \(0\) 到 \(PRF\)，但实际多普勒频率是正负对称的，目标靠近对应负多普勒，远离对应正多普勒。若不 shift，零频（静止目标）位于数组首位，正负多普勒分居两端，频率读数不直观，且 \(0\) 和 \(PRF\) 附近的模糊区域容易混淆。通过 `fftshift` 将零频移到中心，左右两侧分别对应负多普勒和正多普勒，这样频率轴就变成 \([-PRF/2,PRF/2)\)，即离散形式 \((-N_d/2, N_d/2-1)/N_d \times PRF\)，与实际物理意义完全对应，便于正确判别目标运动方向和读取速度。

# 5 动目标显示（MTI）

雷达回波中除运动目标外，还存在大量杂波，如地物、云雨等。杂波功率通常远强于目标，且多普勒频率集中在零频附近。若直接做 MTD，强杂波的旁瓣会淹没慢速小目标。因此需要在 MTD 之前滤除零频附近的杂波，这就是 MTI。MTI 本质上是零频凹口滤波器。

单延迟线对消器的时域形式为
\[
y(m)=x(m)-x(m-1)，
\]
其 Z 变换为
\[
H(z)=1-z^{-1}。
\]
令 \(z=e^{j\omega}\)，\(\omega\) 为数字角频率，则频响为
\[
H(e^{j\omega})=1-e^{-j\omega}
=2j e^{-j\omega/2}\sin\left(\frac{\omega}{2}\right)，
\]
幅度响应为
\[
|H(e^{j\omega})|=2\left|\sin\left(\frac{\omega}{2}\right)\right|。
\]
在 \(\omega=0\) 处为零，形成零频凹口，其形状就是 sinc 型凹口。

双延迟线对消器的时域形式为
\[
y(m)=x(m)-2x(m-1)+x(m-2)，
\]
Z 变换为
\[
H(z)=(1-z^{-1})^2，
\]
频响幅度为
\[
|H(e^{j\omega})|=4\sin^2\left(\frac{\omega}{2}\right)。
\]
双对消器的凹口比单对消器更宽、更深，对消效果更好，但代价是主瓣增益损失更大，对慢速目标的抑制更强。实际工程中，若杂波带宽较窄、目标速度较高，单对消即可满足；若杂波频谱较宽或对消深度要求更高，则采用双对消或更高阶对消器。

脉冲压缩之后，MTI 与 MTD 的先后顺序有两种常见方案。第一种是脉冲压缩后先在时域做相邻脉冲对消，再加窗做 MTD：
\[
\text{脉冲压缩} \rightarrow \text{时域对消} \rightarrow \text{加窗} \rightarrow \text{MTD}。
\]
第二种是脉冲压缩后先加窗做 MTD，再在频域做对消：
\[
\text{脉冲压缩} \rightarrow \text{加窗} \rightarrow \text{MTD} \rightarrow \text{频域对消}。
\]
两种方案在数学上都是合理的，因为时域差分与频域相乘本质上等价。但工程上一般选择第一种方案，即先在时域做 MTI 对消，再加窗做 MTD。原因是先对消可以在进入 FFT 之前就抑制强杂波，避免其旁瓣泄漏污染整个多普勒谱；若先做 FFT，杂波能量已经通过旁瓣扩散到所有多普勒通道，此时再在频域乘零频凹口，只能抑制零频附近主瓣区域的杂波，已经泄漏到其他通道的杂波旁瓣无法消除，弱目标仍可能被淹没。因此，实际系统通常采用先时域对消、再加窗 MTD 的处理顺序。

# 6 细节

我们按照C/C++，Python的内存布局，遵守行主序原则，

原始数据矩阵为 \(\text{raw}[N_c][N_w]\)，每一行是一个脉冲的接收窗采样，列是距离维快时间。发射 LFM 为 \(s_{tx}[N_s]\)，\(N_s=\lfloor T_p f_s\rfloor\)。接收窗从 \(t_{\text{start}}\) 到 \(t_{\text{end}}\)，满足 \(t_{\text{start}}\ge T_p\)，\(t_{\text{end}}\le T_r\)，\(T_r=1/PRF\)，窗内点数 \(N_w=\lfloor (t_{\text{end}}-t_{\text{start}})f_s\rfloor\)。最终期望得到多普勒-距离图 \(\text{drmap}[N_d][N_r]\)，其中 \(d\) 在前，\(r\) 在后，\(N_r\) 为保留的距离点数。

距离维匹配滤波直接在频域完成。取 FFT 长度 \(N_{\text{fft}}\ge N_s+N_w-1\)，通常取下一个 2 的幂。对发射 LFM 补零后做 FFT，取共轭得匹配滤波器频响
\[
H[k]=\operatorname{conj}\left(\operatorname{FFT}_{N_{\text{fft}}}\{s_{tx}\}[k]\right)。
\]
对 \(\text{raw}\) 的每一行补零到 \(N_{\text{fft}}\)，做 FFT，乘以 \(H[k]\)，再 IFFT，得到线性卷积结果。由于卷积结果第 \(0\) 个点对应时刻 \(t_{\text{start}}-(N_s-1)T_s\)，\(T_s=1/f_s\)，因此截取索引范围
\[
N_s-1 : N_s-1+N_r
\]
得到脉冲压缩后矩阵 \(\text{pc}[N_c][N_r]\)。距离轴为
\[
r_j=\frac{c}{2}t_{\text{start}}+j\Delta r,\qquad j=0,1,\dots,N_r-1，
\]
其中距离单元格大小为
\[
\Delta r=\frac{c}{2f_s}。
\]
真实物理分辨率为
\[
\delta r=\frac{c}{2B}，
\]
过采样时 \(\Delta r<\delta r\)，一个分辨单元对应多个距离单元格。

若需 MTI，对 \(\text{pc}\) 沿慢时间维（行）做对消。单对消为
\[
f[m,:]=pc[m,:]-pc[m-1,:],\qquad m=1,\dots,N_c-1，
\]
边界 \(m=0\) 无前值，可置零或丢弃。双对消为
\[
f[m,:]=pc[m,:]-2pc[m-1,:]+pc[m-2,:],\qquad m=2,\dots,N_c-1，
\]
边界 \(m=0,1\) 可置零或丢弃。通常直接保留有效行，单对消输出 \(N_c-1\) 行，双对消输出 \(N_c-2\) 行。对消后沿慢时间维乘窗函数 \(w[m]\)，如 Hamming、Hann 等。

慢时间维 MTD 对加窗后的矩阵沿行方向做 FFT，补零到 \(N_d\) 点，得到
\[
\text{drmap}[N_d][N_r]=\operatorname{fftshift}\left(\operatorname{FFT}_{N_d}\{\text{pc}_{\text{win}}\},\ \text{axis}=0\right)。
\]
频率轴为
\[
f_k=\frac{k}{N_d}PRF,\qquad k=-\frac{N_d}{2},\dots,\frac{N_d}{2}-1，
\]
数字频率 \(k/N_d\) 对应实际多普勒频率。速度轴由 \(f_d=-2v_r/\lambda\) 得
\[
v_k=-\frac{\lambda}{2}f_k=-\frac{\lambda}{2}\frac{k}{N_d}PRF。
\]
速度单元格大小为
\[
\Delta v=\frac{\lambda}{2N_d T_r}，
\]
真实速度分辨率为
\[
\delta v=\frac{\lambda}{2N_c T_r}。
\]
若 \(N_c=N_d=2^n\)，则 \(\Delta v=\delta v\)；若 \(N_d>N_c\)，则 \(\Delta v<\delta v\)，每个分辨单元对应多个速度单元格。\(\Delta v\) 是格子大小，\(\delta v\) 是物理分辨率，二者必须区分。

伪代码如下。

```
# 输入：raw[Nc][Nw], s_tx[Ns], 参数 Tp, fs, B, K, Nc, Nd, Nr, PRF, lambda, c
Ns = floor(Tp * fs)
Nw = floor((t_end - t_start) * fs)
N_fft = nextpow2(Ns + Nw - 1)

# 距离维匹配滤波
S_tx = fft(s_tx, N_fft)
H = conj(S_tx)
pc = zeros(Nc, Nr)
for m in range(Nc):
    Raw = fft(raw[m], N_fft)
    conv_full = ifft(Raw * H)
    pc[m] = conv_full[Ns-1 : Ns-1+Nr]

# 可选 MTI
if use_MTI:
    if mode == 'single':
        f = zeros(Nc, Nr)
        for m in range(1, Nc):
            f[m, :] = pc[m, :] - pc[m-1, :]
        # f[0, :] 置零或丢弃
        pc_use = f
    elif mode == 'double':
        f = zeros(Nc, Nr)
        for m in range(2, Nc):
            f[m, :] = pc[m, :] - 2*pc[m-1, :] + pc[m-2, :]
        # f[0, :], f[1, :] 置零或丢弃
        pc_use = f
else:
    pc_use = pc

# 加窗
w = window(pc_use.shape[0])   # 如 Hamming
pc_win = pc_use * w[:, None]

# MTD
drmap = fftshift(fft(pc_win, n=Nd, axis=0), axes=0)   # [Nd][Nr]

# 坐标轴
r_axis = c/2 * t_start + arange(Nr) * c/(2*fs)
f_axis = arange(-Nd/2, Nd/2) / Nd * PRF
v_axis = -lambda/2 * f_axis

# 绘图
plot_2d(abs(drmap), r_axis, v_axis)
plot_3d(abs(drmap), r_axis, v_axis)
```

三个版本中，Python CPU 版用 NumPy 批量 FFT，Python CuPy 版将数组迁至 GPU 用 CuPy 对应接口，CUDA C/C++ 版手写核函数做距离维频域匹配滤波与慢时间维 FFT，并用 Nsight 分析访存与占用率。原始数据 \(\text{raw}[N_c][N_w]\) 是输入，最终输出 \(\text{drmap}[N_d][N_r]\) 即多普勒-距离图。

