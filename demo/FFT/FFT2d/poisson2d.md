# 2D 泊松方程的 FFT 求解

## 问题

要解的方程是二维泊松方程：

$$
\nabla^2 u + f = 0
$$

求解域是 $[0,L_x]\times[0,L_y]$。$f$ 已知，$u$ 未知。

我们取 $f$ 在网格点上的值作为输入：

$$
f_{ij} = f(i h_x,\ j h_y), \qquad
i = 0,\dots,N_x-1, \quad j = 0,\dots,N_y-1
$$

其中 $h_x = L_x/N_x$，$h_y = L_y/N_y$。

目标是求出 $u$ 在网格点上的值 $u_{ij} = u(i h_x,\ j h_y)$。

## CFS 到 FFT

周期延拓。CFS 展开要求函数是周期的，把 $f$、$u$ 从 $[0,L_x]\times[0,L_y]$ 做周期延拓（或直接假设周期），周期分别为 $L_x$、$L_y$。

连续傅里叶级数。$f$、$u$ 展开为复指数基 $\phi_{mn} = e^{2\pi i(mx/L_x + ny/L_y)}$：

$$
f(x,y) = \sum_{m,n} \hat f_{mn}\,\phi_{mn}, \qquad
u(x,y) = \sum_{m,n} \hat u_{mn}\,\phi_{mn}
$$

基函数是拉普拉斯算子的特征函数：

$$
\nabla^2 \phi_{mn} = -\lambda_{mn}\,\phi_{mn}, \qquad
\lambda_{mn} = \left(\frac{2\pi m}{L_x}\right)^2 + \left(\frac{2\pi n}{L_y}\right)^2
$$

代入 $\nabla^2 u + f = 0$，由正交性逐项相等，得到系数关系：

$$
\hat u_{mn} = \frac{\hat f_{mn}}{\lambda_{mn}}, \qquad (m,n) \neq (0,0)
$$

零频 $\lambda_{00} = 0$ 处 $\hat u_{00}$ 无法由方程确定（对应常数自由度），需另行约定，后面再处理。

DFT 系数与级数系数的关系。一维情形：周期 $L$ 函数 $f$，级数系数 $C_m$，采样 $f_n = f(nL/N)$。在采样点处基函数取值（$n$ 为整数索引）：

$$
e^{2\pi i m x_n/L} = e^{2\pi i mn/N}
$$

频率相差 $N$ 的基函数在采样点上不可区分：

$$
e^{2\pi i (m+N) n/N} = e^{2\pi i mn/N}
$$

代入 DFT 定义，内层是 $N$ 点几何级数，只在 $m-k$ 为 $N$ 的倍数时非零：

$$
X_k = \sum_{n=0}^{N-1} f_n\, e^{-2\pi i kn/N}
= \sum_m C_m \sum_{n=0}^{N-1} e^{2\pi i (m-k)n/N}
= N \sum_{j\in\mathbb{Z}} C_{k+jN}
$$

即一维 DFT 系数是级数系数的混叠和。二维直接类比（$X$ 方向模 $N_x$ 折叠，$Y$ 方向模 $N_y$ 折叠）：

$$
X_{kl} = \sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1} f_{ij}\, e^{-2\pi i(ki/N_x + lj/N_y)}
= N_x N_y \sum_{p,q\in\mathbb{Z}} \hat f_{\,k+pN_x,\; l+qN_y}
$$

用 FFT 代替 CFS。系数关系 $\hat u = \hat f/\lambda$ 需要连续级数系数 $\hat f_{mn}$，但我们只能通过 FFT 得到混叠和 $X_{kl}$。做法：用 FFT 算出的混叠和代替 CFS 系数代入系数关系。

若 $f$ 本身是带限的（$\hat f_{mn} = 0$ 当 $|m| \ge N_x/2$ 或 $|n| \ge N_y/2$），混叠项全部为零，这时

$$
X_{kl} = N_x N_y\, \hat f_{kl}
$$

与 $\hat f_{kl}$ 精确相等（只差常数因子 $N_xN_y$，归一化时处理）。

## 算法具体实现

输入 $f_{ij}$，输出 $u_{ij}$，共四步。

1. 二维 FFT。把 $f$ 的混叠和算出来：

$$
X_{kl} = \text{FFT2}(f)_{kl}, \qquad
k = 0,\dots,N_x-1, \quad l = 0,\dots,N_y-1
$$

2. 构造物理波数。FFT 输出的桶 $k > N_x/2$ 装的是负频率系数（$m = k-N_x$），波数必须取负：

$$
\omega_x(k) = \begin{cases} \dfrac{2\pi k}{L_x}, & k \le \dfrac{N_x}{2} \\[10pt] \dfrac{2\pi(k-N_x)}{L_x}, & k > \dfrac{N_x}{2} \end{cases}, \qquad
\omega_y(l) = \begin{cases} \dfrac{2\pi l}{L_y}, & l \le \dfrac{N_y}{2} \\[10pt] \dfrac{2\pi(l-N_y)}{L_y}, & l > \dfrac{N_y}{2} \end{cases}
$$

> 为什么要折到 $-N/2$ 到 $N/2$：$f$ 是实函数，频谱共轭对称，正负频率成对出现。但 FFT 只能输出 $0..N-1$ 这 $N$ 个桶，周期延拓性质把负频系数折进了高半段桶，所以 $k > N/2$ 的桶装的本来就是负频率。因此波数必须取负，范围正好是 $-\pi N/L$ 到 $+\pi N/L$（奈奎斯特频率）。

3. 频域求解。由系数关系 $\hat u = \hat f/\lambda$ 与混叠关系 $X = N_xN_y\,\hat f$ 合并（$N_xN_y$ 因子与第 4 步逆变换的归一化抵消）：

$$
\hat U_{kl} = \frac{X_{kl}}{\omega_x(k)^2 + \omega_y(l)^2}, \qquad (k,l) \neq (0,0)
$$

零频 $(0,0)$ 处分母为零：方程要求 $\hat f_{00} = 0$（即 $\int f = 0$，否则无解），此时 $\hat U_{00}$ 自由，约定取 0（钉住常数自由度）。

4. 二维逆 FFT 回时域（标准归一化逆 DFT）：

$$
u_{ij} = \frac{1}{N_x N_y} \sum_{k,l} \hat U_{kl}\, e^{2\pi i(ki/N_x + lj/N_y)}
$$

说明：混叠因子 $N_xN_y$（步骤 3 前）与逆变换的 $1/(N_xN_y)$ 精确抵消，故频域解直接除 $\lambda$ 即可。若所用的 FFT 库逆变换不归一化（如 cuFFT），需在频域多除一个 $N_xN_y$（即 $\hat U_{kl} = X_{kl}/\big(N_xN_y\lambda_{kl}\big)$），或逆变换后手动乘 $1/(N_xN_y)$。
