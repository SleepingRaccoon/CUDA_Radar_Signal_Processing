# CBF：常规波束形成

## 信号模型

窄带、远场、平面波假设下，$M$ 元均匀线阵（ULA），阵元间距 $d$，波长为 $\lambda$。空间有 $P$ 个目标，第 $p$ 个目标的复包络为 $s_p[n]$，以角度 $\theta_p$ 入射（相对阵列法线的夹角）。

先看单个目标。从 $\theta_p$ 方向来的信号，相邻阵元波程差为 $d\sin\theta_p$，相位差：

$$ \Delta\phi_p = \frac{2\pi d\sin\theta_p}{\lambda} $$

以 0 号阵元为参考，第 $p$ 个目标的导向矢量（steering vector）：

$$ \mathbf{a}(\theta_p) = \left[1,\; e^{j\frac{2\pi d\sin\theta_p}{\lambda}},\; \ldots,\; e^{j(M-1)\frac{2\pi d\sin\theta_p}{\lambda}}\right]^T $$


接收数据矩阵：

$$ \mathbf{X} = \mathbf{A}\mathbf{S} + \mathbf{N} $$

| 矩阵 | 符号 | 维度 | 分块表示 |
|---|---|---|---|
| 接收数据矩阵 | $\mathbf{X}$ | $M \times N$ | $\mathbf{X} = [\mathbf{x}[0], \ldots, \mathbf{x}[N-1]]$，第 $n$ 列 $\mathbf{x}[n]$ 是第 $n$ 个快拍的 $M \times 1$ 接收向量 |
| 阵列流型矩阵 | $\mathbf{A}$ | $M \times P$ | $\mathbf{A} = [\mathbf{a}(\theta_1), \ldots, \mathbf{a}(\theta_P)]$，每列是一个目标的导向矢量 |
| 信号矩阵 | $\mathbf{S}$ | $P \times N$ | $\mathbf{S} = [\mathbf{s}[0], \ldots, \mathbf{s}[N-1]]$，第 $n$ 列 $\mathbf{s}[n] = [s_1[n], \ldots, s_P[n]]^T$ 是第 $n$ 个快拍 $P$ 个目标的包络向量 |
| 噪声矩阵 | $\mathbf{N}$ | $M \times N$ | $\mathbf{N} = [\mathbf{n}[0], \ldots, \mathbf{n}[N-1]]$，第 $n$ 列 $\mathbf{n}[n]$ 是第 $n$ 个快拍的噪声向量 |

按列分块，$\mathbf{X} = \mathbf{A}\mathbf{S} + \mathbf{N}$ 展开为 $\mathbf{x}[n] = \mathbf{A}\mathbf{s}[n] + \mathbf{n}[n]$，$n = 0, \ldots, N-1$。

## 波束方向图

对来向固定为 $\theta_0$ 的目标，波束指向 $\theta$ 时 CBF 输出的复增益（array factor）：

$$ B(\theta) = \mathbf{w}^H(\theta)\,\mathbf{a}(\theta_0) = \sum_{m=0}^{M-1} e^{jm\frac{2\pi d}{\lambda}(\sin\theta_0 - \sin\theta)} $$

等比级数可闭式求和，令 $u = \frac{2\pi d}{\lambda}(\sin\theta - \sin\theta_0)$：

$$ B(\theta) = \sum_{m=0}^{M-1} e^{-jmu} = \frac{1 - e^{-jMu}}{1 - e^{-ju}} = e^{-j\frac{M-1}{2}u}\,\frac{\sin(Mu/2)}{\sin(u/2)} $$

相位因子 $e^{-j\frac{M-1}{2}u}$ 是阵列相位中心引起的线性相位，对功率谱 $|B(\theta)|^2$ 无影响，取模后得 sinc 型方向图：主瓣在波束指向与来向一致处（$\theta = \theta_0$，峰值 $M$），第一旁瓣相对主瓣约 $-13$ dB，3 dB 波束宽度：

$$ \text{BW}_{3\text{dB}} \approx 0.886\,\frac{\lambda}{Md\cos\theta_0} $$

波束宽度反比于阵列孔径 $Md$——孔径越大波束越窄。这就是 CBF 的角度分辨率极限（瑞利限）：两个来波夹角小于波束宽度时无法分辨。

## 空间谱

遍历扫描角 $\theta$（候选来波方向），对每个方向求输出功率，得到空间谱：

$$ P(\theta) = E\left[|y|^2\right] = \mathbf{a}^H(\theta)\,\mathbf{R}\,\mathbf{a}(\theta) $$

其中 $\mathbf{R} = E[\mathbf{x}\mathbf{x}^H]$ 是 $M \times M$ 接收协方差矩阵。由多目标模型 $\mathbf{x} = \mathbf{A}\mathbf{s} + \mathbf{n}$，在信号与噪声不相关、各目标信号互不相关的假设下：

$$ \mathbf{R} = \mathbf{A}\,\mathbf{R}_s\,\mathbf{A}^H + \sigma^2\mathbf{I} $$

其中 $\mathbf{R}_s = E[\mathbf{s}\mathbf{s}^H]$ 是信号协方差矩阵（$P \times P$），$\sigma^2$ 是噪声功率。实际用 $N$ 个快拍做样本平均估计：

$$ \hat{\mathbf{R}} = \frac{1}{N}\sum_{n=0}^{N-1}\mathbf{x}[n]\,\mathbf{x}^H[n] = \frac{1}{N}\mathbf{X}\mathbf{X}^H $$


