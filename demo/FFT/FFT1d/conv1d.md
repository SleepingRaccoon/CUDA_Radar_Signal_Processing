# 一维卷积

直接卷积的定义为：

$$ y[n] = \sum_{k} x[k] \cdot h[n-k] $$

其中 $x$ 长度为 $M$，$h$ 长度为 $N$，结果长度为 $M+N-1$。直接计算的时间复杂度是 $O(M \cdot N)$，当 $M$ 和 $N$ 较大时成本很高。

另一种方式是借助卷积定理：

$$ \mathcal{F}\{x * h\} = \mathcal{F}\{x\} \cdot \mathcal{F}\{h\} $$

其中 $\mathcal{F}$ 表示傅里叶变换，$*$ 表示卷积。时域的卷积等价于频域的逐点相乘。

快速卷积的步骤：

1. 对 $x$ 和 $h$ 分别做 FFT，得到 $X$ 和 $H$
2. 计算 $Y = X \cdot H$（逐点复数乘法）
3. 对 $Y$ 做逆 FFT，得到卷积结果 $y$

整体时间复杂度为 $O(L \log L)$，其中 $L$ 是 FFT 的长度。

由于线性卷积的长度为 $M+N-1$，需要将 $x$ 和 $h$ 都补零到 $L \ge M+N-1$，本例中取 $L = M+N-1$，以避免循环卷积的混叠。

cuFFT 的正、逆变换都不自动归一化，所以频域点乘的结果要乘以 $1/L$（本例在逐点乘法 kernel 中乘；由乘法线性性，等价于对逆变换输出乘 $1/L$）。不乘则幅值偏大 $L$ 倍。

性能对比方面，分别测量 CPU 直接卷积和 GPU 快速卷积的平均耗时，两者比值即为加速比。GPU 的优势来自 FFT 的高度并行化，数据量越大加速效果越明显。

若需要提取与深度学习框架中 same 卷积对应的有效区间，应取卷积结果的中心部分，起始位置为 $\lfloor (N-1)/2 \rfloor$，长度为 $M$，此时核的中心与信号每个位置对齐。

---

## 高级 plan API：cufftXtMakePlanMany

完整签名（逐参数拆解）：

```c
cufftResult cufftXtMakePlanMany(
    cufftHandle plan,        // cufftCreate() 创建的计划句柄
    int rank,                // 维度数：1 / 2 / 3
    long long int *n,        // 长度 rank 的数组，每维的变换长度，如 {8,8,8}
    // ---------- 输入侧布局 ----------
    long long int *inembed,  // 输入物理嵌入尺寸；NULL = 紧密连续
    long long int istride,   // 输入相邻元素间距（单位：元素，通常 1）
    long long int idist,     // 输入相邻两个 batch 的起始间距（单位：元素）
    cudaDataType inputtype,  // 输入数据类型，如 CUDA_C_32F
    // ---------- 输出侧布局 ----------
    long long int *onembed,  // 输出物理嵌入尺寸；NULL = 紧密连续
    long long int ostride,   // 输出相邻元素间距（通常 1）
    long long int odist,     // 输出相邻两个 batch 的起始间距
    cudaDataType outputtype, // 输出数据类型
    // ---------- 批量与执行 ----------
    long long int batch,     // 一次执行几个独立变换
    size_t *workSize,        // 返回所需 workspace 字节数（可 NULL）
    cudaDataType executiontype); // 执行精度（可与存储精度不同）
```

### 索引公式：cuFFT 怎么找元素

每个元素在内存中的位置由下式决定（核心）：

$$
\text{offset} = b \cdot \text{idist} + j_0 \cdot s_0 + j_1 \cdot s_1 + j_2 \cdot s_2
$$

其中：

- $b$：第几个 batch，$j_0, j_1, j_2$：各维坐标
- $s_0, s_1, s_2$：各维 stride，由嵌入尺寸推导：
  - `inembed = NULL`：$s_j = n_{j+1} \cdot n_{j+2} \cdots$（紧密行主序、无空洞）
  - `inembed = {E0, E1, E2}`：$s_j = E_{j+1} \cdot E_{j+2} \cdots$（数据物理上放在更大的数组里，$n$ 是其中的子块）
- `istride`：相邻两个元素的间距（默认 1 = 连续）
- `idist`：相邻两个 batch 起始点的间距（单位：元素，不是字节）

输出侧同理（`onembed` / `ostride` / `odist`）。

### 场景 1：大矩阵中，对某一块做 FFT（inembed）

物理矩阵 $M \times N$（行主序），要对从 $(r_0, c_0)$ 开始的 $h \times w$ 子块做 FFT：

```
物理矩阵 M×N:
  ┌─────────────────────────┐
  │                         │
  │      ┌─────────┐        │
  │      │   h×w   │        │   ← 起点 (r0, c0)
  │      └─────────┘        │
  │                         │
  └─────────────────────────┘
```

做法分两件事：

1. **指针移到块起点**（inembed 没有"偏移"参数，块的位置靠指针表达）；
2. **n 填逻辑块尺寸，inembed 填物理矩阵尺寸**（用来推 stride）。

```c
long long n[2]       = {h, w};       // 逻辑：块
long long inembed[2] = {M, N};       // 物理：整个矩阵
cufftComplex *d_block = d_mat + r0 * N + c0;   // 指针移到块起点

cufftXtMakePlanMany(plan, 2, n,
                    inembed, 1, 1, CUDA_C_32F,   // 行 stride 自动 = N（物理行宽）
                    inembed, 1, 1, CUDA_C_32F,
                    1, &work_size, CUDA_C_32F);
cufftExecC2C(plan, d_block, d_block, CUFFT_FORWARD);   // in-place，只变换块
```

代入索引公式（b = 0，istride = 1）：块内元素 $(i, j)$ 偏移 = $i \cdot N + j$——**行 stride 是物理行宽 N，不是逻辑块宽 w**，这就是 inembed 的作用。

### 场景 2：RGBRGB 排列的图像，做 FFT（stride）

图像按像素交错存储：内存是 `R G B R G B ...`，即 $[H][W][3]$ 布局（每像素 3 个分量连续）。现在只对 **R 通道**做二维 FFT（$n = \{H, W\}$）：

- 同一通道相邻像素（同行相邻列）中间隔着 G、B：**元素间距 `istride = 3`**
- 物理行宽是 $3W$ 个元素：`inembed = {H, 3*W}`，行 stride 自动 = 3W
- 通道内元素 $(y, x)$ 偏移 = $y \cdot 3W + 3x$（行 stride 3W，列 stride 3）

```c
long long n[2]       = {H, W};
long long inembed[2] = {H, 3 * W};   // 物理行宽 3W

cufftXtMakePlanMany(plan, 2, n,
                    inembed, 3, 1, CUDA_C_32F,   // istride=3：同通道相邻像素隔 3
                    inembed, 3, 1, CUDA_C_32F,
                    1, &work_size, CUDA_C_32F);
```

### 场景 3：多批次 RGBRGB 排列的图像（stride + batch）

B 张这样的图，内存 `[B][H][W][3]` 一路排下来。注意：**一张图内 R/G/B 三个通道的起点是相邻的（间距 1），但跨图间距是 3HW，间距不恒定**——cuFFT 的 batch 要求等距，所以一张图内的 3 个通道用 batch=3、idist=1 一次算完，跨图用循环 + 指针偏移：

```c
long long n[2]       = {H, W};
long long inembed[2] = {H, 3 * W};

cufftXtMakePlanMany(plan, 2, n,
                    inembed, 3, 1, CUDA_C_32F,     // stride=3，idist=1
                    inembed, 3, 1, CUDA_C_32F,
                    3, &work_size, CUDA_C_32F);    // batch=3：R、G、B 三个通道

for (int b = 0; b < B; ++b) {
    // 第 b 张图的 3 个通道一次算完，指针移 b * 3HW
    cufftExecC2C(plan, d_img + b * 3 * H * W, d_img + b * 3 * H * W, CUFFT_FORWARD);
}
```

代入索引公式：第 b 张图第 c 通道（$c \in \{R,G,B\}$）的 $(y, x)$ 元素偏移：

$$
b \cdot 3HW + c \cdot 1 + y \cdot 3W + 3x
$$

正好就是 `[B][H][W][3]` 交错布局的地址。参数各司其职：`istride = 3` 管通道内列方向、`inembed` 管行方向、`idist = 1` 管同图内 R/G/B 三个 batch、`batch = 3` 管通道数，跨图由循环指针偏移 `b * 3HW` 承担。
