/*
============================================================
hgemv.cu -- 行主序 HGEMV（half 精度）：y[m] = sum_k A[m][k] * x[k]

三个 kernel 的适用条件（K = 列数，约定 K 规则整齐，面向 AI 算子场景）：
  k16  : K <= 16 且为 2 的幂，一个 warp 拆成多行并行（行内 K 个 lane 归约）；
  k32  : K 是 32 的倍数，一行分配一个 warp，lane 沿 K 方向标量累加；
  k64  : K 是 64 的倍数，一行分配一个 warp，每 lane 用 half2 一次取 2 元素。
         （K 是 128 的倍数也满足 64 的倍数条件，直接用本 kernel，无需单独版本）

精度设计：A、x 为 half，但乘法与累加全程在 float 上进行，
          避免 half*half 的截断误差与累加舍入，写回 y 时才转 half。

主程序对 K=16 与 K=1024 两组场景做 CPU / cuBLAS / 自写 kernel 对比：
  K=16   -> k16
  K=1024 -> k32 与 k64 同条件对比（1024 同时是 32、64 的倍数）

指标：正确性(最大相对误差) / 平均耗时 / 有效带宽 / 带宽利用率 /
      GFLOPS / 算术强度(FLOP/byte) / vsCPU

算术强度 = 总运算次数 / 总读写字节数。HGEMV 总运算 = 2MK，
读写字节 = A(M*K) + x(K) + y(M)，half 每元素 2 字节。
算术强度低(<2)说明访存密集，高说明计算密集。

注：cuBLAS 无 fp16 GEMV 单 API，故只对比 CPU，不做 vsBLAS。

Build:
  nvcc -o demo\kernels\hgemv\hgemv.exe demo\kernels\hgemv\hgemv.cu -Xcompiler /utf-8
Run:
  .\demo\kernels\hgemv\hgemv.exe
============================================================
*/

#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

/* 一个 warp 内若干线程的子集规约（NUM_THREADS<=32 且为 2 的幂），float 累加 */
template <int NUM_THREADS>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    static_assert(NUM_THREADS <= WARP_SIZE &&
                      ((NUM_THREADS & (NUM_THREADS - 1)) == 0),
                  "NUM_THREADS must be power of 2 <= 32");
#pragma unroll
    for (int offset = NUM_THREADS >> 1; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

/*
  适用 K <= 16 且为 2 的幂：一个 warp 同时处理 WARP_SIZE/K 行，
  行内 K 个 lane 各自乘一个元素（转 float），__shfl_xor 行内子集归约。
  注意：不提前 return，整个 warp 始终一起执行 shuffle（mask 全 32），
  越界行只参与归约、不写回。
*/
template <int K>
__global__ void hgemv_k16_f16_kernel(const half* __restrict__ A,
                                     const half* __restrict__ x,
                                     half* __restrict__ y,
                                     int M) {
    static_assert(K <= 16 && ((K & (K - 1)) == 0),
                  "K must be power of 2 <= 16");
    assert(blockDim.x == WARP_SIZE);
    constexpr int ROWS   = WARP_SIZE / K;   // 一个 warp 处理的行数
    int lane   = threadIdx.x;
    int row_id = lane / K;
    int col_id = lane % K;
    int m = ROWS * (blockIdx.x * blockDim.y + threadIdx.y) + row_id;

    float sum = 0.0f;
    if (m < M) {
        float av = __half2float(A[m * K + col_id]);
        float xv = __half2float(x[col_id]);
        sum = av * xv;
    }
    sum = warp_reduce_sum_f32<K>(sum);      // 行内 K 个 lane 子集归约
    if (col_id == 0 && m < M)
        y[m] = __float2half(sum);
}

/*
  适用 K 为 32 的倍数：一行一个 warp，lane 沿 K 方向标量累加。
  同一 warp 的所有 lane 共享同一个 m（threadIdx.y 决定行），
  因此 if(m < M) 内整 warp 同进同出，shuffle 安全。
*/
__global__ void hgemv_k32_f16_kernel(const half* __restrict__ A,
                                     const half* __restrict__ x,
                                     half* __restrict__ y,
                                     int M, int K) {
    assert(blockDim.x == WARP_SIZE);
    assert(K % WARP_SIZE == 0);
    int lane = threadIdx.x;
    int m    = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M)
        return;
    float sum = 0.0f;
    for (int i = 0; i < K / WARP_SIZE; ++i) {
        int k = i * WARP_SIZE + lane;
        sum += __half2float(A[m * K + k]) * __half2float(x[k]);
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        y[m] = __float2half(sum);
}

/*
  适用 K 为 64 的倍数：一行一个 warp，每 lane 用 half2 一次取 2 个 half。
  一个 warp 每轮覆盖 64 个 half（32 lane * half2），循环 K/64 轮。
  K 为 128 的倍数同样满足条件，无需单独的 k128 kernel。
*/
__global__ void hgemv_k64_f16x2_kernel(const half* __restrict__ A,
                                       const half* __restrict__ x,
                                       half* __restrict__ y,
                                       int M, int K) {
    assert(blockDim.x == WARP_SIZE);
    assert(K % (2 * WARP_SIZE) == 0);       // K 为 64 的倍数
    int lane = threadIdx.x;
    int m    = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M)
        return;
    // K 为 64 的倍数 -> 行内元素数与行首偏移均为偶数，half2 4B 对齐成立
    const half2* A2 = reinterpret_cast<const half2*>(A + m * K);
    const half2* x2 = reinterpret_cast<const half2*>(x);
    float sum = 0.0f;
    for (int i = 0; i < K / (2 * WARP_SIZE); ++i) {
        int    k2 = i * WARP_SIZE + lane;   // half2 单元下标
        half2  av = A2[k2];
        half2  xv = x2[k2];
        sum += __half2float(av.x) * __half2float(xv.x) +
               __half2float(av.y) * __half2float(xv.y);
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        y[m] = __float2half(sum);
}

/* ===================== host 工具 ===================== */

/* CPU 参考：half 输入，double 累加避免顺序误差 */
static void hgemv_cpu(const half* A, const half* x, float* y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += (double)__half2float(A[m * K + k]) * __half2float(x[k]);
        y[m] = (float)acc;
    }
}

/* 最大相对误差（参考值过小处按 1e-2 保底），got 为 half 结果 */
static double max_rel_err_half(const half* got, const float* ref, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        double refv = (double)ref[i];
        double gotv = (double)__half2float(got[i]);
        double denom = fabs(refv);
        double rel   = fabs(gotv - refv) / (denom > 1e-2 ? denom : 1e-2);
        if (rel > worst)
            worst = rel;
    }
    return worst;
}

/* 有效数据量：读 A(M*K) + 读 x(K) + 写 y(M)，half 每元素 2 字节 */
static double data_bytes(int M, int K) {
    return (double)(M * K + K + M) * sizeof(half);
}

/* CPU 参考耗时：5 次平均 */
static double cpu_ms(const half* A, const half* x, float* y, int M, int K) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
        hgemv_cpu(A, x, y, M, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
}

/* kernel 计时：warmup 1 次后跑 reps 次取平均（launch 为调用点的 kernel 启动） */
template <typename Launch>
static double kernel_ms(Launch launch, int reps) {
    launch();                             // warmup
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i)
        launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / reps;
}

/* 打印一行结果：pass / ms / GB/s / %BW / GFLOPS / AI / vsCPU / rel */
static void print_row(const char* name, bool pass, double ms, double bytes,
                      double bwTheor, int M, int K, double tCpu,
                      double rel) {
    double gbps  = bytes / (ms * 1e-3) / 1e9;
    double pctBW = gbps / bwTheor * 100.0;
    double gflop = 2.0 * M * K / (ms * 1e-3) / 1e9;
    double ai    = (2.0 * M * K) / bytes;   // FLOP/byte
    printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %6.2f %7.2f  rel=%.1e\n",
           name, pass ? "PASS" : "FAIL", ms, gbps, pctBW, gflop, ai,
           tCpu / ms, rel);
}

int main()
{
    const int M     = 4096;
    const int REPS  = 50;
    const int N16   = 16;      // 场景 1：k16 适用
    const int N1024 = 1024;    // 场景 2：k32/k64 适用
    const float TOL16   = 1e-3f;   // half 输入量化 + float 累加，容差放宽
    const float TOL1024 = 1e-3f;

    // 理论带宽 = memoryClockRate(kHz) * 2(DDR) * busWidth/8 / 1e6 -> GB/s
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int memClock = 0;
    cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, 0);
    double bwTheor = 2.0 * memClock * (prop.memoryBusWidth / 8) / 1e6;
    printf("Device: %s | theoretical BW = %.1f GB/s\n", prop.name, bwTheor);
    printf("reps = %d (avg after warmup) | AI = FLOP/byte\n", REPS);

    std::mt19937 rng(42);                   // 固定种子可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    /* ========== 场景 1：K=16（k16） ========== */
    {
        const int N = N16;
        printf("\n[K=%d] k16 case (CPU / k16)\n", N);
        printf("  %-8s %-4s %8s %8s %6s%% %8s %6s %7s  %s\n",
               "name", "pass", "avg_ms", "GB/s", "BW", "GFLOPS",
               "AI", "vsCPU", "rel_err");

        // host 数据与结果
        half* hA     = new half[M * N];
        half* hx     = new half[N];
        float* hy_cpu = new float[M];
        half* hy_k16 = new half[M];
        for (int i = 0; i < M * N; ++i) hA[i] = __float2half(dist(rng));
        for (int i = 0; i < N; ++i)     hx[i] = __float2half(dist(rng));
        hgemv_cpu(hA, hx, hy_cpu, M, N);
        double tCpu = cpu_ms(hA, hx, hy_cpu, M, N);

        // device 数据
        half *dA, *dx, *dy;
        cudaMalloc(&dA, (size_t)M * N * sizeof(half));
        cudaMalloc(&dx, (size_t)N * sizeof(half));
        cudaMalloc(&dy, (size_t)M * sizeof(half));
        cudaMemcpy(dA, hA, (size_t)M * N * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, (size_t)N * sizeof(half), cudaMemcpyHostToDevice);

        // k16：每 block 处理 blockDim.y * (32/K) 行
        int rowsPerWarp = WARP_SIZE / N;      // N=16 -> 2 行/warp
        dim3 block(WARP_SIZE, 8);
        dim3 gridK16((M + 8 * rowsPerWarp - 1) / (8 * rowsPerWarp));

        double tK16 = kernel_ms([&]() {
            hgemv_k16_f16_kernel<16><<<gridK16, block>>>(dA, dx, dy, M);
        }, REPS);
        cudaMemcpy(hy_k16, dy, (size_t)M * sizeof(half), cudaMemcpyDeviceToHost);
        double relK16 = max_rel_err_half(hy_k16, hy_cpu, M);

        double bytes = data_bytes(M, N);
        print_row("k16", relK16 <= TOL16, tK16, bytes, bwTheor, M, N,
                  tCpu, relK16);
        printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %6.2f %7.2f  rel=--\n",
               "CPU", "--", tCpu, bytes / (tCpu * 1e-3) / 1e9,
               bytes / (tCpu * 1e-3) / 1e9 / bwTheor * 100.0,
               2.0 * M * N / (tCpu * 1e-3) / 1e9,
               (2.0 * M * N) / bytes, 1.0);

        cudaFree(dA);
        cudaFree(dx);
        cudaFree(dy);
        delete[] hA;
        delete[] hx;
        delete[] hy_cpu;
        delete[] hy_k16;
    }

    /* ========== 场景 2：K=1024（k32 / k64） ========== */
    {
        const int N = N1024;
        printf("\n[K=%d] k32/k64 case (CPU / k32 / k64)\n", N);
        printf("  %-8s %-4s %8s %8s %6s%% %8s %6s %7s  %s\n",
               "name", "pass", "avg_ms", "GB/s", "BW", "GFLOPS",
               "AI", "vsCPU", "rel_err");

        half* hA     = new half[M * N];
        half* hx     = new half[N];
        float* hy_cpu = new float[M];
        half* hy_k32 = new half[M];
        half* hy_k64 = new half[M];
        for (int i = 0; i < M * N; ++i) hA[i] = __float2half(dist(rng));
        for (int i = 0; i < N; ++i)     hx[i] = __float2half(dist(rng));
        hgemv_cpu(hA, hx, hy_cpu, M, N);
        double tCpu = cpu_ms(hA, hx, hy_cpu, M, N);

        half *dA, *dx, *dy;
        cudaMalloc(&dA, (size_t)M * N * sizeof(half));
        cudaMalloc(&dx, (size_t)N * sizeof(half));
        cudaMalloc(&dy, (size_t)M * sizeof(half));
        cudaMemcpy(dA, hA, (size_t)M * N * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, (size_t)N * sizeof(half), cudaMemcpyHostToDevice);

        dim3 block(WARP_SIZE, 8);             // 8 个 warp / block
        dim3 grid32((M + 7) / 8);             // 每 warp 一行

        double tK32 = kernel_ms([&]() {
            hgemv_k32_f16_kernel<<<grid32, block>>>(dA, dx, dy, M, N);
        }, REPS);
        cudaMemcpy(hy_k32, dy, (size_t)M * sizeof(half), cudaMemcpyDeviceToHost);
        double relK32 = max_rel_err_half(hy_k32, hy_cpu, M);

        double tK64 = kernel_ms([&]() {
            hgemv_k64_f16x2_kernel<<<grid32, block>>>(dA, dx, dy, M, N);
        }, REPS);
        cudaMemcpy(hy_k64, dy, (size_t)M * sizeof(half), cudaMemcpyDeviceToHost);
        double relK64 = max_rel_err_half(hy_k64, hy_cpu, M);

        double bytes = data_bytes(M, N);
        print_row("k32", relK32 <= TOL1024, tK32, bytes, bwTheor, M, N,
                  tCpu, relK32);
        print_row("k64", relK64 <= TOL1024, tK64, bytes, bwTheor, M, N,
                  tCpu, relK64);
        printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %6.2f %7.2f  rel=--\n",
               "CPU", "--", tCpu, bytes / (tCpu * 1e-3) / 1e9,
               bytes / (tCpu * 1e-3) / 1e9 / bwTheor * 100.0,
               2.0 * M * N / (tCpu * 1e-3) / 1e9,
               (2.0 * M * N) / bytes, 1.0);

        cudaFree(dA);
        cudaFree(dx);
        cudaFree(dy);
        delete[] hA;
        delete[] hx;
        delete[] hy_cpu;
        delete[] hy_k32;
        delete[] hy_k64;
    }

    return 0;
}
