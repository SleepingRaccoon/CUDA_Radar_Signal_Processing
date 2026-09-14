/*
============================================================
sgemv.cu -- 行主序 SGEMV：y[m] = sum_k A[m][k] * x[k]

三个 kernel 的适用条件（K = 列数）：
  k16  : K 是 16 的因数（K<=16 且为 2 的幂），一个 warp 拆成多行并行；
  k32  : K 是 32 的倍数，一行分配一个 warp，lane 沿 K 方向规约；
  k128 : K 是 128 的倍数，一行分配一个 warp，每 lane 用 float4 一次取 4 元素。

主程序对 K=16 与 K=1024 两组场景做 CPU / cuBLAS / 自写 kernel 对比：
  K=16   -> k16
  K=1024 -> k32 与 k128 同条件对比（1024 同时是 32、128 的倍数）

指标：正确性(最大相对误差) / 平均耗时 / 有效带宽 / 带宽利用率 /
      GFLOPS / vsCPU / vsBLAS

Build:
  nvcc -o demo\kernels\sgemv\sgemv.exe demo\kernels\sgemv\sgemv.cu -lcublas -Xcompiler /utf-8
Run:
  .\demo\kernels\sgemv\sgemv.exe
============================================================
*/

#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

/* 一个 warp 内若干线程的子集规约（NUM_THREADS<=32 且为 2 的幂） */
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

/* 适用 K 为 32 的倍数：一行一个 warp */
__global__ void sgemv_k32_f32_kernel(const float* __restrict__ A,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int M, int K) {
    assert(K % WARP_SIZE == 0);
    assert(blockDim.x == WARP_SIZE);
    int lane = threadIdx.x;
    int m    = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M)
        return;
    float sum = 0.0f;
    for (int i = 0; i < K / WARP_SIZE; ++i) {
        int k = i * WARP_SIZE + lane;
        sum += A[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        y[m] = sum;
}

/* 适用 K 为 128 的倍数：一行一个 warp，float4 向量化访存 */
__global__ void sgemv_k128_f32x4_kernel(const float* __restrict__ A,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int M, int K) {
    assert(K % (4 * WARP_SIZE) == 0);
    assert(blockDim.x == WARP_SIZE);
    int lane = threadIdx.x;
    int m    = blockIdx.x * blockDim.y + threadIdx.y;
    if (m >= M)
        return;
    float sum = 0.0f;
    // K % 128 == 0 保证每行首地址 16B 对齐，float4 访问合法
    const float4* A4 = reinterpret_cast<const float4*>(A + m * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = 0; i < K / (4 * WARP_SIZE); ++i) {
        int    k4 = i * WARP_SIZE + lane;      // float4 单元下标
        float4 av = A4[k4];
        float4 xv = x4[k4];
        sum += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0)
        y[m] = sum;
}

/*
  适用 K 为 16 的因数：一个 warp 同时处理 WARP_SIZE/K 行，
  行内 K 个 lane 各自乘一个元素后用 __shfl_xor 做子集规约。
*/
template <int K>
__global__ void sgemv_k16_f32_kernel(const float* __restrict__ A,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int M) {
    static_assert(K <= 16 && ((K & (K - 1)) == 0),
                  "K must be power of 2 <= 16");
    assert(blockDim.x == WARP_SIZE);
    int lane   = threadIdx.x;
    int rows   = WARP_SIZE / K;               // 一个 warp 处理的行数
    int row_id = lane / K;                    // 本 lane 属于第几行
    int col_id = lane % K;                    // 本 lane 负责第几个元素
    int m = rows * (blockIdx.x * blockDim.y + threadIdx.y) + row_id;
    if (m >= M)
        return;
    float sum = A[m * K + col_id] * x[col_id];
    sum = warp_reduce_sum_f32<K>(sum);        // 只在行内的 K 个 lane 间规约
    if (col_id == 0)
        y[m] = sum;
}

/* ===================== host 工具 ===================== */

/* CPU 参考：double 累加避免顺序误差 */
static void sgemv_cpu(const float* A, const float* x, float* y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += (double)A[m * K + k] * x[k];
        y[m] = (float)acc;
    }
}

/* 最大相对误差（参考值过小处按 1e-2 保底） */
static double max_rel_err(const float* got, const float* ref, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        double denom = fabs((double)ref[i]);
        double rel   = fabs((double)got[i] - ref[i]) / (denom > 1e-2 ? denom : 1e-2);
        if (rel > worst)
            worst = rel;
    }
    return worst;
}

/* 有效数据量：读 A(M*K) + 读 x(K) + 写 y(M) */
static double data_bytes(int M, int K) {
    return (double)(M * K + K + M) * sizeof(float);
}

/* CPU 参考耗时：5 次平均 */
static double cpu_ms(const float* A, const float* x, float* y, int M, int K) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
        sgemv_cpu(A, x, y, M, K);
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

/* 打印一行结果：pass / ms / GB/s / %BW / GFLOPS / vsCPU / vsBLAS / rel */
static void print_row(const char* name, bool pass, double ms, double bytes,
                      double bwTheor, int M, int K, double tCpu,
                      double tBlas, double rel) {
    double gbps  = bytes / (ms * 1e-3) / 1e9;
    double pctBW = gbps / bwTheor * 100.0;
    double gflop = 2.0 * M * K / (ms * 1e-3) / 1e9;
    printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %7.2f %7.2f  rel=%.1e\n",
           name, pass ? "PASS" : "FAIL", ms, gbps, pctBW, gflop,
           tCpu / ms, tBlas / ms, rel);
}

int main()
{
    const int M     = 4096;
    const int REPS  = 50;
    const int N16   = 16;      // 场景 1：k16 适用
    const int N1024 = 1024;    // 场景 2：k32/k128 适用
    const float TOL16   = 1e-4f;
    const float TOL1024 = 1e-3f;

    // 理论带宽 = memoryClockRate(kHz) * 2(DDR) * busWidth/8 / 1e6 -> GB/s
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int memClock = 0;
    cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, 0);
    double bwTheor = 2.0 * memClock * (prop.memoryBusWidth / 8) / 1e6;
    printf("Device: %s | theoretical BW = %.1f GB/s\n", prop.name, bwTheor);
    printf("reps = %d (avg after warmup)\n", REPS);

    std::mt19937 rng(42);                   // 固定种子可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cublasHandle_t hBlas;
    cublasCreate(&hBlas);
    const float alpha = 1.0f, beta = 0.0f;

    /* ========== 场景 1：K=16（k16） ========== */
    {
        const int N = N16;
        printf("\n[K=%d] k16 case (CPU / cuBLAS / k16)\n", N);
        printf("  %-8s %-4s %8s %8s %6s%% %8s %7s %7s  %s\n",
               "name", "pass", "avg_ms", "GB/s", "BW",
               "GFLOPS", "vsCPU", "vsBLAS", "rel_err");

        // host 数据与结果
        float* hA     = new float[M * N];
        float* hx     = new float[N];
        float* hy_cpu = new float[M];
        float* hy_cublas = new float[M];
        float* hy_k16 = new float[M];
        for (int i = 0; i < M * N; ++i) hA[i] = dist(rng);
        for (int i = 0; i < N; ++i)     hx[i] = dist(rng);
        sgemv_cpu(hA, hx, hy_cpu, M, N);
        double tCpu = cpu_ms(hA, hx, hy_cpu, M, N);

        // device 数据
        float *dA, *dx, *dy;
        cudaMalloc(&dA, (size_t)M * N * sizeof(float));
        cudaMalloc(&dx, (size_t)N * sizeof(float));
        cudaMalloc(&dy, (size_t)M * sizeof(float));
        cudaMemcpy(dA, hA, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

        // cuBLAS：行主序 A 等价列主序 A^T，用 CUBLAS_OP_T 计算 A*x
        double tBlas = kernel_ms([&]() {
            cublasSgemv(hBlas, CUBLAS_OP_T, N, M, &alpha, dA, N,
                        dx, 1, &beta, dy, 1);
        }, REPS);
        cudaMemcpy(hy_cublas, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
        double relBlas = max_rel_err(hy_cublas, hy_cpu, M);

        // k16：每 block 处理 blockDim.y * (32/K) 行
        int rowsPerWarp = WARP_SIZE / N;      // N=16 -> 2 行/warp
        dim3 block(WARP_SIZE, 8);
        dim3 gridK16((M + 8 * rowsPerWarp - 1) / (8 * rowsPerWarp));

        double tK16 = kernel_ms([&]() {
            sgemv_k16_f32_kernel<16><<<gridK16, block>>>(dA, dx, dy, M);
        }, REPS);
        cudaMemcpy(hy_k16, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
        double relK16 = max_rel_err(hy_k16, hy_cpu, M);

        double bytes = data_bytes(M, N);
        print_row("k16", relK16 <= TOL16, tK16, bytes, bwTheor, M, N,
                  tCpu, tBlas, relK16);
        print_row("cuBLAS", relBlas <= TOL16, tBlas, bytes, bwTheor, M, N,
                  tCpu, tBlas, relBlas);
        printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %7.2f %7.2f  rel=--\n",
               "CPU", "--", tCpu, bytes / (tCpu * 1e-3) / 1e9,
               bytes / (tCpu * 1e-3) / 1e9 / bwTheor * 100.0,
               2.0 * M * N / (tCpu * 1e-3) / 1e9, 1.0, 1.0);

        cudaFree(dA);
        cudaFree(dx);
        cudaFree(dy);
        delete[] hA;
        delete[] hx;
        delete[] hy_cpu;
        delete[] hy_cublas;
        delete[] hy_k16;
    }

    /* ========== 场景 2：K=1024（k32 / k128） ========== */
    {
        const int N = N1024;
        printf("\n[K=%d] k32/k128 case (CPU / cuBLAS / k32 / k128)\n", N);
        printf("  %-8s %-4s %8s %8s %6s%% %8s %7s %7s  %s\n",
               "name", "pass", "avg_ms", "GB/s", "BW",
               "GFLOPS", "vsCPU", "vsBLAS", "rel_err");

        float* hA     = new float[M * N];
        float* hx     = new float[N];
        float* hy_cpu = new float[M];
        float* hy_cublas = new float[M];
        float* hy_k32 = new float[M];
        float* hy_k128 = new float[M];
        for (int i = 0; i < M * N; ++i) hA[i] = dist(rng);
        for (int i = 0; i < N; ++i)     hx[i] = dist(rng);
        sgemv_cpu(hA, hx, hy_cpu, M, N);
        double tCpu = cpu_ms(hA, hx, hy_cpu, M, N);

        float *dA, *dx, *dy;
        cudaMalloc(&dA, (size_t)M * N * sizeof(float));
        cudaMalloc(&dx, (size_t)N * sizeof(float));
        cudaMalloc(&dy, (size_t)M * sizeof(float));
        cudaMemcpy(dA, hA, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

        double tBlas = kernel_ms([&]() {
            cublasSgemv(hBlas, CUBLAS_OP_T, N, M, &alpha, dA, N,
                        dx, 1, &beta, dy, 1);
        }, REPS);
        cudaMemcpy(hy_cublas, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
        double relBlas = max_rel_err(hy_cublas, hy_cpu, M);

        dim3 block(WARP_SIZE, 8);             // 8 个 warp / block
        dim3 grid32((M + 7) / 8);             // 每 warp 一行

        double tK32 = kernel_ms([&]() {
            sgemv_k32_f32_kernel<<<grid32, block>>>(dA, dx, dy, M, N);
        }, REPS);
        cudaMemcpy(hy_k32, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
        double relK32 = max_rel_err(hy_k32, hy_cpu, M);

        double tK128 = kernel_ms([&]() {
            sgemv_k128_f32x4_kernel<<<grid32, block>>>(dA, dx, dy, M, N);
        }, REPS);
        cudaMemcpy(hy_k128, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);
        double relK128 = max_rel_err(hy_k128, hy_cpu, M);

        double bytes = data_bytes(M, N);
        print_row("k32", relK32 <= TOL1024, tK32, bytes, bwTheor, M, N,
                  tCpu, tBlas, relK32);
        print_row("k128", relK128 <= TOL1024, tK128, bytes, bwTheor, M, N,
                  tCpu, tBlas, relK128);
        print_row("cuBLAS", relBlas <= TOL1024, tBlas, bytes, bwTheor, M, N,
                  tCpu, tBlas, relBlas);
        printf("  %-8s %-4s %8.4f %8.1f %6.1f%% %8.1f %7.2f %7.2f  rel=--\n",
               "CPU", "--", tCpu, bytes / (tCpu * 1e-3) / 1e9,
               bytes / (tCpu * 1e-3) / 1e9 / bwTheor * 100.0,
               2.0 * M * N / (tCpu * 1e-3) / 1e9, 1.0, 1.0);

        cudaFree(dA);
        cudaFree(dx);
        cudaFree(dy);
        delete[] hA;
        delete[] hx;
        delete[] hy_cpu;
        delete[] hy_cublas;
        delete[] hy_k32;
        delete[] hy_k128;
    }

    cublasDestroy(hBlas);
    return 0;
}
