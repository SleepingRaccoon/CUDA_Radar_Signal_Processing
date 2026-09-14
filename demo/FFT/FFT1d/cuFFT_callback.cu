/*
============================================================
cuFFT_callback.cu -- 用 cuFFT 回调做一维 FFT 卷积（补零到 10100）

M+N-1 = 10099 是质数，cuFFT 回调不支持质数尺寸（Bluestein 路径），
故 FFT_LEN 取 10100。

改编自 NVIDIA cuda-samples "simpleCUFFT_callback"
（4_CUDA_Libraries/simpleCUFFT_callback），回调走 CUDA 13 LTO 路线。
Copyright (c) 2022, NVIDIA CORPORATION.

Build && Run:
  nvcc -o output\FFT\FFT1d\cuFFT_callback.exe demo\FFT\FFT1d\cuFFT_callback.cu -lcufft -Xcompiler /utf-8
  output\FFT\FFT1d\cuFFT_callback.exe > output\FFT\FFT1d\cuFFT_callback.txt
============================================================
*/

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "cuFFT_cb_fatbin.h"   /* LTO 回调 fatbin（与源文件同目录） */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   /* C11 timespec_get（CPU 参考计时） */

#define SIGNAL_SIZE        10000
#define FILTER_KERNEL_SIZE 100
#define FULL_LEN           (SIGNAL_SIZE + FILTER_KERNEL_SIZE - 1)  /* 10099：线性卷积长度 */
#define FFT_LEN            (FULL_LEN + 1)  /* 10100：10099 是质数，callback 不支持质数尺寸 */

#define CPU_REPS 50
#define GPU_REPS 50

typedef float2 Complex;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t cuda_err = (call);                                       \
        if (cuda_err != cudaSuccess) {                                       \
            printf("CUDA error: %s (%s)\n", cudaGetErrorString(cuda_err),    \
                   #call);                                                   \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(call)                                                    \
    do {                                                                     \
        cufftResult cufft_err = (call);                                      \
        if (cufft_err != CUFFT_SUCCESS) {                                    \
            printf("cuFFT error: code %d (%s)\n", (int)cufft_err, #call);    \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

/* ------------------------------------------------------------------
cuFFT 回调：设备函数在 cuFFT_cb_kernel.cu（经 cuFFT_cb_fatbin.h
嵌入）；cb_params 布局与它一致。
------------------------------------------------------------------ */

typedef struct _cb_params
{
    Complex *filter;
    float    scale;
} cb_params;

/* CPU 参考：完整线性卷积，输出长度 M + N - 1。 */

static void cpu_conv1d(const Complex *x, int M, const Complex *h, int N, Complex *y)
{
    for (int n = 0; n < M + N - 1; ++n)
    {
        y[n].x = 0.0f, y[n].y = 0.0f;
        for (int k = 0; k < M; ++k)
        {
            if (0 <= n - k && n - k < N)
            {
                y[n] = ComplexAdd(y[n], ComplexMul(x[k], h[n - k]));
            }
        }
    }
}

/* ------------------------------------------------------------------
main
------------------------------------------------------------------ */

int main(void)
{
    Complex *h_x = NULL;
    Complex *h_h = NULL;
    Complex *h_y1 = NULL;
    Complex *h_y2 = NULL;

    Complex *d_x = NULL;
    Complex *d_X = NULL;
    Complex *d_h = NULL;
    Complex *d_H = NULL;
    Complex *d_y = NULL;
    Complex *d_params = NULL;

    cufftHandle plan, cb_plan;
    size_t      work_size = 0;
    cudaEvent_t t0 = NULL;
    cudaEvent_t t1 = NULL;
    struct timespec ts_t0;
    struct timespec ts_t1;
    float scale = 1.0f / (float)FFT_LEN;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("cuFFT_callback -- 1D convolution with cuFFT load callback (%s)\n", prop.name);
    printf("signal %d samples, filter %d taps\n\n", SIGNAL_SIZE, FILTER_KERNEL_SIZE);

    /* ---- host 数据：直接分配 FFT_LEN，前段随机、后段补零 ---- */

    h_x = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_h = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_y1 = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_y2 = (Complex *)malloc(sizeof(Complex) * FFT_LEN);

    if (!h_x || !h_h || !h_y1 || !h_y2)
    {
        printf("malloc failed\n");
        return 1;
    }

    for (int i = 0; i < SIGNAL_SIZE; ++i)
    {
        h_x[i].x = rand() / (float)RAND_MAX;
        h_x[i].y = 0.0f;
    }
    for (int i = SIGNAL_SIZE; i < FFT_LEN; ++i)
    {
        h_x[i].x = 0.0f;
        h_x[i].y = 0.0f;
    }

    for (int i = 0; i < FILTER_KERNEL_SIZE; ++i)
    {
        h_h[i].x = rand() / (float)RAND_MAX;
        h_h[i].y = 0.0f;
    }
    for (int i = FILTER_KERNEL_SIZE; i < FFT_LEN; ++i)
    {
        h_h[i].x = 0.0f;
        h_h[i].y = 0.0f;
    }

    /* ---- CPU 参考，计时（多次取平均） ---- */

    timespec_get(&ts_t0, TIME_UTC);
    for (int rep = 0; rep < CPU_REPS; ++rep)
    {
        cpu_conv1d(h_x, SIGNAL_SIZE, h_h, FILTER_KERNEL_SIZE, h_y1);
    }
    timespec_get(&ts_t1, TIME_UTC);

    double cpu_ms = ((double)(ts_t1.tv_sec - ts_t0.tv_sec) * 1e9
                   + (double)(ts_t1.tv_nsec - ts_t0.tv_nsec)) / 1e6 / CPU_REPS;

    /* ---- device 内存 ---- */

    CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(Complex) * FFT_LEN));
    CUDA_CHECK(cudaMalloc((void **)&d_X, sizeof(Complex) * FFT_LEN));
    CUDA_CHECK(cudaMalloc((void **)&d_h, sizeof(Complex) * FFT_LEN));
    CUDA_CHECK(cudaMalloc((void **)&d_H, sizeof(Complex) * FFT_LEN));
    CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(Complex) * FFT_LEN));
    CUDA_CHECK(cudaMalloc((void **)&d_params, sizeof(cb_params)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(Complex) * FFT_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, h_h, sizeof(Complex) * FFT_LEN, cudaMemcpyHostToDevice));

    /* ---- plan：普通前向 + 带回调的逆变换 ---- */

    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftCreate(&cb_plan));

    cb_params h_params;
    h_params.filter = d_H;
    h_params.scale  = scale;

    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(cb_params), cudaMemcpyHostToDevice));

    /* LTO 回调挂到逆变换 plan（必须在 cufftMakePlan* 之前） */
    CUFFT_CHECK(cufftXtSetJITCallback(cb_plan, "ComplexPointwiseMulAndScale",
                                      my_lto_callback_fatbin, sizeof(my_lto_callback_fatbin),
                                      CUFFT_CB_LD_COMPLEX, (void **)&d_params));

    CUFFT_CHECK(cufftMakePlan1d(plan, FFT_LEN, CUFFT_C2C, 1, &work_size));
    CUFFT_CHECK(cufftMakePlan1d(cb_plan, FFT_LEN, CUFFT_C2C, 1, &work_size));

    /* warmup：回调首次执行有 JIT 开销，先跑一遍再计时 */
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_x, (cufftComplex *)d_X, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_h, (cufftComplex *)d_H, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(cb_plan, (cufftComplex *)d_X, (cufftComplex *)d_y, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(Complex) * FFT_LEN, cudaMemcpyHostToDevice));

    /* ---- GPU 卷积，计时（多次取平均） ---- */

    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int rep = 0; rep < GPU_REPS; ++rep)
    {
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_x, (cufftComplex *)d_X, CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)d_h, (cufftComplex *)d_H, CUFFT_FORWARD));
        /* 逆变换读 d_X 时回调做 X*H*scale，结果写 d_y */
        CUFFT_CHECK(cufftExecC2C(cb_plan, (cufftComplex *)d_X, (cufftComplex *)d_y, CUFFT_INVERSE));
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gpu_ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_total, t0, t1));
    double gpu_ms = gpu_ms_total / GPU_REPS;

    /* ---- 读回并验证 ---- */

    CUDA_CHECK(cudaMemcpy(h_y2, d_y, sizeof(Complex) * FFT_LEN, cudaMemcpyDeviceToHost));

    double err = 0.0;
    double ref = 0.0;
    for (int i = 0; i < FULL_LEN; ++i)
    {
        double dx = (double)h_y2[i].x - (double)h_y1[i].x;
        double dy = (double)h_y2[i].y - (double)h_y1[i].y;
        err += dx * dx + dy * dy;
        ref += (double)h_y1[i].x * (double)h_y1[i].x + (double)h_y1[i].y * (double)h_y1[i].y;
    }
    double rel = sqrt(err / (ref + 1e-30));

    double speedup = cpu_ms / gpu_ms;
    char buf[64];

    printf("\n");
    printf("+-----------------------+----------------------+\n");
    printf("| %-21s | %-20s |\n", "Metric", "Value");
    printf("+-----------------------+----------------------+\n");
    printf("| %-21s | %-20d |\n", "FFT length", FFT_LEN);
    printf("| %-21s | %-20.3e |\n", "Relative error (GPU)", rel);
    snprintf(buf, sizeof(buf), "%.4f ms", cpu_ms);
    printf("| %-21s | %-20s |\n", "CPU conv time (avg)", buf);
    snprintf(buf, sizeof(buf), "%.4f ms", gpu_ms);
    printf("| %-21s | %-20s |\n", "GPU conv time (avg)", buf);
    snprintf(buf, sizeof(buf), "%.2f x", speedup);
    printf("| %-21s | %-20s |\n", "Speedup (CPU / GPU)", buf);
    printf("+-----------------------+----------------------+\n\n");

    printf("(CPU time = full direct convolution, avg of %d runs)\n", CPU_REPS);
    printf("(GPU time = 2 FFTs + inverse FFT with callback, avg of %d runs)\n\n", GPU_REPS);

    /* ---- 清理 ---- */

    CUFFT_CHECK(cufftDestroy(cb_plan));
    CUFFT_CHECK(cufftDestroy(plan));
    cudaFree(d_params);
    cudaFree(d_y);
    cudaFree(d_H);
    cudaFree(d_h);
    cudaFree(d_X);
    cudaFree(d_x);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    free(h_y2);
    free(h_y1);
    free(h_h);
    free(h_x);

    int ok = (rel < 1e-5f);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
