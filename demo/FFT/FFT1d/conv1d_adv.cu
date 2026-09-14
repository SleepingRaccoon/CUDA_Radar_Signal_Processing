/*
============================================================
conv1d_adv.cu -- 1D linear convolution via FFT, ADVANCED plan API

A teaching rewrite of the NVIDIA cuda-samples example
"simpleCUFFT" (4_CUDA_Libraries/simpleCUFFT), using the advanced
plan API: cufftCreate + cufftXtMakePlanMany.
Copyright (c) 2022, NVIDIA CORPORATION.

Build:
  nvcc -o .\output\FFT\FFT1d\conv1d_adv.exe .\demo\FFT\FFT1d\conv1d_adv.cu -lcufft -Xcompiler /utf-8
Run:
  .\output\FFT\FFT1d\conv1d_adv.exe > .\output\FFT\FFT1d\conv1d_adv.txt
============================================================
*/
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIGNAL_LEN 10000
#define KERNEL_LEN 100

#define FFT_LEN ((SIGNAL_LEN) + (KERNEL_LEN) - 1)

#define CPU_REPS 10
#define GPU_REPS 10

typedef float2 Complex;

static __device__ __host__ inline Complex Complex_Add(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

static __device__ __host__ inline Complex Complex_Mul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

static __device__ __host__ inline Complex Complex_Scale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

__global__ void Complex_Pointwise_Mul_and_Scale(const Complex *a, const Complex *b, Complex *c, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = Complex_Scale(Complex_Mul(a[idx], b[idx]), s);
}

static void cpu_conv1d(const Complex *x, int M, const Complex *h, int N, Complex *y) {
    for (int n = 0; n < M + N - 1; ++n) {
        y[n].x = 0.0f, y[n].y = 0.0f;
        for (int k = 0; k < M; ++k) {
            if (0 <= n - k && n - k < N)
                y[n] = Complex_Add(y[n], Complex_Mul(x[k], h[n - k]));
        }
    }
}

int main(void) {

    Complex *h_x = NULL;
    Complex *h_h = NULL;
    Complex *h_y1 = NULL;

    Complex *h_y2 = NULL;

    Complex *d_x = NULL;
    Complex *d_X = NULL;

    Complex *d_h = NULL;
    Complex *d_H = NULL;

    Complex *d_y = NULL;
    Complex *d_Y = NULL;

    cufftHandle plan;
    cudaEvent_t t0 = NULL;
    cudaEvent_t t1 = NULL;
    struct timespec ts_t0;
    struct timespec ts_t1;
    float scale = 1.0f / (float)FFT_LEN;

    h_x = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_h = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_y1 = (Complex *)malloc(sizeof(Complex) * FFT_LEN);
    h_y2 = (Complex *)malloc(sizeof(Complex) * FFT_LEN);

    if (!h_x || !h_h || !h_y1 || !h_y2) {
        printf("malloc failed! \n");
        return 1;
    }

    srand(42);

    for (int i = 0; i < SIGNAL_LEN; ++i) {
        h_x[i].x = (float)rand() / RAND_MAX;
        h_x[i].y = 0.0f;
    }
    for (int i = SIGNAL_LEN; i < FFT_LEN; ++i)
    {
        h_x[i].x = 0.0f;
        h_x[i].y = 0.0f;
    }

    for (int i = 0; i < KERNEL_LEN; ++i) {
        h_h[i].x = (float)rand() / RAND_MAX;
        h_h[i].y = 0.0f;
    }
    for (int i = KERNEL_LEN; i < FFT_LEN; ++i)
    {
        h_h[i].x = 0.0f;
        h_h[i].y = 0.0f;
    }

    timespec_get(&ts_t0, TIME_UTC);
    for (int rep = 0; rep < CPU_REPS; ++rep) {
        cpu_conv1d(h_x, SIGNAL_LEN, h_h, KERNEL_LEN, h_y1);
    }
    timespec_get(&ts_t1, TIME_UTC);

    double cpu_ms = ((double)(ts_t1.tv_sec - ts_t0.tv_sec) * 1e9
                   + (double)(ts_t1.tv_nsec - ts_t0.tv_nsec)) / 1e6 / CPU_REPS;

    if (cudaMalloc(&d_x, sizeof(Complex) * FFT_LEN) != cudaSuccess ||
        cudaMalloc(&d_X, sizeof(Complex) * FFT_LEN) != cudaSuccess ||
        cudaMalloc(&d_h, sizeof(Complex) * FFT_LEN) != cudaSuccess ||
        cudaMalloc(&d_H, sizeof(Complex) * FFT_LEN) != cudaSuccess ||
        cudaMalloc(&d_y, sizeof(Complex) * FFT_LEN) != cudaSuccess ||
        cudaMalloc(&d_Y, sizeof(Complex) * FFT_LEN) != cudaSuccess)
    {
        printf("cudaMalloc failed! \n");
        return 1;
    }

    if (cudaMemcpy(d_x, h_x, sizeof(Complex) * FFT_LEN, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_h, h_h, sizeof(Complex) * FFT_LEN, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("cudaMemcpy H2D failed! \n");
        return 1;
    }

    /* ---- advanced plan API: cufftCreate + cufftXtMakePlanMany ---- */
    long long int fft_len = FFT_LEN;
    size_t work_size = 0;

    if (cufftCreate(&plan) != CUFFT_SUCCESS) {
        printf("cufftCreate failed! \n");
        return 1;
    }

    /* 1D, dims = {FFT_LEN}, contiguous, batch = 1 */
    if (cufftXtMakePlanMany(plan, 1, &fft_len,
                            NULL, 1, 1, CUDA_C_32F,
                            NULL, 1, 1, CUDA_C_32F,
                            1, &work_size, CUDA_C_32F) != CUFFT_SUCCESS) {
        printf("cufftXtMakePlanMany failed! \n");
        return 1;
    }
    printf("cuFFT advanced plan workspace size: %zu bytes\n", work_size);

    if (cudaEventCreate(&t0) != cudaSuccess || cudaEventCreate(&t1) != cudaSuccess) {
        printf("cudaEventCreate failed! \n");
        return 1;
    }
    cudaEventRecord(t0);

    for (int rep = 0; rep < GPU_REPS; ++rep) {

        cufftResult r_x = cufftExecC2C(plan, d_x, d_X, CUFFT_FORWARD);
        cufftResult r_h = cufftExecC2C(plan, d_h, d_H, CUFFT_FORWARD);

        Complex_Pointwise_Mul_and_Scale <<<(FFT_LEN + 255) / 256, 256>>> (d_X, d_H, d_Y, FFT_LEN, scale);
        cudaError_t err_kern = cudaGetLastError();

        cufftResult r_y = cufftExecC2C(plan, d_Y, d_y, CUFFT_INVERSE);

        if (r_x != CUFFT_SUCCESS || r_h != CUFFT_SUCCESS || r_y != CUFFT_SUCCESS ||
            err_kern != cudaSuccess)
        {
            printf("GPU pipeline error at rep %d: cufft=%d/%d/%d, kernel=%s\n",
                   rep, (int)r_x, (int)r_h, (int)r_y, cudaGetErrorString(err_kern));
            break;
        }
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float gpu_ms_total = 0.0f;
    cudaEventElapsedTime(&gpu_ms_total, t0, t1);

    double gpu_ms = gpu_ms_total / GPU_REPS;

    if (cudaMemcpy(h_y2, d_y, sizeof(Complex) * FFT_LEN, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("cudaMemcpy D2H failed! \n");
        return 1;
    }

    double err = 0.0;
    double ref = 0.0;

    for (int i = 0; i < FFT_LEN; ++i) {
        double dx = h_y2[i].x - h_y1[i].x;
        double dy = h_y2[i].y - h_y1[i].y;
        err += dx * dx + dy * dy;
        ref += h_y1[i].x * h_y1[i].x + h_y1[i].y * h_y1[i].y;
    }

    double rel = sqrt(err / (ref + 1e-30));

    double speedup = cpu_ms / gpu_ms;
    char buf[64];

    printf("\n");
    printf("+-----------------------+----------------------+\n");
    printf("| %-21s | %-20s |\n", "Metric", "Value");
    printf("+-----------------------+----------------------+\n");
    printf("| %-21s | %-20d |\n", "FFT length M+N-1", FFT_LEN);
    printf("| %-21s | %-20.3e |\n", "Relative error (GPU)", rel);
    snprintf(buf, sizeof(buf), "%.4f ms", cpu_ms);
    printf("| %-21s | %-20s |\n", "CPU conv time (avg)", buf);
    snprintf(buf, sizeof(buf), "%.4f ms", gpu_ms);
    printf("| %-21s | %-20s |\n", "GPU conv time (avg)", buf);
    snprintf(buf, sizeof(buf), "%.2f x", speedup);
    printf("| %-21s | %-20s |\n", "Speedup (CPU / GPU)", buf);
    printf("+-----------------------+----------------------+\n\n");

    printf("(CPU time = full direct convolution, avg of %d runs)\n", CPU_REPS);
    printf("(GPU time = 2 FFTs + pointwise multiply + IFFT, avg of %d runs)\n\n", GPU_REPS);

    int start = (KERNEL_LEN - 1) / 2;
    printf("same (center-aligned) convolution = y[%d..%d]\n", start, start + SIGNAL_LEN - 1);

    int ok = (rel < 1e-5f);

    cufftDestroy(plan);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_x);
    cudaFree(d_X);
    cudaFree(d_h);
    cudaFree(d_H);
    cudaFree(d_y);
    cudaFree(d_Y);

    free(h_x);
    free(h_h);
    free(h_y1);
    free(h_y2);

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
