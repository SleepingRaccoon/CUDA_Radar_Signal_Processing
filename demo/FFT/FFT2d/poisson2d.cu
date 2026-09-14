/*
 * ============================================================
 * poisson2d.cu -- Solve the 2D Poisson equation by FFT (CUDA)
 *
 *   laplace(u) + f = 0     on [0, Lx] x [0, Ly], periodic
 *
 * Method (same as demo/FFT/FFT2d/poisson2d.py, ported to CUDA):
 *   X     = fft2(f)                       (aliased spectrum)
 *   U_hat = X / (kx^2 + ky^2)             frequency-domain solve
 *   u     = real(ifft2(U_hat)) / (Nx*Ny)  cuFFT inverse is NOT
 *                                         normalized, and X carries
 *                                         the extra Nx*Ny from the
 *                                         aliasing relation, so the
 *                                         result is divided by Nx*Ny
 *
 * General case: Lx, Ly, Nx, Ny are all explicit parameters
 * (no Lx = Ly = 1 special case). Bins k > N/2 map to negative
 * wavenumbers 2*pi*(k-N)/L (see poisson2d.md).
 *
 * Wavenumbers (x direction, Nx bins):
 *   kx[i] = 2*pi*i/Lx       for i <= Nx/2
 *   kx[i] = 2*pi*(i-Nx)/Lx  for i >  Nx/2
 * and the same for ky with Ny bins. The DC bin (0, 0) is pinned to
 * zero (arbitrary constant of the Poisson solution).
 *
 * Verification: source f = -laplace(u_a) for a Gaussian u_a, so the
 * exact solution is known and checked by relative L2 error (both
 * compared with their means removed, as the solution carries an
 * arbitrary constant).
 *
 * Build (from the project root, Windows):
 *   nvcc -o demo/FFT/FFT2d/poisson2d.exe demo/FFT/FFT2d/poisson2d.cu -lcufft -Xcompiler /utf-8
 * Run (redirect output next to the exe):
 *   demo\FFT\FFT2d\poisson2d.exe > demo\FFT\FFT2d\poisson2d.txt
 * ============================================================
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- problem parameters (general case, not specialized) ---- */

#define Lx 2.0f
#define Ly 1.5f
#define NX 128
#define NY 96
#define S 0.12f

#define BSZ_X 16
#define BSZ_Y 16

#define PI 3.14159265358979323846

/* ------------------------------------------------------------------
 * Error checks (replacement for NVIDIA's checkCudaErrors).
 * ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------
 * Frequency-domain Poisson solve on one GPU:
 *   U_hat[kx][ky] = X[kx][ky] / (kx^2 + ky^2)
 * The DC bin is pinned to zero (arbitrary constant / mean).
 * ------------------------------------------------------------------ */

__global__ void solvePoisson(cufftComplex *ft, cufftComplex *ft_k,
                             const float *kx, const float *ky,
                             int Nx, int Ny)
{
    int i     = blockIdx.x * blockDim.x + threadIdx.x;   /* x bin  (col) */
    int j     = blockIdx.y * blockDim.y + threadIdx.y;   /* y bin  (row) */
    int index = j * Nx + i;

    if (i < Nx && j < Ny)
    {
        if (i == 0 && j == 0)
        {
            ft_k[index].x = 0.0f;                        /* pin DC */
            ft_k[index].y = 0.0f;
        }
        else
        {
            float k2 = kx[i] * kx[i] + ky[j] * ky[j];
            ft_k[index].x = ft[index].x / k2;
            ft_k[index].y = ft[index].y / k2;
        }
    }
}

/* ------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------ */

int main(void)
{
    /* host buffers */
    cufftComplex *h_f = NULL;      /* source f on the grid       */
    float *h_u_a = NULL;           /* exact solution u_a         */
    cufftComplex *h_u_c = NULL;    /* solution (complex readback)*/
    float *kx = NULL, *ky = NULL;  /* wavenumbers                */

    /* device buffers: one variable per stage, no reuse */
    cufftComplex *d_f = NULL;      /* f, time domain             */
    cufftComplex *d_f_FFT = NULL;  /* f, frequency domain (X)    */
    cufftComplex *d_u_FFT = NULL;  /* u, frequency domain        */
    cufftComplex *d_u = NULL;      /* u, time domain             */
    float *d_kx = NULL, *d_ky = NULL;

    cufftHandle plan;
    cudaEvent_t t0 = NULL;
    cudaEvent_t t1 = NULL;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("poisson2d -- 2D Poisson equation by FFT (%s)\n", prop.name);
    printf("domain [%g] x [%g], grid %d x %d, s = %g\n", Lx, Ly, NX, NY, S);

    /* ---- 1. host data: f = -laplace(u_a) for a Gaussian u_a ---- */

    h_f  = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
    h_u_a = (float *)malloc(sizeof(float) * NX * NY);
    h_u_c = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
    kx   = (float *)malloc(sizeof(float) * NX);
    ky   = (float *)malloc(sizeof(float) * NY);

    if (!h_f || !h_u_a || !h_u_c || !kx || !ky)
    {
        printf("malloc failed\n");
        return 1;
    }

    float hx = Lx / (float)NX;
    float hy = Ly / (float)NY;
    float s2 = S * S;

    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            float x = i * hx;
            float y = j * hy;
            float r2 = (x - Lx * 0.5f) * (x - Lx * 0.5f)
                     + (y - Ly * 0.5f) * (y - Ly * 0.5f);

            h_u_a[j * NX + i] = expf(-r2 / (2.0f * s2));
            /* laplace(u_a) = (r2 - 2s^2)/s^4 * u_a, so f = -laplace(u_a) */
            h_f[j * NX + i].x = (2.0f * s2 - r2) / (s2 * s2) * expf(-r2 / (2.0f * s2));
            h_f[j * NX + i].y = 0.0f;
        }
    }

    /* ---- 2. wavenumbers (FFT bin order) ---- */

    for (int i = 0; i <= NX / 2; ++i)
    {
        kx[i] = (float)(2.0 * PI * i / Lx);
    }
    for (int i = NX / 2 + 1; i < NX; ++i)
    {
        kx[i] = (float)(2.0 * PI * (i - NX) / Lx);
    }

    for (int j = 0; j <= NY / 2; ++j)
    {
        ky[j] = (float)(2.0 * PI * j / Ly);
    }
    for (int j = NY / 2 + 1; j < NY; ++j)
    {
        ky[j] = (float)(2.0 * PI * (j - NY) / Ly);
    }

    /* ---- 3. plan and device memory ---- */

    CUFFT_CHECK(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));

    CUDA_CHECK(cudaMalloc((void **)&d_f,     sizeof(cufftComplex) * NX * NY));
    CUDA_CHECK(cudaMalloc((void **)&d_f_FFT, sizeof(cufftComplex) * NX * NY));
    CUDA_CHECK(cudaMalloc((void **)&d_u_FFT, sizeof(cufftComplex) * NX * NY));
    CUDA_CHECK(cudaMalloc((void **)&d_u,     sizeof(cufftComplex) * NX * NY));
    CUDA_CHECK(cudaMalloc((void **)&d_kx,    sizeof(float) * NX));
    CUDA_CHECK(cudaMalloc((void **)&d_ky,    sizeof(float) * NY));

    CUDA_CHECK(cudaMemcpy(d_f,  h_f,  sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kx, kx,   sizeof(float) * NX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ky, ky,   sizeof(float) * NY, cudaMemcpyHostToDevice));

    /* ---- 4. warmup (cuFFT pays plan JIT cost on first exec) ---- */

    CUFFT_CHECK(cufftExecC2C(plan, d_f, d_f_FFT, CUFFT_FORWARD));
    {
        dim3 grid(NX / BSZ_X, NY / BSZ_Y);
        dim3 block(BSZ_X, BSZ_Y);
        solvePoisson<<<grid, block>>>(d_f_FFT, d_u_FFT, d_kx, d_ky, NX, NY);
        CUDA_CHECK(cudaGetLastError());
    }
    CUFFT_CHECK(cufftExecC2C(plan, d_u_FFT, d_u, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_f, h_f, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice));

    /* ---- 5. GPU solve, timed: fft2 + freq divide + ifft2 ---- */

    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    CUFFT_CHECK(cufftExecC2C(plan, d_f, d_f_FFT, CUFFT_FORWARD));

    {
        dim3 grid(NX / BSZ_X, NY / BSZ_Y);
        dim3 block(BSZ_X, BSZ_Y);
        solvePoisson<<<grid, block>>>(d_f_FFT, d_u_FFT, d_kx, d_ky, NX, NY);
        CUDA_CHECK(cudaGetLastError());
    }

    CUFFT_CHECK(cufftExecC2C(plan, d_u_FFT, d_u, CUFFT_INVERSE));

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));

    /* ---- 6. read back, normalize by Nx*Ny (cuFFT inverse is
     *        unnormalized and X carries the aliasing factor) ---- */

    CUDA_CHECK(cudaMemcpy(h_u_c, d_u, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost));

    float inv_size = 1.0f / (float)(NX * NY);

    /* ---- 7. verify: relative L2 vs exact, means removed ---- */

    double mean_u = 0.0, mean_ua = 0.0;
    for (int i = 0; i < NX * NY; ++i)
    {
        mean_u += (double)h_u_c[i].x * inv_size;
        mean_ua += (double)h_u_a[i];
    }
    mean_u /= (double)(NX * NY);
    mean_ua /= (double)(NX * NY);

    double sum_delta2 = 0.0, sum_ref2 = 0.0;
    for (int i = 0; i < NX * NY; ++i)
    {
        double du = ((double)h_u_c[i].x * inv_size - mean_u) - ((double)h_u_a[i] - mean_ua);
        double ref = (double)h_u_a[i] - mean_ua;
        sum_delta2 += du * du;
        sum_ref2 += ref * ref;
    }

    double rel_l2 = sqrt(sum_delta2 / sum_ref2);
    double mpix_s = (double)NX * (double)NY * 1e-6 / ((double)gpu_ms * 0.001);

    double mean_f = 0.0;
    for (int i = 0; i < NX * NY; ++i)
    {
        mean_f += (double)h_f[i].x;
    }
    mean_f /= (double)(NX * NY);

    printf("cuFFT plan     : %d x %d C2C\n", NY, NX);
    printf("compatibility  : int(f) dOmega = %.3e\n", mean_f * (double)Lx * (double)Ly);
    printf("GPU solve time : %.4f ms\n", gpu_ms);
    printf("Throughput     : %.1f MPix/s\n", mpix_s);
    printf("rel L2 error   : %.3e\n", rel_l2);

    /* ---- 8. cleanup ---- */

    cudaFree(d_ky);
    cudaFree(d_kx);
    cudaFree(d_u);
    cudaFree(d_u_FFT);
    cudaFree(d_f_FFT);
    cudaFree(d_f);
    CUFFT_CHECK(cufftDestroy(plan));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    free(ky);
    free(kx);
    free(h_u_c);
    free(h_u_a);
    free(h_f);

    int ok = (rel_l2 < 1e-4f);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
