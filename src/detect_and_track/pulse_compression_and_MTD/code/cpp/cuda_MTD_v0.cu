/*
脉冲压缩 + MTD（CUDA 版）

流程与 code/py/cupy_MTD_v0.py 一一对应：
补零 -> 距离维 FFT -> 乘 H -> 逆变换 -> 缩放截取 -> 加窗转置 -> 慢时间 FFT -> fftshift
输入读 NumPy 版落的 raw 与参考 drmap、计时 json，结果落 output/cpp。
计时：REPS 次取平均，用 cudaEvent；回波生成不计时。

Build: nvcc -O2 -std=c++17 -Xcompiler /utf-8
       -o src/detect_and_track/pulse_compression_and_MTD/code/cpp/cuda_MTD_v0.exe
       src/detect_and_track/pulse_compression_and_MTD/code/cpp/cuda_MTD_v0.cu -lcufft
Run:   src/detect_and_track/pulse_compression_and_MTD/code/cpp/cuda_MTD_v0.exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define PI 3.14159265358979323846
#define REPEATS 10

static const int    Nc = 256;
static const int    Nd = 256;
static const int    Nr = 1024;
static const double c = 3e8;
static const double lam = 3e8 / 10e9;
static const double Tp = 5e-6;
static const double K = (20e6) / (5e-6);
static const double fr = 15e3;
static const double fs = 1.2 * (20e6);
static const double Ts = 1.0 / (1.2 * 20e6);
static const int    Ns = (int)(5e-6 * 1.2 * 20e6);                 // 120
static const int    Nw = (int)((1.0 / 15e3 - 5e-6) * 1.2 * 20e6);  // 1480
static const int    N_fft = 2048;                                  // next pow2 >= Nw

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s (line %d)\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define CHECK_CUFFT(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    printf("cuFFT error %d (line %d)\n", (int)r_, __LINE__); exit(1); } } while (0)

// 消掉路径里的 "名字/../" 段，只为了打印好看
static void squeeze_path(char *p)
{
    for (;;) {
        char *s = strstr(p, "/../");
        if (s == NULL) s = strstr(p, "\\..\\");
        if (s == NULL) return;
        char *prev = s;
        while (prev > p && prev[-1] != '/' && prev[-1] != '\\') --prev;
        if (prev == p) return;
        memmove(prev, s + 4, strlen(s + 4) + 1);
    }
}

// 以源文件所在目录为基准拼路径，这样不用管当前工作目录
static void path_near_src(char *buf, size_t n, const char *suffix)
{
    const char *f = __FILE__;
    const char *last = f;
    for (const char *p = f; *p != 0; ++p) {
        if (*p == '/' || *p == '\\') last = p + 1;
    }
    snprintf(buf, n, "%.*s%s", (int)(last - f), f, suffix);
    for (char *q = buf; *q != 0; ++q) {
        if (*q == '\\') *q = '/';   // 统一成正斜杠，Windows 的 fopen 也认
    }
    squeeze_path(buf);
}

// 只支持小端 complex64、二维、C 连续的 .npy
static void load_npy(const char *path, void **data, long *d0, long *d1, long *data_off)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { printf("cannot open %s\n", path); exit(1); }

    char magic[6];
    unsigned char ver[2];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        printf("not a npy file: %s\n", path); exit(1);
    }
    if (fread(ver, 1, 2, f) != 2) { printf("bad npy header\n"); exit(1); }
    unsigned int hlen = 0;
    if (ver[0] == 1) {
        unsigned short h16 = 0;
        if (fread(&h16, 2, 1, f) != 1) { printf("bad npy header\n"); exit(1); }
        hlen = h16;
    } else {
        if (fread(&hlen, 4, 1, f) != 1) { printf("bad npy header\n"); exit(1); }
    }
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { printf("bad npy header\n"); exit(1); }
    hdr[hlen] = 0;

    char *d = strstr(hdr, "descr");
    if (d == NULL || strstr(d, "'<c8'") == NULL) {
        printf("only little-endian complex64 supported: %s\n", path); exit(1);
    }
    if (strstr(hdr, "fortran_order': True") != NULL) {
        printf("fortran_order not supported: %s\n", path); exit(1);
    }
    long a = 0, b = 0;
    char *shape = strchr(hdr, '(');
    if (shape == NULL || sscanf(shape + 1, "%ld, %ld", &a, &b) != 2) {
        printf("only 2-D arrays supported: %s\n", path); exit(1);
    }
    free(hdr);

    *data_off = 10 + (long)hlen;
    size_t n = (size_t)a * (size_t)b;
    void *buf = malloc(n * 8);
    if (fread(buf, 8, n, f) != n) { printf("short read: %s\n", path); exit(1); }
    fclose(f);
    *data = buf;
    *d0 = a;
    *d1 = b;
}

// 写小端 complex64 的 .npy（header 补空格到 64 字节对齐）
static void save_npy(const char *path, const void *data, long d0, long d1)
{
    char txt[256];
    int n = snprintf(txt, sizeof(txt),
                     "{'descr': '<c8', 'fortran_order': False, 'shape': (%ld, %ld), }", d0, d1);
    int total = 10 + n + 1;
    int pad = (64 - (total % 64)) % 64;
    unsigned short hlen = (unsigned short)(n + 1 + pad);

    FILE *f = fopen(path, "wb");
    if (f == NULL) { printf("cannot write %s\n", path); exit(1); }
    fwrite("\x93NUMPY", 1, 6, f);
    fputc(1, f);
    fputc(0, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(txt, 1, n, f);
    for (int i = 0; i < pad; ++i) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, 8, (size_t)d0 * d1, f);
    fclose(f);
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;
    return buf;
}

static double json_number(const char *text, const char *key)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(text, pat);
    if (p == NULL) return -1.0;
    p = strchr(p, ':');
    if (p == NULL) return -1.0;
    return strtod(p + 1, NULL);
}

static inline cufftResult cufft_exec_c2c(cufftHandle plan, float2 *in, float2 *out, int dir)
{
    // CUDA 13 起 cuFFT 去掉了 cufftExecCFFT，统一走 cufftExecC2C
    return cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out, dir);
}
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// 把 [rows][src_cols] 拷到 [rows][dst_cols]，dst 需先清零
__global__ void pad_kernel(const float2 *src, float2 *dst, long rows, int src_cols, int dst_cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * src_cols) return;
    long r = i / src_cols;
    int cc = (int)(i % src_cols);
    dst[r * dst_cols + cc] = src[i];
}

__global__ void conj_kernel(float2 *x, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i].y = -x[i].y;
}

__global__ void mul_H_kernel(float2 *x, const float2 *H, int cols, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = cmul(x[i], H[i % cols]);
}

// 逆变换后乘 1/N_fft，同时截取前 dst_cols 列
__global__ void scale_extract_kernel(const float2 *src, float2 *dst, long rows, int src_cols, int dst_cols, float scale)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dst_cols) return;
    long r = i / dst_cols;
    int cc = (int)(i % dst_cols);
    float2 v = src[r * src_cols + cc];
    dst[i] = make_float2(v.x * scale, v.y * scale);
}

// 加窗 + 转置：[Nc][Nr] -> [Nr][Nc]，dst[j*Nc + i] = pc[i*Nr + j] * w[i]
__global__ void window_transpose_kernel(const float2 *pc, const float *w, float2 *dst, int n_pulse, int n_range)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n_pulse * n_range) return;
    int j = (int)(i / n_pulse);
    int ip = (int)(i % n_pulse);
    float2 v = pc[(long)ip * n_range + j];
    float wv = w[ip];
    dst[i] = make_float2(v.x * wv, v.y * wv);
}

// 每行循环右移 cols/2，等价于 np.fft.fftshift(axis=1)
__global__ void fftshift_kernel(const float2 *src, float2 *dst, long rows, int cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    long r = i / cols;
    int cc = (int)(i % cols);
    dst[i] = src[r * cols + (cc + cols / 2) % cols];
}

static void make_hamming(float *w, int M)
{
    // 与 np.hamming(M+1)[:-1] 一致，分母取 M
    for (int n = 0; n < M; ++n) {
        w[n] = (float)(0.54 - 0.46 * cos(2.0 * PI * n / M));
    }
}

static void make_s_tx_bb(float2 *s, int n_s)
{
    // 中心在 Tp/2 的 LFM 复基带，相位用 double 算
    for (int n = 0; n < n_s; ++n) {
        double t = n * Ts - Tp / 2.0;
        double ph = PI * K * t * t;
        s[n] = make_float2((float)cos(ph), (float)sin(ph));
    }
}

int main(void)
{
    char p_raw[512], p_ref[512], p_timing[512], p_out[512];
    path_near_src(p_raw, sizeof(p_raw), "../../output/py/numpy_MTD_v0_raw.npy");
    path_near_src(p_ref, sizeof(p_ref), "../../output/py/numpy_MTD_v0_drmap.npy");
    path_near_src(p_timing, sizeof(p_timing), "../../output/py/numpy_MTD_v0_timing.json");
    path_near_src(p_out, sizeof(p_out), "../../output/cpp/cuda_MTD_v0_drmap.npy");

    printf("==============================================================\n");
    printf("pulse compression + MTD  (CUDA)\n");
    printf("==============================================================\n");
    printf("radar parameters\n");
    printf("  %-34s %d\n", "Nc (chirps)", Nc);
    printf("  %-34s %d\n", "Nw (window samples)", Nw);
    printf("  %-34s %d\n", "Nr (kept range cells)", Nr);
    printf("  %-34s %d\n", "N_fft (range FFT)", N_fft);
    printf("  %-34s %d\n", "Nd (Doppler bins)", Nd);
    printf("  %-34s %d\n", "Ns (pulse samples)", Ns);
    printf("  %-34s %s\n", "window", "hamming");
    printf("  %-34s %s\n", "dtype", "complex64");
    printf("  %-34s %s\n", "input", p_raw);
    printf("  %-34s %s\n", "reference", p_ref);

    // ---- 主机侧输入 ----
    void *h_raw = NULL, *h_ref = NULL;
    long r0 = 0, r1 = 0, g0 = 0, g1 = 0, off_raw = 0, off_ref = 0;
    load_npy(p_raw, &h_raw, &r0, &r1, &off_raw);
    load_npy(p_ref, &h_ref, &g0, &g1, &off_ref);
    if (r0 != Nc || r1 != Nw) { printf("raw shape mismatch: %ld x %ld\n", r0, r1); exit(1); }
    if (g0 != Nd || g1 != Nr) { printf("drmap shape mismatch: %ld x %ld\n", g0, g1); exit(1); }

    float2 *h_rd = (float2 *)malloc((size_t)Nr * Nd * 8);   // [Nr][Nd]，device 回传
    float2 *h_dr = (float2 *)malloc((size_t)Nd * Nr * 8);   // [Nd][Nr]，落盘
    float *h_W = (float *)malloc((size_t)Nc * sizeof(float));
    float2 *h_stx = (float2 *)malloc((size_t)N_fft * 8);
    make_hamming(h_W, Nc);
    make_s_tx_bb(h_stx, Ns);
    for (int i = Ns; i < N_fft; ++i) h_stx[i] = make_float2(0.0f, 0.0f);

    // ---- 设备侧缓冲 ----
    float2 *d_raw, *d_pad, *d_spec, *d_pc_full, *d_pc, *d_win, *d_win_pad, *d_rd, *d_shift, *d_H, *d_stx;
    float *d_W;
    CHECK_CUDA(cudaMalloc(&d_raw, (size_t)Nc * Nw * 8));
    CHECK_CUDA(cudaMalloc(&d_pad, (size_t)Nc * N_fft * 8));
    CHECK_CUDA(cudaMalloc(&d_spec, (size_t)Nc * N_fft * 8));
    CHECK_CUDA(cudaMalloc(&d_pc_full, (size_t)Nc * N_fft * 8));
    CHECK_CUDA(cudaMalloc(&d_pc, (size_t)Nc * Nr * 8));
    CHECK_CUDA(cudaMalloc(&d_win, (size_t)Nr * Nc * 8));
    CHECK_CUDA(cudaMalloc(&d_win_pad, (size_t)Nr * Nd * 8));
    CHECK_CUDA(cudaMalloc(&d_rd, (size_t)Nr * Nd * 8));
    CHECK_CUDA(cudaMalloc(&d_shift, (size_t)Nr * Nd * 8));
    CHECK_CUDA(cudaMalloc(&d_H, (size_t)N_fft * 8));
    CHECK_CUDA(cudaMalloc(&d_stx, (size_t)N_fft * 8));
    CHECK_CUDA(cudaMalloc(&d_W, (size_t)Nc * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, (size_t)Nc * sizeof(float), cudaMemcpyHostToDevice));

    // ---- H = conj(FFT(s_tx_bb, N_fft))，与 NumPy 版同口径 ----
    int n_fft_dim[1] = {N_fft};
    int n_dop_dim[1] = {Nd};
    cufftHandle plan_h, plan_range, plan_dopp;
    CHECK_CUFFT(cufftPlan1d(&plan_h, N_fft, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftPlanMany(&plan_range, 1, n_fft_dim, n_fft_dim, 1, N_fft, n_fft_dim, 1, N_fft, CUFFT_C2C, Nc));
    CHECK_CUFFT(cufftPlanMany(&plan_dopp, 1, n_dop_dim, n_dop_dim, 1, Nd, n_dop_dim, 1, Nd, CUFFT_C2C, Nr));

    CHECK_CUDA(cudaMemcpy(d_stx, h_stx, (size_t)N_fft * 8, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufft_exec_c2c(plan_h, d_stx, d_H, CUFFT_FORWARD));
    conj_kernel<<<(N_fft + 255) / 256, 256>>>(d_H, N_fft);

    // ---- 预热一次 ----
    CHECK_CUDA(cudaMemcpy(d_raw, h_raw, (size_t)Nc * Nw * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_pad, 0, (size_t)Nc * N_fft * 8));
    pad_kernel<<<(Nc * Nw + 255) / 256, 256>>>(d_raw, d_pad, Nc, Nw, N_fft);
    CHECK_CUFFT(cufft_exec_c2c(plan_range, d_pad, d_spec, CUFFT_FORWARD));
    mul_H_kernel<<<(Nc * N_fft + 255) / 256, 256>>>(d_spec, d_H, N_fft, (long)Nc * N_fft);
    CHECK_CUFFT(cufft_exec_c2c(plan_range, d_spec, d_pc_full, CUFFT_INVERSE));
    scale_extract_kernel<<<(Nc * Nr + 255) / 256, 256>>>(d_pc_full, d_pc, Nc, N_fft, Nr, 1.0f / (float)N_fft);
    window_transpose_kernel<<<(Nr * Nc + 255) / 256, 256>>>(d_pc, d_W, d_win, Nc, Nr);
    CHECK_CUDA(cudaMemset(d_win_pad, 0, (size_t)Nr * Nd * 8));
    pad_kernel<<<(Nr * Nc + 255) / 256, 256>>>(d_win, d_win_pad, Nr, Nc, Nd);
    CHECK_CUFFT(cufft_exec_c2c(plan_dopp, d_win_pad, d_rd, CUFFT_FORWARD));
    fftshift_kernel<<<(Nr * Nd + 255) / 256, 256>>>(d_rd, d_shift, Nr, Nd);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 计时：REPS 次取平均 ----
    cudaEvent_t ev[4];
    for (int i = 0; i < 4; ++i) CHECK_CUDA(cudaEventCreate(&ev[i]));
    double t_io_in = 0.0, t_calc = 0.0, t_io_out = 0.0;
    for (int rep = 0; rep < REPEATS; ++rep) {
        float ms = 0.0f;

        CHECK_CUDA(cudaEventRecord(ev[0], 0));
        {
            FILE *f = fopen(p_raw, "rb");
            if (f == NULL) { printf("cannot open %s\n", p_raw); exit(1); }
            fseek(f, off_raw, SEEK_SET);
            if (fread(h_raw, 8, (size_t)Nc * Nw, f) != (size_t)Nc * Nw) { printf("short read\n"); exit(1); }
            fclose(f);
            CHECK_CUDA(cudaMemcpy(d_raw, h_raw, (size_t)Nc * Nw * 8, cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaEventRecord(ev[1], 0));

        CHECK_CUDA(cudaMemset(d_pad, 0, (size_t)Nc * N_fft * 8));
        pad_kernel<<<(Nc * Nw + 255) / 256, 256>>>(d_raw, d_pad, Nc, Nw, N_fft);
        CHECK_CUFFT(cufft_exec_c2c(plan_range, d_pad, d_spec, CUFFT_FORWARD));
        mul_H_kernel<<<(Nc * N_fft + 255) / 256, 256>>>(d_spec, d_H, N_fft, (long)Nc * N_fft);
        CHECK_CUFFT(cufft_exec_c2c(plan_range, d_spec, d_pc_full, CUFFT_INVERSE));
        scale_extract_kernel<<<(Nc * Nr + 255) / 256, 256>>>(d_pc_full, d_pc, Nc, N_fft, Nr, 1.0f / (float)N_fft);
        window_transpose_kernel<<<(Nr * Nc + 255) / 256, 256>>>(d_pc, d_W, d_win, Nc, Nr);
        CHECK_CUDA(cudaMemset(d_win_pad, 0, (size_t)Nr * Nd * 8));
        pad_kernel<<<(Nr * Nc + 255) / 256, 256>>>(d_win, d_win_pad, Nr, Nc, Nd);
        CHECK_CUFFT(cufft_exec_c2c(plan_dopp, d_win_pad, d_rd, CUFFT_FORWARD));
        fftshift_kernel<<<(Nr * Nd + 255) / 256, 256>>>(d_rd, d_shift, Nr, Nd);
        CHECK_CUDA(cudaEventRecord(ev[2], 0));

        CHECK_CUDA(cudaMemcpy(h_rd, d_shift, (size_t)Nr * Nd * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < Nd; ++i) {
            for (int j = 0; j < Nr; ++j) {
                h_dr[(size_t)i * Nr + j] = h_rd[(size_t)j * Nd + i];   // rd -> dr 布局
            }
        }
        save_npy(p_out, h_dr, Nd, Nr);
        CHECK_CUDA(cudaEventRecord(ev[3], 0));

        CHECK_CUDA(cudaEventSynchronize(ev[3]));
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev[0], ev[1]));
        t_io_in += ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev[1], ev[2]));
        t_calc += ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev[2], ev[3]));
        t_io_out += ms;
    }
    t_io_in /= REPEATS;
    t_calc /= REPEATS;
    t_io_out /= REPEATS;
    double t_io = t_io_in + t_io_out;
    double t_total = t_io + t_calc;

    // ---- 误差（在 drmap 上比，统一 [Nd][Nr]） ----
    float *hr = (float *)h_dr;
    float *hg = (float *)h_ref;
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < (size_t)Nd * Nr * 2; ++i) {
        double d = fabs((double)hr[i] - (double)hg[i]);
        if (d > err) err = d;
        double a = fabs((double)hg[i]);
        if (a > ref) ref = a;
    }

    printf("timing  (CUDA, mean of %d runs)\n", REPEATS);
    printf("  %-34s %10.3f ms\n", "io in (read npy + host to device)", t_io_in);
    printf("  %-34s %10.3f ms\n", "compute (kernel + cufft)", t_calc);
    printf("  %-34s %10.3f ms\n", "io out (device to host + save npy)", t_io_out);
    printf("  %-34s %10.3f ms\n", "io total", t_io);
    printf("  %-34s %10.3f ms\n", "total (= io + compute)", t_total);

    char *txt = read_text(p_timing);
    if (txt != NULL) {
        double a_total = json_number(txt, "total_ms");
        double a_calc = json_number(txt, "compute_ms");
        double a_io = json_number(txt, "io_ms");
        printf("speedup vs NumPy  (numpy_MTD_v0_timing.json)\n");
        if (a_total > 0.0) printf("  %-34s %10.3f / %8.3f ms = %7.1f x\n", "total", a_total, t_total, a_total / t_total);
        if (a_calc > 0.0) printf("  %-34s %10.3f / %8.3f ms = %7.1f x\n", "compute", a_calc, t_calc, a_calc / t_calc);
        if (a_io > 0.0) printf("  %-34s %10.3f / %8.3f ms = %7.1f x\n", "io", a_io, t_io, a_io / t_io);
        free(txt);
    } else {
        printf("speedup vs NumPy  (missing numpy_MTD_v0_timing.json)\n");
    }

    printf("error on drmap [Nd][Nr]  (CUDA vs NumPy)\n");
    printf("  %-34s %.6e\n", "max absolute error", err);
    printf("  %-34s %.6e\n", "max relative error", err / ref);

    // ---- 自检：峰值位置 ----
    const double tgt[2][2] = {{2000.0, 40.0}, {4000.0, -50.0}};
    double range_cell = c / (2.0 * fs);
    double r0_axis = c / 2.0 * Tp;
    printf("self-check  (peak near each target, CUDA vs NumPy)\n");
    for (int k = 0; k < 2; ++k) {
        int lo = 0, hi = Nr - 1;
        for (int j = 0; j < Nr; ++j) {
            double rg = r0_axis + j * range_cell;
            if (rg >= tgt[k][0] - 5.0 * range_cell) { lo = j; break; }
        }
        for (int j = Nr - 1; j >= 0; --j) {
            double rg = r0_axis + j * range_cell;
            if (rg <= tgt[k][0] + 5.0 * range_cell) { hi = j; break; }
        }
        double best_g = -1.0, best_r = -1.0;
        int bi_g = lo, bj_g = 0, bi_r = lo, bj_r = 0;
        for (int i = 0; i < Nd; ++i) {
            for (int j = lo; j <= hi; ++j) {
                double a = (double)hr[((size_t)i * Nr + j) * 2] * hr[((size_t)i * Nr + j) * 2]
                         + (double)hr[((size_t)i * Nr + j) * 2 + 1] * hr[((size_t)i * Nr + j) * 2 + 1];
                if (a > best_g) { best_g = a; bi_g = i; bj_g = j; }
                double b = (double)hg[((size_t)i * Nr + j) * 2] * hg[((size_t)i * Nr + j) * 2]
                         + (double)hg[((size_t)i * Nr + j) * 2 + 1] * hg[((size_t)i * Nr + j) * 2 + 1];
                if (b > best_r) { best_r = b; bi_r = i; bj_r = j; }
            }
        }
        double vcell = lam * fr / (2.0 * Nd);
        double vg = -(lam / 2.0) * ((bi_g - Nd / 2) * fr / Nd);
        double vr = -(lam / 2.0) * ((bi_r - Nd / 2) * fr / Nd);
        printf("  target %d   NumPy R = %8.1f m v = %+7.2f m/s   |   CUDA R = %8.1f m v = %+7.2f m/s\n",
               k + 1, r0_axis + bj_r * range_cell, vr, r0_axis + bj_g * range_cell, vg);
        (void)vcell;
    }

    printf("saved: %s\n", p_out);

    free(h_raw); free(h_ref); free(h_rd); free(h_dr); free(h_W); free(h_stx);
    return 0;
}
