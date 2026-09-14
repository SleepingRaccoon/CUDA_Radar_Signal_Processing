/*
cuFFT_cb_kernel.cu -- cuFFT_callback 的 LTO 回调设备代码。

单独编译成 LTO-IR 后经 bin2c 转成 cuFFT_cb_fatbin.h 嵌入 host 程序。

构建（Windows，项目根目录）：
  nvcc -gencode=arch=compute_86,code=lto_86 -dc -fatbin demo\FFT\FFT1d\cuFFT_cb_kernel.cu -o output\FFT\FFT1d\cuFFT_cb_kernel.fatbin -Xcompiler /utf-8
  bin2c --name my_lto_callback_fatbin --type longlong output\FFT\FFT1d\cuFFT_cb_kernel.fatbin > demo\FFT\FFT1d\cuFFT_cb_fatbin.h
*/

#include <cufft.h>

/* 布局必须和 cuFFT_callback.cu 里的 cb_params 一致。 */
typedef struct _cb_params
{
    cufftComplex *filter;
    float         scale;
} cb_params;

/* load 回调：out = a[offset] * filter[offset] * scale */
__device__ cufftComplex ComplexPointwiseMulAndScale(void *a, unsigned long long offset, void *cb_info, void *sharedmem)
{
    cb_params *p = (cb_params *)cb_info;

    cufftComplex val    = ((cufftComplex *)a)[offset];
    cufftComplex filter = p->filter[offset];
    float        s      = p->scale;

    cufftComplex r;
    r.x = (val.x * filter.x - val.y * filter.y) * s;
    r.y = (val.x * filter.y + val.y * filter.x) * s;
    return r;
}
