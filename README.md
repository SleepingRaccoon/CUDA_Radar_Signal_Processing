# CUDA雷达信号处理

01 kernels：重要内核，sgemm，sgemv，transpose，reduce，以及未来有关AI Infra的attention，tensor，wwm，cute，cutlass等。这部分是cuda的基础。

02 nvidia samples，重要官方样例，NVIDIA官方仓库提供了诸多的样例，有些例子还是挺重要的，但是NVIDIA的例子中很多写的过于繁琐，涉及很多无关逻辑的命令行解析等，我们会把重要的例子重写，单个cu文件编译运行的那种，这部分也是cuda的基础。

03 DBF：数字波束成形。

04 detect：雷达信号处理中的检测问题。

05 track：雷达信号处理中的跟踪问题，Kalman，particle filter，RFS等，因为跟踪大部分可能是串行算法，因此我们可能会研究CPU的加速技术作为GPU加速技术的互补。

06 SAR：成像算法，RDA，CSA，Omega-K，SPECAN等。

07 rsp_pipeline：GPU，CPU的异构流水线。