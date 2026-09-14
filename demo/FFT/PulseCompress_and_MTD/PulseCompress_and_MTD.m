% ===================================================
% 脉冲多普勒雷达 RD 图仿真 (无显式fd版)
%   - 接收窗 [Tp, PRI)，一个 PRI 内回波全部接收
%   - 基带回波仅含：延时包络 + LFM相位 + 载波相位 (无fd)
%   - 速度：远离为正
%   - 设计指标:
%       最大不模糊距离  R_ua  = c/(2*PRF)         >= 10 km
%       距离分辨率      dr    = c/(2*B)           <= 5 m
%       最大不模糊速度  v_ua  = lambda*PRF/4      >= 100 m/s
%       速度分辨率      dv    = lambda*PRF/(2*M)  <= 0.5 m/s
%   - 概念区分（物理分辨率 vs 栅格）:
%       dr_bin = c/(2*fs)  < dr : 过采样 fs>B 使距离栅格比物理分辨率细，
%                                  峰位插值更准，但两目标仍须分开 dr 才能分辨
%       dv_bin = lambda*PRF/(2*N_fft) < dv : FFT 补零使多普勒栅格更密，
%                                  真实分辨率仍由 M 个真实脉冲决定 (dv)
% ===================================================
clear; close all; clc;

%% 1. 雷达系统参数
c = 3e8;
fc = 10e9;              % 载频 10 GHz
lambda = c / fc;

B  = 40e6;              % 带宽 40 MHz  -> dr = 3.75 m
Tp = 10e-6;             % 脉宽 10 us
K_chirp = B / Tp;       % 调频斜率

PRF = 14e3;             % 脉冲重复频率 14 kHz -> R_ua = 10.71 km, v_ua = 105 m/s
PRI = 1 / PRF;
M  = 512;               % 一帧脉冲数（2 的幂，决定 dv）
N_fft = 1024;           % MTD FFT 点数（补零，决定 dv_bin）

% 基带复采样率：2 倍过采样 (fs = 2B) -> dr_bin = dr/2
fs = B;
Ts = 1 / fs;

% 接收窗口 [Tp, PRI)，前闭后开
t_start = Tp;
t_end   = PRI;

% 快时间轴（接收窗采样）
N = floor((t_end - t_start) / Ts);      % 采样点数
t_fast = t_start + (0:N-1) * Ts;
range_axis = c * t_fast / 2;            % 距离轴，起始于 R_min = c*Tp/2

% 多普勒栅格（内部计算用；远离为正 -> 多普勒频率为负）
df_bin = PRF / N_fft;
doppler_axis = (-N_fft/2 : N_fft/2-1) * df_bin;
% 速度轴（绘图用，单位 m/s；远离为正）。fftshift 输出列与 doppler_axis
% 一一对应（fd 递减 -> 速度递增），绘图时翻转为升序并同步翻转数据列：
% 接近 (v<0, fd>0) 在左，远离 (v>0) 在右
speed_axis = -doppler_axis * lambda / 2;
speed_axis_plot = speed_axis(end:-1:1);

% ---- 设计指标核对（严格区分物理分辨率与栅格） ----
R_ua = c / (2*PRF);                     % 最大不模糊距离
dr   = c / (2*B);                       % 距离分辨率（物理，由 B 决定）
dr_bin = c / (2*fs);                    % 距离单元格（由 fs 决定，过采样 -> 更细）
v_ua = lambda * PRF / 4;                % 最大不模糊速度
dv   = lambda * PRF / (2*M);            % 速度分辨率（物理，由 M 个真实脉冲决定）
dv_bin = lambda * PRF / (2*N_fft);      % 速度单元格（由 FFT 补零点数决定）
fprintf('设计指标:\n');
fprintf('  R_ua = %.2f km (要求 >=10 km)\n', R_ua/1e3);
fprintf('  dr   = %.2f m (要求 <=5 m);  dr_bin = %.2f m (fs=%.0f MHz, %.1fx 过采样)\n', ...
    dr, dr_bin, fs/1e6, fs/B);
fprintf('  v_ua = %.1f m/s (要求 >=100 m/s)\n', v_ua);
fprintf('  dv   = %.3f m/s (要求 <=0.5 m/s, 由 M=%d 个真实脉冲决定)\n', dv, M);
fprintf('  dv_bin = %.3f m/s (N_fft=%d 补零栅格, 不提高真实分辨率)\n', dv_bin, N_fft);

%% 2. 目标参数 (距离 m, 径向速度 m/s: 远离为正)
% 目标速度均在 ±v_ua 内；示例速度控制跨距离单元走动 <= ~1 个 dr_bin
targets = [
    4000,  40;      % 目标1: 4km, 远离 40 m/s  -> fd = -2667 Hz
    7000,   0;      % 目标2: 7km, 静止         -> fd = 0 Hz
    9000, -50;      % 目标3: 9km, 接近 50 m/s  -> fd = +3333 Hz
];
num_targets = size(targets, 1);

%% 3. 发射基带信号与匹配滤波器
K_pulse = ceil(Tp * fs);                % 脉宽内采样点数
t_pulse = (0:K_pulse-1) * Ts;
s_pulse = exp(1j * pi * K_chirp * t_pulse.^2);
h_match = conj(fliplr(s_pulse));        % 匹配滤波器，长度 K_pulse

%% 4. 回波生成与信号处理
data_matrix = zeros(N, M);

% 固定噪声功率：回波幅度为 1 -> 单脉冲峰值 SNR = SNR_dB
SNR_dB = -30;
noise_power = 10^(-SNR_dB/10);
noise_std = sqrt(noise_power / 2);

for m = 1 : M
    rx_pulse = zeros(1, N);

    for k = 1 : num_targets
        R0 = targets(k,1);
        v  = targets(k,2);              % 远离为正

        % 瞬时距离，远离时距离增大
        R_inst = R0 + v * ((m-1) * PRI);
        tau_m = 2 * R_inst / c;

        % 有效回波区间 = 接收窗 ∩ 回波包络
        t_echo_start = max(t_start, tau_m);
        t_echo_end   = min(t_end,   tau_m + Tp);

        if t_echo_start < t_echo_end
            idx = (t_fast >= t_echo_start) & (t_fast < t_echo_end);

            % 基带回波：仅由延时 τ 决定 (无任何 fd 项)
            t_rel = t_fast(idx) - tau_m;
            echo_segment = exp(1j * pi * K_chirp * t_rel.^2);   % LFM相位
            carrier_phase = exp(-1j * 2 * pi * fc * tau_m);     % 载波常数相位

            % 多普勒效应已包含在载波相位随慢时间的变化中，无需额外乘项
            rx_pulse(idx) = rx_pulse(idx) + echo_segment .* carrier_phase;
        end
    end

    % 添加复高斯白噪声（固定功率）
    noise = noise_std * (randn(1, N) + 1j*randn(1, N));
    rx_pulse = rx_pulse + noise;

    % 脉冲压缩（频域线性卷积 + 精确时间对齐）
    L = N + K_pulse - 1;
    % h_match = conj(fliplr(s)) 已是时域匹配滤波器，直接乘其 FFT 即可；
    % 不要再取 conj(fft(...))，否则等效于拿 S(f) 滤波（失配：峰展宽+错位）。
    y_conv = ifft( fft(rx_pulse, L) .* fft(h_match, L) );
    % 截取起始索引 = K_pulse（对应时间 t_start）。对齐验证：
    % 时延 τ 的回波首个采样在 0-based 索引 n0=round((τ-t_start)/Ts)，
    % 匹配滤波峰值在 y_conv 的 n0+K_pulse-1，截取后落在 pc_pulse(n0+1)，
    % 恰对应 t_fast(n0+1) ≈ τ -> 距离轴 range_axis 直接对应目标位置。
    pc_pulse = y_conv(K_pulse : K_pulse+N-1);
    data_matrix(:, m) = pc_pulse;
end

%% 5. MTD（慢时间维 FFT，N_fft 点补零）
window = hamming(M)';
data_win = data_matrix .* window;

rd_fft = fftshift(fft(data_win, N_fft, 2), 2);
rd_map = abs(rd_fft);                   % 线性幅度（用户要求），列与 doppler_axis 对应
rd_map_plot = fliplr(rd_map);           % 翻转为与升序速度轴对应

%% 6. 绘制距离-多普勒图（左：二维，右：三维；MATLAB 默认色图 parula）
figure('Position', [100, 100, 1400, 600]);

% ---- 左：二维图 ----
subplot(1, 2, 1);
imagesc(speed_axis_plot, range_axis/1e3, rd_map_plot);
set(gca, 'YDir', 'normal');
xlabel('速度 (m/s)');
ylabel('距离 (km)');
title(sprintf('RD Map (二维, 线性幅度)  R_{ua}=%.1f km, v_{ua}=%.0f m/s', R_ua/1e3, v_ua));
colorbar;

% 固定坐标范围，防止标注越界时自动扩展把图像压缩变形
xlim([speed_axis_plot(1), speed_axis_plot(end)]);
ylim([range_axis(1), range_axis(end)] / 1e3);

% 标注目标理论位置（直接在速度轴上标注；远离为正）
hold on;
for k = 1:num_targets
    R = targets(k,1);
    v = targets(k,2);
    plot(v, R/1e3, 'wo', 'MarkerSize', 10, 'LineWidth', 2);
    text(v + 3, R/1e3 + 0.1, ...
        sprintf('R=%.1fkm, v=%.0fm/s', R/1e3, v), ...
        'Color', 'w', 'FontSize', 9);
end
hold off;

% ---- 右：三维图 ----
subplot(1, 2, 2);
surf(speed_axis_plot, range_axis/1e3, rd_map_plot, 'EdgeColor', 'none');
shading interp;
xlabel('速度 (m/s)');
ylabel('距离 (km)');
zlabel('幅度（线性）');
title('RD Map (三维, 线性幅度)');
colorbar;
xlim([speed_axis_plot(1), speed_axis_plot(end)]);
ylim([range_axis(1), range_axis(end)] / 1e3);
view(3);
