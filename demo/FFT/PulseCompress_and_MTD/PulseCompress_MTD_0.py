"""
PulseCompress_and_MTD.py -- RD map simulation (pulse compression + MTD)

Python reference for the C implementation. Indexing is 0-based so the
two languages can be cross-checked line by line.

Array layout (row-major; axis 0 = slow time v, axis 1 = fast time r):
    raw_matrix (Nc, Ns) : raw collected echo matrix, before pulse compression
    rd_map     (Nd, Nr) : visualization matrix after pulse compression,
                          range truncation and slow-time FFT
    A chirp (one received pulse) occupies one row: writing the echo and
    the fast-time (range) pulse-compression FFT are both contiguous.

Dimensions:
    Ns : fast-time samples collected in the receive window
    Nr : range cells kept for visualization (hard-coded 1024, truncation)
    Nc : chirps per frame (power of 2, sets dv)
    Nd : Doppler bins after the slow-time FFT (Nd = Nc here)
    Np : samples per pulse, Np = floor(Tp*fs)

Resolution vs grid:
    dr     = c/(2B)           physical range resolution   (set by B)
    dr_bin = c/(2fs)          range cell size             (set by fs)
    dv     = lam*PRF/(2Nc)    physical velocity resolution (set by Nc real chirps)
    dv_bin = lam*PRF/(2Nd)    velocity grid step           (set by Nd FFT bins)
    fs = alpha_os * B; complex baseband only needs alpha_os >= 1.
    R_ua = c*(PRI-Tp)/2       the max delay for FULL reception is PRI - Tp
                              (the echo occupies [tau, tau+Tp) inside [Tp, PRI))

Signal model:
    Each target is [range m, velocity m/s, amplitude dB, phase rad].
    The noise is complex white Gaussian with unit total variance, so the
    signal amplitude is derived from dB: amp = 10^(dB/20) and the
    per-sample SNR of that target is amp^2.

Pulse compression:
    Standard matched filter in the frequency domain:
        y = IFFT( FFT(rx) .* conj(FFT(s_pulse)) )
    The peak of y sits at the delay index n0, so the first Ns samples of
    y map 1:1 onto the range axis (element 0 = t_start).

Note:
    This file stops at pulse compression + MTD: it produces the raw
    data matrix (0_raw.npy, Ns x Nc) and the truncated RD map
    (0_trunc.npy, Nr x Nd), plus a 2D figure (0_trunc_2d.png) and an
    interactive 3D view (0_trunc_3d.html). Target detection (CFAR)
    and clustering belong to the demo/CFAR folder, which consumes
    0_trunc.npy. The circles on the 2D plot are ground-truth targets
    for reference only.

Plot:
    2D RD map with matplotlib; axes are drawn as-is, no flipping.
    3D surface is exported to an interactive HTML via plotly.

matplotlib has no built-in parula colormap, so 'viridis' is used.

Run: python3 PulseCompress_MTD_0.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# 1. radar system parameters (round numbers, approximate targets)
# ---------------------------------------------------------------
c = 3e8
fc = 10e9               # carrier 10 GHz
lam = c / fc

B = 20e6                # bandwidth 20 MHz  -> dr = 7.5 m
Tp = 5e-6               # pulse width 5 us
K_chirp = B / Tp

PRF = 15e3              # 15 kHz
PRI = 1.0 / PRF

alpha_os = 1.2          # oversampling factor: fs = alpha_os*B (complex, >= 1)
fs = alpha_os * B
Ts = 1.0 / fs

Nc = 256                # chirps per frame (power of 2, sets dv)
Nd = Nc                 # Doppler bins = Nc (2^N, no zero padding)
Np = int(np.floor(Tp * fs))     # samples per pulse

t_start = Tp            # receive window [Tp, PRI)
t_end = PRI

Ns = int(np.floor((t_end - t_start) / Ts))      # all collected samples
Nr = 1024                                       # range cells kept for viewing
Nr = min(Nr, Ns)

t_fast = t_start + np.arange(Ns) * Ts
range_axis = c * t_fast / 2

df_bin = PRF / Nd
doppler_axis = np.arange(-Nd // 2, Nd // 2) * df_bin   # -Nd/2 .. Nd/2-1
velocity_axis = -doppler_axis * lam / 2                # receding positive

R_ua = c * (PRI - Tp) / 2.0     # max delay for FULL reception is PRI - Tp
R_min = c * Tp / 2.0            # closest visible range, min delay is Tp
dr = c / (2.0 * B)
dr_bin = c / (2.0 * fs)
v_ua = lam * PRF / 4.0
dv = lam * PRF / (2.0 * Nc)
dv_bin = lam * PRF / (2.0 * Nd)

print("design metrics:")
print(f"  R_ua   = {R_ua/1e3:.2f} km (max full-reception delay = PRI - Tp)")
print(f"  R_min  = {R_min/1e3:.2f} km (closest visible range, min delay = Tp)")
print(f"  dr     = {dr:.2f} m (approx);  dr_bin = {dr_bin:.2f} m (fs = {alpha_os:.2f}x B)")
print(f"  v_ua   = {v_ua:.1f} m/s (approx)")
print(f"  dv     = {dv:.3f} m/s (approx, set by Nc = {Nc} real chirps)")
print(f"  dv_bin = {dv_bin:.3f} m/s (grid step, set by Nd = {Nd} FFT bins)")
print(f"  Ns = {Ns} samples | Nr = {Nr} range cells | Np = {Np} samples per pulse")
print(f"  Nc = {Nc} chirps | Nd = {Nd} Doppler bins")
print(f"  range axis (Ns): {range_axis[0]/1e3:.2f} .. {range_axis[-1]/1e3:.2f} km")
print(f"  range axis (Nr): {range_axis[0]/1e3:.2f} .. {range_axis[Nr-1]/1e3:.2f} km")

# ---------------------------------------------------------------
# 2. targets (range m, velocity m/s, amplitude dB, phase rad)
#    all placed inside the Nr range window (~7.1 km)
# ---------------------------------------------------------------
targets = np.array([
    [2500.0,  40.0, 0.0,  np.pi / 3],    # 2.5 km, receding 40 m/s, 10 dB
    [5000.0,   0.0,  5.0, -np.pi / 4],    # 5 km, stationary, 6 dB
    [6500.0, -50.0, -5.0,  np.pi / 6],          # 6.5 km, approaching 50 m/s, 12 dB
])
num_targets = targets.shape[0]

# ---------------------------------------------------------------
# 3. transmit baseband signal (matched filter is built from its FFT)
# ---------------------------------------------------------------
t_pulse = np.arange(Np) * Ts
s_pulse = np.exp(1j * np.pi * K_chirp * t_pulse**2)

# ---------------------------------------------------------------
# 4. echo collection (raw_matrix) and pulse compression
# ---------------------------------------------------------------
raw_matrix = np.zeros((Nc, Ns), dtype=complex)      # (chirp, sample) = (v, r)
data_matrix = np.zeros((Nc, Ns), dtype=complex)     # after pulse compression

rng = np.random.default_rng(42)

for m in range(Nc):
    rx_pulse = np.zeros(Ns, dtype=complex)

    for k in range(num_targets):
        R0 = targets[k, 0]
        v = targets[k, 1]
        amp = 10.0 ** (targets[k, 2] / 20.0)        # dB -> amplitude
        phase = targets[k, 3]

        R_inst = R0 + v * (m * PRI)     # instantaneous range
        tau_m = 2.0 * R_inst / c        # delay of this pulse

        t_echo_start = max(t_start, tau_m)
        t_echo_end = min(t_end, tau_m + Tp)

        if t_echo_start < t_echo_end:
            idx = (t_fast >= t_echo_start) & (t_fast < t_echo_end)
            t_rel = t_fast[idx] - tau_m
            echo_segment = np.exp(1j * np.pi * K_chirp * t_rel**2)
            carrier_phase = np.exp(-1j * 2.0 * np.pi * fc * tau_m)
            rx_pulse[idx] += amp * np.exp(1j * phase) * echo_segment * carrier_phase

    # complex white Gaussian noise, unit total variance
    noise = (1.0 / np.sqrt(2.0)) * (rng.standard_normal(Ns) + 1j * rng.standard_normal(Ns))
    rx_pulse = rx_pulse + noise

    raw_matrix[m, :] = rx_pulse         # keep the raw collection matrix

    # pulse compression: standard matched filter y = IFFT(R * conj(S))
    L = Ns + Np - 1
    S_pulse = np.fft.fft(s_pulse, L)
    y_conv = np.fft.ifft(np.fft.fft(rx_pulse, L) * np.conj(S_pulse))
    # peak of y_conv sits at the delay index n0; take the first Ns samples
    # so pc[0] maps to t_start (range_axis[0]).
    data_matrix[m, :] = y_conv[:Ns]

# ---------------------------------------------------------------
# 5. MTD along slow time (axis 0 = v), on the truncated Nr window
# ---------------------------------------------------------------
pc_trunc = data_matrix[:, :Nr]                      # (Nc, Nr)

window = np.hamming(Nc)
rd_fft = np.fft.fftshift(np.fft.fft(pc_trunc * window[:, None], Nd, axis=0),
                         axes=0)
rd_map = np.abs(rd_fft)                             # (Nd, Nr)

# ---------------------------------------------------------------
# 6. save the two data products
# ---------------------------------------------------------------
np.save(os.path.join(HERE, "0_raw.npy"), raw_matrix)         # (Nc, Ns) complex
np.save(os.path.join(HERE, "0_trunc.npy"), rd_fft)           # (Nd, Nr) complex, NO abs
print(f"saved 0_raw.npy    ({raw_matrix.shape[0]} x {raw_matrix.shape[1]}, complex)")
print(f"saved 0_trunc.npy  ({rd_fft.shape[0]} x {rd_fft.shape[1]}, complex)")

# ---------------------------------------------------------------
# 7. 2D plot (matplotlib), axes as-is
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.pcolormesh(velocity_axis, range_axis[:Nr] / 1e3, rd_map.T,
                   cmap="viridis", shading="auto")
ax.set_xlabel("velocity (m/s)")
ax.set_ylabel("range (km)")
ax.set_title(f"RD map (2D, linear)  R_ua={R_ua/1e3:.1f} km, v_ua={v_ua:.0f} m/s")
fig.colorbar(im, ax=ax)

# mark targets with their GROUND-TRUTH values (simulation reference
# only; real detection belongs to demo/CFAR)
for k in range(num_targets):
    R = targets[k, 0]
    v = targets[k, 1]
    ax.plot(v, R / 1e3, "o", mfc="none", mec="w", markersize=10, linewidth=2)
    ax.text(v + 3, R / 1e3 + 0.1, f"R={R/1e3:.1f}km, v={v:.0f}m/s",
            color="w", fontsize=9)

fig.savefig(os.path.join(HERE, "0_trunc_2d.png"), dpi=150)
print("saved 0_trunc_2d.png")
plt.show()

# ---------------------------------------------------------------
# 8. 3D surface -> interactive HTML (plotly, smooth)
# ---------------------------------------------------------------
fig3d = go.Figure(data=[
    go.Surface(z=rd_map.T, x=velocity_axis, y=range_axis[:Nr] / 1e3,
               colorscale="Viridis", colorbar={"title": "amplitude"})
])
fig3d.update_layout(
    title="RD map (3D, linear)",
    scene={
        "xaxis_title": "velocity (m/s)",
        "yaxis_title": "range (km)",
        "zaxis_title": "amplitude (linear)",
    },
)
html_path = os.path.join(HERE, "0_trunc_3d.html")
fig3d.write_html(html_path)
print(f"saved {html_path}")
