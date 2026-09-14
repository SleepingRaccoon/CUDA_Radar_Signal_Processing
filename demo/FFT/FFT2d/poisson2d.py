"""
poisson2d_cpu.py -- Solve the 2D Poisson equation

    laplace(u) + f = 0

on a periodic domain [0, Lx] x [0, Ly] sampled on an Nx x Ny grid,
using the FFT method. Pure NumPy CPU implementation, vectorized.

General case: Lx != Ly and Nx != Ny are all explicit parameters
(unlike the NVIDIA sample, which hard-codes Lx = Ly = 1, Nx = Ny).

Method (see poisson2d.md):
    X     = fft2(f)                       aliased spectrum
    U_hat = X / (Nx*Ny * (kx^2 + ky^2))   frequency-domain solve
    u     = real(ifft2(U_hat))            ifft2 carries 1/(Nx*Ny)

kx, ky are the physical wavenumbers 2*pi*k/L; bins k > N/2 map to
negative frequencies. U_hat[0, 0] = 0 pins the arbitrary constant.

Verification: source f = laplace(u_a) for a Gaussian u_a, so the
exact solution is known and checked by relative L2 error.
"""

import numpy as np


def poisson2d(f, Lx, Ly):
    """Solve laplace(u) + f = 0 on a periodic [0, Lx] x [0, Ly] domain.

    f  : (Ny, Nx) array, source sampled on the grid
         (axis 0 = y, axis 1 = x)
    Lx : domain length in x
    Ly : domain length in y

    Returns u : (Ny, Nx) array, solution on the grid.
    """
    Ny, Nx = f.shape

    hx = Lx / Nx
    hy = Ly / Ny

    # Physical wavenumbers in FFT bin order. fftfreq returns the
    # bin-order frequencies k/(N*h) (positive then negative half).
    kx = np.fft.fftfreq(Nx, d=hx) * 2.0 * np.pi
    ky = np.fft.fftfreq(Ny, d=hy) * 2.0 * np.pi

    # Broadcast to (Ny, Nx): axis 0 = ky, axis 1 = kx.
    Kx, Ky = np.meshgrid(kx, ky)
    lam = Kx * Kx + Ky * Ky          # laplacian eigenvalues

    f_hat = np.fft.fft2(f)           # aliased spectrum (unnormalized)

    # Frequency-domain solve. The aliasing relation X = Nx*Ny*f_hat
    # is exactly cancelled by the 1/(Nx*Ny) built into np.fft.ifft2,
    # so no explicit Nx*Ny factor appears here.
    lam[0, 0] = 1.0                  # avoid 0/0 at DC
    u_hat = f_hat / lam
    u_hat[0, 0] = 0.0                # pin the arbitrary constant (mean)

    u = np.fft.ifft2(u_hat).real     # carries the 1/(Nx*Ny)
    return u


def main():
    # General problem: Lx != Ly, Nx != Ny.
    Lx, Ly = 2.0, 1.5
    Nx, Ny = 128, 96
    s = 0.12

    # Gaussian exact solution centered in the domain, small enough
    # that the periodic continuation is smooth (u_a ~ 0 at borders).
    x = np.arange(Nx) * Lx / Nx
    y = np.arange(Ny) * Ly / Ny
    X, Y = np.meshgrid(x, y)                 # (Ny, Nx)
    r2 = (X - Lx / 2.0) ** 2 + (Y - Ly / 2.0) ** 2

    u_a = np.exp(-r2 / (2.0 * s * s))
    # laplace(u_a) = (r2 - 2s^2)/s^4 * u_a. Equation is
    # laplace(u) + f = 0, so f = -laplace(u_a) makes u_a the solution.
    f = (2.0 * s * s - r2) / (s ** 4) * np.exp(-r2 / (2.0 * s * s))

    # Compatibility condition: mean of f must be ~0 on a periodic
    # domain, otherwise no solution exists.
    print("domain [%g] x [%g], grid %d x %d, s = %g" % (Lx, Ly, Nx, Ny, s))
    print("compatibility  int(f) dOmega = %.3e" % (np.mean(f) * Lx * Ly))

    u = poisson2d(f, Lx, Ly)

    # The Poisson solution carries an arbitrary constant (DC bin is
    # pinned to zero, so u has zero mean). Align both by removing
    # their means before comparing.
    u_pin = u - np.mean(u)
    u_a_pin = u_a - np.mean(u_a)
    rel_l2 = np.linalg.norm(u_pin - u_a_pin) / np.linalg.norm(u_a_pin)
    print("rel L2 error vs exact = %.3e" % rel_l2)

    # Threshold 1e-6: double-precision spectral error is ~1e-8 here;
    # the residual comes from the tiny discontinuity of the Gaussian's
    # periodic continuation (int(f) ~ 1e-7, see above).
    ok = rel_l2 < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
