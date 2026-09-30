#!/usr/bin/env python3
"""
Test the mismatch correction for remainder step ρ in isolation.
Compare against exact substep evolution for pure diffusion.
"""

import numpy as np
from scipy.linalg import expm

# ============================================================
# Test: Exact subcycling vs mismatch-corrected remainder step
# ============================================================

D1 = 1.0
D2 = 4.5
R = np.array([[0.0, 1.0],
              [1.0, 0.0]])

def tau_opt(D, dx):
    return dx**2 / (6.0 * D)

def laplacian(u, dx):
    return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2

def wide_stencil_raw(u):
    return (np.roll(u, -2) - 4*np.roll(u, -1) + 6*u
            - 4*np.roll(u, 1) + np.roll(u, 2))

def mismatch_correction(u, D, rho, dx):
    wide = wide_stencil_raw(u)
    coeff = + (rho**2 * D**2 / 2.0 - rho * D * dx**2 / 12.0) / dx**4
    return coeff * wide


def exact_step(u, D, dt, dx, k):
    """Exact single step for mode k."""
    return u * np.exp(-D * k**2 * dt)

def fe_step(u, D, dt, dx, k):
    """Forward Euler step for mode k."""
    return u * (1 - D * k**2 * dt)


def test_mismatch_correction():
    """Test mismatch correction for a single mode."""
    dx = 2*np.pi / 128
    k = 2*np.pi / 2 / dx  # Mode 2
    tau1 = dx**2 / 6.0
    tau2 = tau1 / 4.5
    n_sub = 4
    rho = tau1 - n_sub * tau2
    
    print(f"dx={dx:.6f}")
    print(f"tau1={tau1:.6f}")
    print(f"tau2={tau2:.6f}")
    print(f"rho={rho:.6f}")
    print(f"rho/tau2={rho/tau2:.6f}")
    
    # Initial values
    u0 = 1.0
    
    # Exact evolution: 4 steps of tau2 + 1 step of rho
    u_exact = u0
    u_exact = exact_step(u_exact, D2, tau2, dx, k)
    u_exact = exact_step(u_exact, D2, tau2, dx, k)
    u_exact = exact_step(u_exact, D2, tau2, dx, k)
    u_exact = exact_step(u_exact, D2, tau2, dx, k)
    u_exact = exact_step(u_exact, D2, rho, dx, k)
    
    # Numerical: 4 FE steps of tau2 + 1 FE+mismatch step of rho
    u_num = u0
    for _ in range(4):
        u_num = fe_step(u_num, D2, tau2, dx, k)
    u_num = fe_step(u_num, D2, rho, dx, k)
    
    # Now with mismatch correction
    # The mismatch correction is applied in physical space
    # For a single Fourier mode u = A cos(kx), u_xxxx = k^4 u
    # wide_stencil_raw(cos(kx)) = 16 * sin^4(k dx/2) * cos(kx) ≈ (k dx)^4 cos(kx) for small k dx
    # Actually for k=2: k dx = 2*2π/128 = π/16, small
    
    # Let's test in physical space with a single mode
    N = 128
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    u = np.cos(2*x)
    
    # Exact at final time
    t_final = tau1
    k_mode = 2
    u_exact = np.cos(2*x) * np.exp(-D2 * k_mode**2 * t_final)
    
    # Numerical with exact subcycling (4 steps of tau2)
    u_num = np.cos(2*x).copy()
    for _ in range(4):
        u_num = u_num + tau2 * D2 * laplacian(u_num, dx)
    
    # Remainder step with FE
    u_num_fe = u_num + rho * D2 * laplacian(u_num, dx)
    
    # Remainder step with FE + mismatch
    u_num_corr = u_num + rho * D2 * laplacian(u_num, dx) + mismatch_correction(u_num, D2, rho, dx)
    
    # Error
    err_fe = np.sqrt(np.mean((u_num_fe - u_exact)**2))
    err_corr = np.sqrt(np.mean((u_num_corr - u_exact)**2))
    
    print(f"\nSingle mode test:")
    print(f"  Exact remainder step error (FE): {err_fe:.2e}")
    print(f"  Corrected remainder step error:  {err_corr:.2e}")
    print(f"  Ratio: {err_fe/err_corr:.2f}x improvement")
    
    # Check the correction coefficient
    # For u = cos(kx), u_xxxx = k^4 u
    # wide_stencil gives: 16 * sin^4(k dx/2) / dx^4 * u
    kdx = k_mode * dx
    exact_xxxx = k_mode**4
    stencil_xxxx = 16 * np.sin(kdx/2)**4 / dx**4
    print(f"\n  k={k_mode}, k*dx={kdx:.6f}")
    print(f"  exact u_xxxx/u = {exact_xxxx:.6f}")
    print(f"  stencil u_xxxx/u = {stencil_xxxx:.6f}")
    print(f"  ratio = {stencil_xxxx/exact_xxxx:.6f}")
    
    # The mismatch correction coefficient
    coeff = (rho**2 * D2**2 / 2.0 - rho * D2 * dx**2 / 12.0) / dx**4
    print(f"\n  mismatch coeff = {coeff:.6f}")
    print(f"  coeff * stencil_xxxx = {coeff * stencil_xxxx:.6f}")
    print(f"  expected error = rho*D2*exact_xxxx*[(rho*D2)/2 - dx^2/12] = {rho * D2 * exact_xxxx * (rho*D2/2 - dx**2/12):.6f}")
    
    return err_fe, err_corr


def test_full_cycle():
    """Test full macro cycle for D2 with remainder."""
    dx = 2*np.pi / 128
    N = 128
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    tau1 = dx**2 / 6.0
    tau2 = tau1 / 4.5
    rho = tau1 - 4*tau2
    
    print(f"\n\nFull macro cycle test (H = tau1 = {tau1:.6f})")
    print(f"tau2={tau2:.6f}, rho={rho:.6f}")
    
    # Initial condition: mix of modes
    u0 = np.sin(x) + 0.5*np.cos(2*x) + 0.3*np.sin(3*x)
    
    # Exact evolution
    t_final = tau1
    k = 2*np.pi * np.fft.fftfreq(N, dx)
    u_hat = np.fft.fft(u0)
    u_exact_hat = u_hat * np.exp(-D2 * k**2 * t_final)
    u_exact = np.fft.ifft(u_exact_hat).real
    
    # Numerical: 4 FE substeps of tau2 + 1 FE+mismatch of rho
    u_num = u0.copy()
    for _ in range(4):
        u_num = u_num + tau2 * D2 * laplacian(u_num, dx)
    u_num = u_num + rho * D2 * laplacian(u_num, dx) + mismatch_correction(u_num, D2, rho, dx)
    
    # Error
    err = np.sqrt(np.mean((u_num - u_exact)**2))
    print(f"  L2 error with mismatch correction: {err:.2e}")
    
    # Without correction
    u_num_fe = u0.copy()
    for _ in range(4):
        u_num_fe = u_num_fe + tau2 * D2 * laplacian(u_num_fe, dx)
    u_num_fe = u_num_fe + rho * D2 * laplacian(u_num_fe, dx)
    err_fe = np.sqrt(np.mean((u_num_fe - u_exact)**2))
    print(f"  L2 error without correction: {err_fe:.2e}")
    print(f"  Improvement: {err_fe/err:.2f}x")


if __name__ == "__main__":
    test_mismatch_correction()
    test_full_cycle()