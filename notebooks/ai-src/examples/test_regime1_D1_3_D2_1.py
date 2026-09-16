#!/usr/bin/env python3
"""
Convergence test for coupled OTS-NIDC with D₁=3, D₂=1 (Regime 1: r=3).
Uses the exact same OTS-NIDC machinery as the verified single-species implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'implementation'))

from exact_moments import initial_coeffs


def exact_coupled_solution(m1_0_func, m2_0_func, L, N, D1, D2, R, T):
    """
    Exact solution for the coupled M-block system using Fourier-cosine modes.
    m1_0_func, m2_0_func: callable initial condition functions
    """
    x = np.linspace(0, L, N + 1)
    dx = L / N
    
    # Compute cosine coefficients for both initial conditions
    coeffs1 = initial_coeffs(m1_0_func, L, n_max=80)
    coeffs2 = initial_coeffs(m2_0_func, L, n_max=80)
    
    # Time evolution for each mode
    m1_exact = np.zeros(N + 1)
    m2_exact = np.zeros(N + 1)
    
    n_max = len(coeffs1) - 1
    for n in range(n_max + 1):
        if n == 0:
            lam1 = 0.0
            lam2 = 0.0
        else:
            k = n * np.pi / L
            lam1 = D1 * k**2
            lam2 = D2 * k**2
        
        # Coupled system for mode n: da/dt = A a
        A = np.array([[-lam1 + R[0,0], R[0,1]],
                      [R[1,0], -lam2 + R[1,1]]])
        from scipy.linalg import expm
        expA = expm(A * T)
        a0 = np.array([coeffs1[n], coeffs2[n]])
        a_final = expA @ a0
        
        # Add to solution
        mode = np.cos(n * np.pi * x / L)
        m1_exact += a_final[0] * mode
        m2_exact += a_final[1] * mode
    
    return m1_exact, m2_exact


def laplacian_neumann(u, dx):
    """Neumann Laplacian with ghost cells (from scalar_nidc.py)"""
    N = len(u) - 1
    D2 = np.empty_like(u)
    D2[1:N] = (u[2:] - 2.0 * u[1:N] + u[:-2]) / dx**2
    D2[0] = (2.0 * u[1] - 2.0 * u[0]) / dx**2
    D2[N] = (2.0 * u[N - 1] - 2.0 * u[N]) / dx**2
    return D2


def wide_stencil_raw_neumann(u):
    """5-point stencil for u_xxxx (Neumann BC via ghost cells)"""
    N = len(u) - 1
    wide = np.zeros_like(u)
    # Interior points
    for j in range(2, N - 1):
        wide[j] = u[j-2] - 4*u[j-1] + 6*u[j] - 4*u[j+1] + u[j+2]
    return wide


def ots_correction_p1_optimal(u, D, dt, dx):
    """Standard OTS-NIDC p=1 correction at optimum dt = dx²/(6D)"""
    wide = wide_stencil_raw_neumann(u)
    return - wide / 72.0


def mismatch_correction_species1(u1):
    """Exact mismatch correction for D=3 at shared step Δt* = dx²/6"""
    wide = wide_stencil_raw_neumann(u1)
    return - wide / 12.0


def coupled_step_regime1(m1, m2, dx, dt, D1, D2, R):
    """
    One shared step for D₁=3, D₂=1 with exact mismatch correction.
    Uses the exact same structure as scalar_nidc.py but for coupled system.
    """
    # Laplacians
    lap1 = laplacian_neumann(m1, dx)
    lap2 = laplacian_neumann(m2, dx)
    
    # Reaction terms (coupling)
    react1 = R[0,0]*m1 + R[0,1]*m2
    react2 = R[1,0]*m1 + R[1,1]*m2
    
    # Standard OTS corrections at respective optimums
    # Species 2 is at its optimum (dt = dx²/(6*1) = dx²/6)
    # Species 1 is NOT at its optimum (its optimum would be dx²/18)
    c2_std = ots_correction_p1_optimal(m2, D2, dt, dx)
    
    # Mismatch correction for species 1
    c1_mismatch = mismatch_correction_species1(m1)
    
    # Total corrections
    c1_total = c1_mismatch  # No standard correction since not at optimum
    c2_total = c2_std
    
    # Updates (using midpoint coupling for 2nd order in time)
    # Predictor
    m1_mid = m1 + 0.5 * dt * (D1 * lap1 + react1)
    m2_mid = m2 + 0.5 * dt * (D2 * lap2 + react2)
    
    lap1_mid = laplacian_neumann(m1_mid, dx)
    lap2_mid = laplacian_neumann(m2_mid, dx)
    react1_mid = R[0,0]*m1_mid + R[0,1]*m2_mid
    react2_mid = R[1,0]*m1_mid + R[1,1]*m2_mid
    
    m1_new = m1 + dt * (D1 * lap1_mid + react1_mid) + c1_total
    m2_new = m2 + dt * (D2 * lap2_mid + react2_mid) + c2_total
    
    return m1_new, m2_new


def run_convergence_test():
    L = 1.0
    D1 = 3.0
    D2 = 1.0
    R = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    T = 0.05
    
    # Initial conditions: single cosine mode (n=1)
    m1_0_func = lambda x: np.cos(np.pi * x / L)
    m2_0_func = lambda x: np.sin(np.pi * x / L)
    
    resolutions = [32, 64, 128, 256]
    errors_m1 = []
    errors_m2 = []
    
    print("=" * 60)
    print("Coupled OTS-NIDC Convergence Test: D₁=3, D₂=1 (Regime 1)")
    print("=" * 60)
    
    for N in resolutions:
        dx = L / N
        x = np.linspace(0, L, N + 1)
        
        m1_0 = m1_0_func(x)
        m2_0 = m2_0_func(x)
        
        # Exact solution
        m1_exact, m2_exact = exact_coupled_solution(m1_0_func, m2_0_func, L, N, D1, D2, R, T)
        
        # Numerical solution
        dt = dx**2 / (6.0 * D2)  # τ₂ = dx²/6
        nsteps = int(round(T / dt))
        actual_dt = T / nsteps
        actual_dt *= 0.9  # Safety factor
        
        m1_num = m1_0.copy()
        m2_num = m2_0.copy()
        
        for step in range(nsteps):
            m1_num, m2_num = coupled_step_regime1(m1_num, m2_num, dx, actual_dt, D1, D2, R)
        
        # L2 error
        err_m1 = np.sqrt(np.mean((m1_num - m1_exact)**2))
        err_m2 = np.sqrt(np.mean((m2_num - m2_exact)**2))
        
        errors_m1.append(err_m1)
        errors_m2.append(err_m2)
        
        print(f"N={N:3d}, dx={dx:.6f}, dt={actual_dt:.6f}, "
              f"err_m1={err_m1:.2e}, err_m2={err_m2:.2e}")
    
    print("\nConvergence rates:")
    for i in range(len(resolutions)-1):
        rate_m1 = np.log(errors_m1[i] / errors_m1[i+1]) / np.log(2)
        rate_m2 = np.log(errors_m2[i] / errors_m2[i+1]) / np.log(2)
        print(f"  {resolutions[i]}->{resolutions[i+1]}: m1 rate={rate_m1:.2f}, m2 rate={rate_m2:.2f}")
    
    return resolutions, errors_m1, errors_m2


if __name__ == "__main__":
    resolutions, errors_m1, errors_m2 = run_convergence_test()
    
    # Plot
    plt.figure(figsize=(8, 6))
    dxs = [1.0/r for r in resolutions]
    plt.loglog(dxs, errors_m1, 'o-', label='Species 1 (D=3)')
    plt.loglog(dxs, errors_m2, 's-', label='Species 2 (D=1)')
    
    plt.loglog(dxs, [errors_m1[0] * (dx/dxs[0])**4 for dx in dxs], 'k--', label='O(Δx⁴)')
    plt.loglog(dxs, [errors_m1[0] * (dx/dxs[0])**2 for dx in dxs], 'k:', label='O(Δx²)')
    
    plt.xlabel('Δx')
    plt.ylabel('L² Error')
    plt.title('Coupled OTS-NIDC: D₁=3, D₂=1 (Regime 1, r=3)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/convergence_D1_3_D2_1.png', dpi=150)
    print("\nPlot saved to figures/convergence_D1_3_D2_1.png")