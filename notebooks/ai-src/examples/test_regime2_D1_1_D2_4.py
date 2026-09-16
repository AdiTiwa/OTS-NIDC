#!/usr/bin/env python3
"""
Convergence test for coupled OTS-NIDC with D₁=1, D₂=4 (Regime 2: r=4).
Full implementation with proper OTS-NIDC p=1 corrections and cubic Hermite interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm

# ============================================================
# Problem Setup
# ============================================================

D1 = 1.0  # slow
D2 = 4.0  # fast
R = np.array([[0.0, 1.0],
              [1.0, 0.0]])

def tau_opt(D, dx):
    return dx**2 / (6.0 * D)

# ============================================================
# Exact Solution for M-block
# ============================================================

def exact_solution_mblock(m1_0, m2_0, dx, t_final):
    N = len(m1_0)
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    
    m1_hat = np.fft.fft(m1_0)
    m2_hat = np.fft.fft(m2_0)
    
    m1_exact = np.zeros(N, dtype=complex)
    m2_exact = np.zeros(N, dtype=complex)
    
    for i in range(N):
        ki = k[i]
        A = np.array([[-D1*ki**2 + R[0,0], R[0,1]],
                      [R[1,0], -D2*ki**2 + R[1,1]]])
        expA = expm(A * t_final)
        a0 = np.array([m1_hat[i], m2_hat[i]])
        a_final = expA @ a0
        m1_exact[i] = a_final[0]
        m2_exact[i] = a_final[1]
    
    return np.fft.ifft(m1_exact).real, np.fft.ifft(m2_exact).real


# ============================================================
# OTS-NIDC Operators
# ============================================================

def laplacian(u, dx):
    return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2

def wide_stencil_raw(u):
    """5-point stencil without dx^4 factor"""
    return (np.roll(u, -2) - 4*np.roll(u, -1) + 6*u 
            - 4*np.roll(u, 1) + np.roll(u, 2))

# ---- Standard OTS-NIDC p=1 correction at optimum τ = dx²/(6D) ----
def ots_correction_p1_optimal(u, D, dt, dx):
    """
    Standard OTS-NIDC p=1 correction when dt = τ = dx²/(6D) (at optimum).
    Leading remaining error: (τ²D²/2) u_xxxx = dx⁴/72 u_xxxx
    Correction = - (1/72) * wide_stencil_raw
    """
    wide = wide_stencil_raw(u)
    return - wide / 72.0


def cubic_hermite_interpolant(m1_n, dm1_n, m1_np1, dm1_np1, H, s):
    """
    Evaluate cubic Hermite interpolant at s ∈ [0,1].
    m1(t_n + s*H) ≈ h00(s)*m1_n + h10(s)*H*dm1_n + h01(s)*m1_np1 + h11(s)*H*dm1_np1
    """
    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2
    return h00*m1_n + h10*H*dm1_n + h01*m1_np1 + h11*H*dm1_np1


def macro_cycle_regime2(m1, m2, dx, H):
    """
    One macro cycle of length H = τ₁ for D₁=1, D₂=4 system.
    r = 4 exactly -> 4 substeps of τ₂ = H/4, no remainder.
    """
    inv_dx2 = 1.0 / dx**2
    tau2 = H / 4.0  # = dx²/24
    
    R11, R12 = R[0,0], R[0,1]
    R21, R22 = R[1,0], R[1,1]
    
    # --- STEP 1: Species 1 derivative at t_n ---
    dm1_n = D1 * laplacian(m1, dx) + R11*m1 + R12*m2
    
    # --- STEP 2: Advance species 1 by one macro step H ---
    # Use implicit coupling for better accuracy? No, keep explicit for now.
    react1 = R11*m1 + R12*m2
    m1_np1 = (m1 
              + H * (D1 * laplacian(m1, dx) + react1)
              + ots_correction_p1_optimal(m1, D1, H, dx))
    
    # --- STEP 3: Species 1 derivative at t_n+H ---
    dm1_np1 = D1 * laplacian(m1_np1, dx) + R11*m1_np1 + R12*m2
    
    # --- STEP 4: Subcycle species 2 (4 steps of tau2) ---
    m2_current = m2.copy()
    for k in range(4):
        s_k = (k + 0.5) / 4.0  # Midpoint of substep for better accuracy
        m1_at_sk = cubic_hermite_interpolant(m1, dm1_n, m1_np1, dm1_np1, H, s_k)
        
        react2 = R21 * m1_at_sk + R22 * m2_current
        m2_current = (m2_current 
                      + tau2 * (D2 * laplacian(m2_current, dx) + react2)
                      + ots_correction_p1_optimal(m2_current, D2, tau2, dx))
    
    return m1_np1, m2_current


# ============================================================
# Convergence Test
# ============================================================

def run_convergence_test():
    L = 2 * np.pi
    t_final = 0.05
    
    resolutions = [32, 64, 128, 256]
    errors_m1 = []
    errors_m2 = []
    
    for N in resolutions:
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        
        m1_0 = np.sin(x)
        m2_0 = np.cos(x)
        
        # Exact solution
        m1_exact, m2_exact = exact_solution_mblock(m1_0, m2_0, dx, t_final)
        
        # Numerical solution
        H = tau_opt(D1, dx)  # macro step = τ₁ = dx²/6
        n_cycles = int(t_final / H + 0.5)
        actual_H = t_final / n_cycles
        
        m1_num = m1_0.copy()
        m2_num = m2_0.copy()
        
        for cycle in range(n_cycles):
            m1_num, m2_num = macro_cycle_regime2(m1_num, m2_num, dx, actual_H)
        
        # L2 error
        err_m1 = np.sqrt(np.mean((m1_num - m1_exact)**2))
        err_m2 = np.sqrt(np.mean((m2_num - m2_exact)**2))
        
        errors_m1.append(err_m1)
        errors_m2.append(err_m2)
        
        print(f"N={N:3d}, dx={dx:.6f}, H={actual_H:.6f}, "
              f"err_m1={err_m1:.2e}, err_m2={err_m2:.2e}")
    
    print("\nConvergence rates:")
    for i in range(len(resolutions)-1):
        rate_m1 = np.log(errors_m1[i] / errors_m1[i+1]) / np.log(2)
        rate_m2 = np.log(errors_m2[i] / errors_m2[i+1]) / np.log(2)
        print(f"  {resolutions[i]}->{resolutions[i+1]}: m1 rate={rate_m1:.2f}, m2 rate={rate_m2:.2f}")
    
    return resolutions, errors_m1, errors_m2


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Convergence Test: Coupled OTS-NIDC D₁=1, D₂=4 (Regime 2)")
    print("Subcycling (4 substeps) + Cubic Hermite Interpolation")
    print("=" * 60)
    
    resolutions, errors_m1, errors_m2 = run_convergence_test()
    
    # Plot
    plt.figure(figsize=(8, 6))
    dxs = [2*np.pi/r for r in resolutions]
    plt.loglog(dxs, errors_m1, 'o-', label='Species 1 (D=1)')
    plt.loglog(dxs, errors_m2, 's-', label='Species 2 (D=4)')
    
    plt.loglog(dxs, [errors_m1[0] * (dx/dxs[0])**4 for dx in dxs], 'k--', label='O(Δx⁴)')
    plt.loglog(dxs, [errors_m1[0] * (dx/dxs[0])**2 for dx in dxs], 'k:', label='O(Δx²)')
    
    plt.xlabel('Δx')
    plt.ylabel('L² Error')
    plt.title('Convergence: D₁=1, D₂=4 (Regime 2, r=4)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/convergence_D1_1_D2_4.png', dpi=150)
    print("\nPlot saved to figures/convergence_D1_1_D2_4.png")