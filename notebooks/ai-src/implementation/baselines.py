"""
baselines.py
============
Reference / competitor methods for the scalar moment problem

    u_t = D u_xx + alpha * sigma^2 * u,      x in [0, L],

with homogeneous Neumann BCs.  The primary focus is the deterministic
moment-equation solve (the object OTS-NIDC targets).  We therefore provide:

  1. solve_4th_order_fd    -- 5-point 4th-order spatial Laplacian + plain
                              Forward Euler at the SAME optimal dt used by
                              OTS-NIDC.  Demonstrates that high-order spatial
                              accuracy alone is NOT enough: with dt_opt ~ dx^2
                              the O(dt) time error still dominates -> ~2nd order.
  2. solve_reference_expm  -- exact-implicit method: 2nd-order FD semi-
                              discretization advanced by the exact matrix
                              exponential expm(A*T) (one step).  Time-exact,
                              spatial 2nd-order; a clean reference.
  3. solve_spectral_cosine -- pseudospectral cosine-collocation reference:
                              expand on the Neumann cosine basis, evolve each
                              mode by its exact exponential.  Spectral accuracy.
  4. euler_maruyama_moment -- pathwise Euler-Maruyama on the *actual SPDE*
                              u_t = D u_xx + sigma u * dot{W}(x,t), Monte-Carlo
                              averaged to estimate the moment E[u^k].  This is
                              the canonical weak-order-1 competitor; included to
                              contextualize OTS-NIDC's weak-convergence claim.

All deterministic solvers advance to final time T on the same grid so the
dx-sweep is a clean spatial(+-time-step) comparison.
"""

import numpy as np
from scipy.integrate import fixed_quad
from scipy.linalg import expm
from scipy.fft import dct, idct

try:
    from exact_moments import initial_coeffs, exact_moment_fast
except ImportError:  # allow standalone import
    pass


# ---------------------------------------------------------------------------
# 1. 4th-order FD (5-point spatial stencil) + plain Forward Euler, same dt_opt
# ---------------------------------------------------------------------------
def _lap4(u, dx):
    """4th-order centered Laplacian with even-reflection (Neumann) ghost cells.

    Interior stencil: ( -u_{j-2} + 16 u_{j-1} - 30 u_j + 16 u_{j+1} - u_{j+2} )
                      / (12 dx^2)
    Boundaries: extend by u(-x) = u(x) (even reflection, exact for u_x=0),
    giving u_{-1}=u_1, u_{-2}=u_2, etc.  This keeps the stencil 4th-order
    consistent at the walls.
    """
    N = len(u) - 1
    up = np.empty(N + 5)
    up[2:N + 3] = u
    up[1] = u[1]      # u_{-1}
    up[0] = u[2]      # u_{-2}
    up[N + 3] = u[N - 1]  # u_{N+1}
    up[N + 4] = u[N - 2]  # u_{N+2}
    lap = (-up[0:N + 1] + 16 * up[1:N + 2] - 30 * up[2:N + 3]
           + 16 * up[3:N + 4] - up[4:N + 5]) / (12.0 * dx**2)
    return lap


def solve_4th_order_fd(L, N, D, alpha, sigma, T, u0):
    x, dx = L / N, L / N
    xs = np.linspace(0.0, L, N + 1)
    u = u0(xs).astype(float).copy()
    dt = dx**2 / (6.0 * D)
    nsteps = max(1, int(round(T / dt)))
    dt = T / nsteps          # plain FE baseline: snapped dt (no OTS requirement)
    a = alpha * sigma**2
    for _ in range(nsteps):
        u = u + dt * (D * _lap4(u, dx) + a * u)
    return xs, u, T


# ---------------------------------------------------------------------------
# 2. Reference: exact-implicit (matrix exponential) on 2nd-order FD matrix
# ---------------------------------------------------------------------------
def _fd_matrix(N, dx, D):
    """Neumann 2nd-order FD Laplacian matrix (size (N+1)x(N+1))."""
    n = N + 1
    A = np.zeros((n, n))
    r = D / dx**2
    for j in range(n):
        A[j, j] = -2.0 * r
        if j >= 1:
            A[j, j - 1] = r
        if j <= N - 1:
            A[j, j + 1] = r
    # Neumann ghost cells: D2[0] = (2 u_1 - 2 u_0)/dx^2
    A[0, 0] = -2.0 * r
    A[0, 1] = 2.0 * r
    A[N, N] = -2.0 * r
    A[N, N - 1] = 2.0 * r
    return A


def solve_reference_expm(L, N, D, alpha, sigma, T, u0):
    x, dx = L / N, L / N
    xs = np.linspace(0.0, L, N + 1)
    u0v = u0(xs).astype(float)
    dt = dx**2 / (6.0 * D)
    nsteps = max(1, int(round(T / dt)))
    t_final = nsteps * dt     # exact in time, so advance by the same t_final
    A = _fd_matrix(N, dx, D) + alpha * sigma**2 * np.eye(N + 1)
    u = expm(A * t_final) @ u0v
    return xs, u, t_final


# ---------------------------------------------------------------------------
# 3. Pseudospectral cosine-collocation reference (exact in time per mode)
# ---------------------------------------------------------------------------
def solve_spectral_cosine(L, N, D, alpha, sigma, T, u0):
    """Cosine-DCT spectral solver; evolve each mode by exp(-lambda_n T)."""
    xs = np.linspace(0.0, L, N + 1)
    u0v = u0(xs).astype(float)
    # match the realized final time of the OTS/dt_opt runs for a fair compare
    dx = L / N
    dt = dx**2 / (6.0 * D)
    nsteps = max(1, int(round(T / dt)))
    t_final = nsteps * dt
    # DCT-I (even reflection) gives the cosine-series coefficients.
    c = dct(u0v, type=1, norm="ortho")
    m = np.arange(N + 1)
    lam = D * (m * np.pi / L) ** 2 - alpha * sigma**2
    c = c * np.exp(-lam * t_final)
    u = idct(c, type=1, norm="ortho")
    return xs, u, t_final


# ---------------------------------------------------------------------------
# 4. Euler-Maruyama on the actual SPDE (pathwise), Monte-Carlo moment
# ---------------------------------------------------------------------------
def euler_maruyama_moment(L, N, D, sigma, T, u0, k_moment,
                          n_paths=2000, seed=12345):
    """Estimate E[u(x,T)^k] of the SPDE u_t = D u_xx + sigma u dot{W}.

    Returns (x, M_est) where M_est[j] ~ E[u(x_j,T)^k_moment].

    Noise discretization (space-time white noise): the per-cell, per-step
    Wiener increment is sigma * u * sqrt(dt/dx) * Z, Z ~ N(0,1).
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, L, N + 1)
    dx = L / N
    dt = dx**2 / (6.0 * D)
    nsteps = max(1, int(round(T / dt)))
    dt = T / nsteps
    u0v = u0(xs).astype(float)
    a = sigma**2  # SPDE noise amplitude
    # precompute 2nd-order FD operator with Neumann BCs (matrix-free)
    r = D / dx**2
    acc = np.zeros_like(u0v)
    Msum = np.zeros_like(u0v)
    for p in range(n_paths):
        u = u0v.copy()
        for _ in range(nsteps):
            # interior
            acc[1:N] = r * (u[2:] - 2.0 * u[1:N] + u[:-2])
            acc[0] = r * (2.0 * u[1] - 2.0 * u[0])
            acc[N] = r * (2.0 * u[N - 1] - 2.0 * u[N])
            # space-time white noise increment
            noise = a * u * np.sqrt(dt / dx) * rng.standard_normal(N + 1)
            u = u + dt * acc + noise
        Msum += u**k_moment
    return xs, Msum / n_paths


if __name__ == "__main__":
    import exact_moments as em
    L, D, alpha, sigma, T = 1.0, 0.1, 1.0, 0.5, 0.05
    u0 = lambda x: np.cos(np.pi * x / L) + 0.3
    coeffs = em.initial_coeffs(u0, L, n_max=80)
    for name, fn in [
        ("4thFD", solve_4th_order_fd),
        ("expm", solve_reference_expm),
        ("spectral", solve_spectral_cosine),
    ]:
        x, u = fn(L, 64, D, alpha, sigma, T, u0)
        uex = em.exact_moment_fast(x, T, D, alpha, sigma, L, coeffs)
        print(f"{name:9s} L2 error:", np.sqrt(np.mean((u - uex) ** 2)))
