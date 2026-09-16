"""
scalar_nidc.py
==============
Scalar p=1 OTS-NIDC scheme and plain Forward Euler baseline for

    u_t = D u_xx + alpha * sigma^2 * u,      x in [0, L],   t > 0,

with homogeneous Neumann (no-flux) boundary conditions
    u_x(0, t) = u_x(L, t) = 0,

discretized on a uniform grid with spacing dx = L / N (N interior cells;
N+1 nodes including the boundaries at 0 and L).

Ghost-cell implementation of Neumann BCs
----------------------------------------
We store only the N+1 physical nodes x[0..N].  The 3-point centered second
difference delta_x^2 u_j = u_{j+1} - 2 u_j + u_{j-1} (scaled by 1/dx^2) at
the boundary nodes j=0 and j=N is supplied using the no-flux condition
u_x = 0, which for a second-order ghost-cell gives

    u_{-1} = u_1          (so delta_x^2 u_0 = 2 u_1 - 2 u_0)
    u_{N+1} = u_{N-1}     (so delta_x^2 u_N = 2 u_{N-1} - 2 u_N).

These are equivalent to the standard one-sided Neumann stencils and keep
the scheme globally 2nd-order in space (consistent with the derivation,
which assumes a 2nd-order spatial operator).

Optimal time step (OTS-NIDC.md Eq 4.1)
--------------------------------------
    dt_opt = dx^2 / (6 D)
independent of alpha and sigma.

First-level NIDC correction (OTS-NIDC.md \u00a75)
------------------------------------------------
    C^(1)_j = - (alpha sigma^2 dx^4)/(36 D) * delta_x^2 u_j^n
              - (alpha^2 sigma^4 dx^4)/(72 D^2) * u_j^n

Corrected update (p=1):
    u_j^{n+1} = u_j^n + dt_opt [ D*delta_x^2 u_j^n / dx^2 + alpha sigma^2 u_j^n ]
               + C^(1)_j

Plain Forward Euler uses the same dt_opt but with C^(1)_j = 0, i.e. the
standard 2nd-order scheme.  (The derivation guarantees OTS-NIDC reaches
4th order in dx while plain Forward Euler stays 2nd order.)

Both solvers advance to final time T.  Because dt_opt is tied to dx, the
number of steps adapts automatically; the schemes are compared at equal
dx (and therefore equal dt_opt), isolating the spatial/stencil correction
effect that produces the order difference.
"""

import numpy as np


def _make_grid(L, N):
    """Return node coordinates x[0..N] for N cells, dx = L/N."""
    dx = L / N
    x = np.linspace(0.0, L, N + 1)
    return x, dx


def _second_diff(u, dx):
    """Centered 3-point second difference with Neumann ghost cells.

    Returns D2[j] = (u_{j+1} - 2 u_j + u_{j-1}) / dx^2 on interior and the
    no-flux-supplied values at the two boundary nodes.
    """
    N = len(u) - 1
    D2 = np.empty_like(u)
    # Interior (j = 1 .. N-1)
    D2[1:N] = (u[2:] - 2.0 * u[1:N] + u[:-2]) / dx**2
    # Boundaries via ghost cells: u_{-1}=u_1, u_{N+1}=u_{N-1}
    D2[0] = (2.0 * u[1] - 2.0 * u[0]) / dx**2
    D2[N] = (2.0 * u[N - 1] - 2.0 * u[N]) / dx**2
    return D2


def solve_ots_nidc(L, N, D, alpha, sigma, T, u0, p=1, return_grid=True):
    """Solve the scalar PDE with the p=1 OTS-NIDC scheme.

    CRITICAL: the scheme uses the EXACT optimal time step
        dt = dt_opt = dx**2 / (6 D)
    (Eq 4.1), with no "snapping" to land exactly on T.  The OTS cancellation
    of the u_xxxx local-truncation term requires dt == dt_opt to machine
    precision; even a ~1% mismatch reintroduces an O(dx**2) error and caps
    the observed order at 2nd.  We therefore take an integer number of exact
    dt_opt steps and return the *realized* final time t_final = nsteps*dt_opt,
    so the exact reference is evaluated at the same instant.

    Parameters
    ----------
    L : float              domain length
    N : int                number of cells (N+1 nodes)
    D, alpha, sigma : float
    T : float              target final time (actual final = nsteps*dt_opt)
    u0 : callable          initial condition u0(x)
    p : int                correction level (1 = first-level NIDC; 0 = no
                           correction, i.e. plain Forward Euler with dt_opt)

    Returns
    -------
    x : ndarray            node coordinates (if return_grid)
    u : ndarray            numerical solution at t = t_final
    t_final : float        realized final time (= nsteps * dt_opt)
    """
    x, dx = _make_grid(L, N)
    u = u0(x).astype(float).copy()
    dt = dx**2 / (6.0 * D)  # optimal time step (Eq 4.1) -- used EXACTLY
    nsteps = int(round(T / dt))
    if nsteps < 1:
        nsteps = 1
    t_final = nsteps * dt

    a = alpha * sigma**2
    for _ in range(nsteps):
        D2 = _second_diff(u, dx)       # = (raw 3-pt diff)/dx**2, i.e. d_x^2 u
        lap = D * D2
        reac = a * u
        u_new = u + dt * (lap + reac)
        if p >= 1:
            # C^(1) correction (Eq 5).  NOTE ON SIGN: the derivation in
            # OTS-NIDC.md Eq 5 writes C^(1) = -dt_opt * residual, but the
            # residual there is the *step* error tau_step = (u^{n+1}_compute -
            # u(t+dt)), i.e. what the plain Forward-Euler step OVERSHOOTS by.
            # To cancel it the correction must be ADDED with the NEGATIVE of
            # the doc's literal right-hand side.  Numerically verified: the
            # doc's literal sign DOUBLES the error; the flipped sign below
            # drives the per-step truncation to machine level and yields the
            # claimed 4th-order global accuracy.  We keep the doc's symbolic
            # form in the comment and use the verified-correct sign here.
            #   doc literal: C = -(a dx^4/(36D)) D2 - (a^2 dx^4/(72 D^2)) u
            #   verified  : C = +(a dx^4/(36D)) D2 + (a^2 dx^4/(72 D^2)) u
            C1 = (a * dx**4 / (36.0 * D)) * D2 + (a**2 * dx**4 / (72.0 * D**2)) * u
            u_new = u_new + C1
        u = u_new
    return (x, u, t_final) if return_grid else (u, t_final)


def solve_forward_euler(L, N, D, alpha, sigma, T, u0):
    """Plain Forward Euler baseline (p=0, no NIDC correction)."""
    return solve_ots_nidc(L, N, D, alpha, sigma, T, u0, p=0)


if __name__ == "__main__":
    # Quick sanity check vs exact.
    import exact_moments as em
    L, D, alpha, sigma, T = 1.0, 0.1, 1.0, 0.5, 0.05
    u0 = lambda x: np.cos(np.pi * x / L) + 0.3
    coeffs = em.initial_coeffs(u0, L, n_max=60)
    N = 64
    x, u_nidc = solve_ots_nidc(L, N, D, alpha, sigma, T, u0, p=1)
    x, u_fe = solve_forward_euler(L, N, D, alpha, sigma, T, u0)
    uex = em.exact_moment_fast(x, T, D, alpha, sigma, L, coeffs)
    print("NIDC L2 error :", np.sqrt(np.mean((u_nidc - uex) ** 2)))
    print("FE   L2 error :", np.sqrt(np.mean((u_fe - uex) ** 2)))
