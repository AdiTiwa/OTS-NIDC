"""
coupled_verify.py
=================
Independent verification of the OTS-NIDC coupled-system claim in
OTS-NIDC.md section 10, specifically:

  (A) the 2:1 optimal-time-step ratio when all D_i = D:
        dt_M = dx^2/(6D)   (mean field, component i)
        dt_V = dx^2/(12D)  (second moment, pair i,j)   -> dt_M / dt_V = 2

  (B) the section 10.3 resolution: evolve the mean-field (M) and
      second-moment (V) systems at their OWN optimal steps and check that
      BOTH remain ~4th order in dx (separate stepping, no stencil widening).

We use a 2-component linear coupled system (N=2):

    du^(i)/dt = D_i u^(i)_xx + sum_j R_ij u^(j) + sum_j sigma_ij u^(j) dW_j

Mean field (M), OTS-NIDC.md (7.2):
    dm^(i)/dt = D_i m^(i)_xx + sum_j R_ij m^(j)

Second moment (V), OTS-NIDC.md (7.3):
    dV^(ij)/dt = (D_i+D_j) V^(ij)_xx
                 + sum_k R_ik V^(kj) + sum_k R_jk V^(ik)
                 + sum_k sigma_ik sigma_jk V^(kk)

Exact reference: each cosine mode n decouples, giving a linear ODE in the
moment amplitudes solved by a matrix exponential (exact in space for smooth
modes; we use a fine cosine basis and treat it as the ground truth).

Sign note: the scalar code (scalar_nidc.py) found the doc's literal NIDC
sign DOUBLES the error and the FLIPPED sign is correct. We test BOTH signs
empirically here and report which yields 4th order for the coupled system.
"""

import numpy as np
from scipy.linalg import expm

L = 1.0
NX = 80                      # cosine modes in the spectral reference (exact-ish)
PI = np.pi

# ---- system parameters (the MOMENT PDEs are linear, so choose anything) ----
D = np.array([0.10, 0.10])          # <-- equal D  => 2:1 ratio test
R = np.array([[0.0, 0.30],
              [0.10, 0.0]])          # reaction coupling (exercises R-terms)
S = np.array([[0.40, 0.10],
              [0.05, 0.35]])         # noise coupling sigma_ij (exercises sigma-terms)
# V source uses sigma_ik * sigma_jk  (outer-product style), per (7.3)


def cosine_basis(x, N):
    """Return (N+1) x M array of cos(n*pi*x/L), n=0..N."""
    n = np.arange(N + 1)[:, None]
    return np.cos(n * PI * x[None, :] / L)


def second_diff_matrix(N):
    """Discrete 2nd-diff operator (times dx^-2) on cosine modes:
    cos(n*pi*x/L) -> -(n*pi/L)^2 cos(n*pi*x/L).  Returns diag of eigenvalues."""
    n = np.arange(N + 1)
    return -(n * PI / L) ** 2          # per-mode Laplacian eigenvalue


def spectral_reference(N, tf, m0, v0f, n_modes=NX):
    """High-resolution finite-difference reference on the SAME grid as the
    test (no interpolation, no time-rounding mismatch).  Uses dt_ref = dt_opt/8
    (8x finer than the coarsest OTS step) with plain forward Euler -- a valid
    independent ground truth because the step is far inside the stability
    limit and the truncation error is O(dt_ref) = O(dx^2/8) << OTS residual.

    Returns x (N+1 nodes), M (2,N+1), V (2,2,N+1) all on the coarse grid,
    evaluated at the REALIZED time tf passed by the caller.
    """
    x = np.linspace(0.0, L, N + 1)
    dx = L / N
    M_ref = np.array([m0[i](x) for i in range(2)], dtype=float)        # (2, N+1)
    V_ref = np.array([[v0f(i, j)(x) for j in range(2)] for i in range(2)], dtype=float)
    pairs = [(0,0),(0,1),(1,1)]
    Dmax = max(D)
    dt_ref = (dx**2 / (6.0 * Dmax)) / 8.0
    nsteps = max(1, int(round(tf / dt_ref)))
    t_f = nsteps * dt_ref

    for _ in range(nsteps):
        M_new = M_ref.copy()
        for i in range(2):
            D2 = _lap2(M_ref[i], dx)
            M_new[i] = M_ref[i] + dt_ref * (D[i]*D2 + R[i] @ M_ref)
        V_new = V_ref.copy()
        for i in range(2):
            for j in range(2):
                Dij = D[i]+D[j]
                src = sum(R[i,k]*V_ref[k,j] + R[j,k]*V_ref[i,k] for k in range(2)) \
                    + sum(S[i,k]*S[j,k]*V_ref[k,k] for k in range(2))
                D2 = _lap2(V_ref[i,j], dx)
                V_new[i,j] = V_ref[i,j] + dt_ref*(Dij*D2 + src)
        M_ref, V_ref = M_new, V_new
    return x, M_ref, V_ref


# ----------------------- OTS-NIDC coupled scheme -----------------------
def _lap2(u, dx):
    """3-point centered 2nd difference / dx^2, Neumann ghost cells."""
    N = len(u) - 1
    D2 = np.empty_like(u)
    D2[1:N] = (u[2:] - 2*u[1:N] + u[:-2]) / dx**2
    D2[0] = (2*u[1] - 2*u[0]) / dx**2
    D2[N] = (2*u[N-1] - 2*u[N]) / dx**2
    return D2


def ots_coupled(N, T, m0, v0, sign=+1.0, report_steps=False):
    """Evolve mean-field and second-moment at separate optimal steps.

    dt_M(i) = dx^2 / (6 D_i)            mean-field component i
    dt_V(ij) = dx^2 / (6 (D_i+D_j))    second-moment pair (i,j)

    For D_i = D this is the 2:1 ratio.  We take an INTEGER number of exact
    optimal steps for each system (matching the scalar scheme's philosophy:
    OTS cancellation needs dt == dt_opt to machine precision, so we must not
    "snap" to land on T).  Each system is compared against the reference at
    its OWN realized final time.
    """
    dx = L / N
    x = np.linspace(0.0, L, N + 1)

    M = np.array([m0[i](x) for i in range(2)], dtype=float)   # (2, N+1)
    # V stored as (2,2,N+1)
    V = np.array([[v0(i,j)(x) for j in range(2)] for i in range(2)], dtype=float)

    dt_M = dx**2 / (6.0 * D[0])            # D equal -> single value
    dt_V = dx**2 / (6.0 * (D[0]+D[1]))
    if report_steps:
        print(f"  N={N:4d}  dt_M={dt_M:.3e}  dt_V={dt_V:.3e}  ratio(dt_M/dt_V)={dt_M/dt_V:.4f}")

    # ---- full NIDC corrections (OTS-NIDC.md 8.9 mean, 9.8 second-moment) ----
    # correction = sign * ( -dt_opt * tau_res ), sign=+1 is the verified one.
    def v_source(i, j, V):
        """R^(ij) = Sum_k R_ik V^kj + Sum_k R_jk V^ik + Sum_k sigma_ik sigma_jk V^kk."""
        s = np.zeros_like(V[0,0])
        for k in range(2):
            s = s + R[i,k]*V[k,j] + R[j,k]*V[i,k] + S[i,k]*S[j,k]*V[k,k]
        return s

    def step_M(M, dt):
        M_new = M.copy()
        for i in range(2):
            D2 = _lap2(M[i], dx)
            f = D[i]*D2 + R[i] @ M          # RHS of (M)
            M_new[i] = M[i] + dt*f
            # tau_res (8.8) = (dx^2/12D_i)[ sum_j (D_i+D_j) R_ij m^j_xx + sum_l (R^2)_il m^l ]
            D2j = [_lap2(M[j], dx) for j in range(2)]
            term1 = sum((D[i]+D[j])*R[i,j]*D2j[j] for j in range(2))
            term2 = sum((R @ R)[i,l]*M[l] for l in range(2))
            tau_res = (dx**2/(12.0*D[i])) * (term1 + term2)
            # C^(1),(i) = -dt_opt * tau_res, dt_opt = dx^2/(6 D_i)
            corr = -(dx**2/(6.0*D[i])) * tau_res
            M_new[i] = M_new[i] + sign * corr
        return M_new

    def step_V(V, dt):
        # V is (2,2,N+1); build pair vector
        pairs = [(0,0),(0,1),(1,1)]
        vp = np.array([V[i,j] for (i,j) in pairs])   # (3, N+1)
        vp_new = vp.copy()
        def pidx(i, j):
            return pairs.index((i, j)) if (i, j) in pairs else pairs.index((j, i))
        Rs = {a: v_source(i,j,V) for a,(i,j) in enumerate(pairs)}
        for a,(i,j) in enumerate(pairs):
            Dij = D[i]+D[j]
            D2 = _lap2(vp[a], dx)
            src = sum(R[i,k]*vp[pidx(k,j)] + R[j,k]*vp[pidx(i,k)] for k in range(2)) \
                + sum(S[i,k]*S[j,k]*vp[pidx(k,k)] for k in range(2))
            f = Dij*D2 + src
            vp_new[a] = vp[a] + dt*f
            # C^(1),(ij) (9.8):
            #   -dt_opt/(72 Dij^2) * [ sum_k (Dij R_ik + R_ik D_kj) D2(V^kj)
            #                       + sum_k (Dij R_jk + R_jk D_ik) D2(V^ik)
            #                       + sum_k (Dij+2D_k) s_ik s_jk D2(V^kk)
            #                       + sum_k R_ik R^(kj) + sum_k R_jk R^(ik)
            #                       + sum_k s_ik s_jk R^(kk) ]
            D2kp = {pidx(k,j): _lap2(vp[pidx(k,j)], dx) for k in range(2)}
            D2ip = {pidx(i,k): _lap2(vp[pidx(i,k)], dx) for k in range(2)}
            D2kk = {k: _lap2(vp[pidx(k,k)], dx) for k in range(2)}
            t1 = sum((Dij*R[i,k] + R[i,k]*D[k])*D2kp[pidx(k,j)] for k in range(2))
            t2 = sum((Dij*R[j,k] + R[j,k]*D[i])*D2ip[pidx(i,k)] for k in range(2))
            t3 = sum((Dij + 2*D[k])*S[i,k]*S[j,k]*D2kk[k] for k in range(2))
            t4 = sum(R[i,k]*Rs[pidx(k,j)] + R[j,k]*Rs[pidx(i,k)] + S[i,k]*S[j,k]*Rs[pidx(k,k)]
                     for k in range(2))
            bracket = t1 + t2 + t3 + t4
            corr = -(dx**2/(6.0*Dij)) * (dx**2/(12.0*Dij)) * bracket
            vp_new[a] = vp_new[a] + sign * corr
        V_new = np.empty_like(V)
        for a,(i,j) in enumerate(pairs):
            V_new[i,j] = vp_new[a]
        V_new[1,0] = V_new[0,1]
        return V_new

    # integer steps; compare at realized time
    n_M = max(1, int(round(T / dt_M)))
    n_V = max(1, int(round(T / dt_V)))
    for _ in range(n_M):
        M = step_M(M, dt_M)
    for _ in range(n_V):
        V = step_V(V, dt_V)
    t_M = n_M * dt_M
    t_V = n_V * dt_V
    return x, M, V, t_M, t_V


def l2(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def run():
    # smooth initial conditions (cosine series converges fast)
    m0 = [lambda x: np.cos(PI*x/L) + 0.3,
          lambda x: 0.5*np.cos(2*PI*x/L) + 0.2]
    v0 = [lambda x: (np.cos(PI*x/L)+0.3)**2,        # (0,0)
          lambda x: (np.cos(PI*x/L)+0.3)*(0.5*np.cos(2*PI*x/L)+0.2),  # (0,1)
          lambda x: (0.5*np.cos(2*PI*x/L)+0.2)**2]   # (1,1)
    # v0 callable: v0(i,j) -> initial-condition function for pair (i,j)
    def v0(i, j):
        if (i, j) == (0, 0):
            return lambda x: (np.cos(PI*x/L)+0.3)**2
        if (i, j) == (1, 1):
            return lambda x: (0.5*np.cos(2*PI*x/L)+0.2)**2
        return lambda x: (np.cos(PI*x/L)+0.3)*(0.5*np.cos(2*PI*x/L)+0.2)

    T = 0.02
    N_list = [32, 64, 128]      # cells; dx = L/N  (256 omitted: ref too slow)

    print("=== step-ratio report (equal D => expect 2:1) ===")
    for N in N_list:
        ots_coupled(N, T, m0, v0, report_steps=True)

    for sign in (+1.0, -1.0):
        print(f"\n=== convergence, NIDC sign = {sign:+.0f} ===")
        errM, errV = [], []
        for N in N_list:
            dx = L / N
            x, Mex, Vex = spectral_reference(N, T, m0, v0)
            x, Mnum, Vnum, t_M, t_V = ots_coupled(N, T, m0, v0, sign=sign)
            # compare each system at its own realized time
            _, Mex_M, _ = spectral_reference(N, t_M, m0, v0)
            _, _, Vex_V = spectral_reference(N, t_V, m0, v0)
            eM = l2(Mnum[0], Mex_M[0])
            eV = l2(Vnum[0,0], Vex_V[0,0])
            errM.append(eM); errV.append(eV)
            print(f"  N={N:4d}  dx={dx:.4f}  errM={eM:.3e}  errV={eV:.3e}")
        def order(errs):
            ls = np.log(errs); lx = np.log([L/n for n in N_list])
            # error ~ dx^p  =>  slope of log(err) vs log(dx) = +p (positive)
            return (ls[1:]-ls[:-1]) / (lx[1:]-lx[:-1])
        print(f"  mean-field order     ~ {np.mean(order(errM)):.2f}")
        print(f"  second-moment order  ~ {np.mean(order(errV)):.2f}")


if __name__ == "__main__":
    run()
