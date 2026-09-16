"""
exact_moments.py
=================
Analytic exact solution for the scalar moment PDE

    u_t = D u_xx + alpha * sigma^2 * u,     x in [0, L],   t > 0,

with homogeneous Neumann (no-flux) boundary conditions

    u_x(0, t) = u_x(L, t) = 0.

Because the Neumann eigenfunctions on [0, L] are the cosine modes
cos(n*pi*x / L) with eigenvalues -(n*pi/L)^2, the solution for an
arbitrary initial condition u(x, 0) = u0(x) expands as

    u(x, t) = a_0/2 + sum_{n>=1} a_n cos(n*pi*x/L) *
              exp( -( D*(n*pi/L)^2 - alpha*sigma^2 ) * t ),

where the cosine-series coefficients of u0 are

    a_0 = (1/L) int_0^L u0(x) dx,
    a_n = (2/L) int_0^L u0(x) cos(n*pi*x/L) dx,   n >= 1.

We compute the coefficients by high-order fixed-order Gauss-Lobatto
quadrature (scipy.integrate.fixed_quad), which is robust for arbitrary
smooth u0 and avoids symbolic work.
"""

import numpy as np
from scipy.integrate import fixed_quad


def _cos_coeff(u0, L, n, n_quad=200):
    """Cosine-series coefficient a_n of u0 on [0, L] (Neumann basis).

    Standard Fourier-cosine convention (cos(0)=1):
        a_n = (2/L) int_0^L u0(x) cos(n*pi*x/L) dx     (all n >= 0)
    so the reconstructed series is
        u(x) = a_0/2 + sum_{n>=1} a_n cos(n*pi*x/L).
    Note a_0/2 = (1/L) int_0^L u0(x) dx, the spatial average.
    """
    val, _ = fixed_quad(lambda x: u0(x) * np.cos(n * np.pi * x / L), 0.0, L, n=n_quad)
    return 2.0 * val / L


def initial_coeffs(u0, L, n_max=200, n_quad=200):
    """Precompute cosine coefficients.

    Returns an array ``coeffs`` of length n_max+1 where
        coeffs[0] = a_0 / 2   (the constant term in the series),
        coeffs[n] = a_n       (n >= 1).
    The reconstructed series is therefore exactly
        u(x,t) = sum_{n=0}^{n_max} coeffs[n] * cos(n*pi*x/L) * exp(-lam_n t)
    with lam_0 = -alpha*sigma^2 and lam_n = D*(n*pi/L)^2 - alpha*sigma^2.
    """
    raw = [_cos_coeff(u0, L, n, n_quad=n_quad) for n in range(n_max + 1)]
    raw[0] = 0.5 * raw[0]
    return np.array(raw, dtype=float)


def exact_moment_fast(x, t, D, alpha, sigma, L, coeffs, n_max=None):
    """Fast exact moment using precomputed coefficients (see initial_coeffs)."""
    if n_max is None:
        n_max = len(coeffs) - 1
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    M = np.zeros_like(x)
    for k in range(n_max + 1):
        if k == 0:
            lam = -alpha * sigma ** 2
        else:
            lam = D * (k * np.pi / L) ** 2 - alpha * sigma ** 2
        M += coeffs[k] * np.cos(k * np.pi * x / L) * np.exp(-lam * t)
    return float(M[0]) if scalar else M


def exact_moment(x, t, D, alpha, sigma, L, u0, n_max=200, n_quad=200):
    """Return the exact moment M(x, t) for the scalar PDE.

    Parameters
    ----------
    x : array_like
        Spatial evaluation points (scalar or array).
    t : float
        Time.
    D, alpha, sigma, L : float
        PDE parameters and domain length.
    u0 : callable
        Initial condition u0(x), defined on [0, L].
    n_max : int
        Number of cosine modes retained in the series.
    """
    coeffs = initial_coeffs(u0, L, n_max=n_max, n_quad=n_quad)
    return exact_moment_fast(x, t, D, alpha, sigma, L, coeffs, n_max=n_max)


if __name__ == "__main__":
    L, D, alpha, sigma = 1.0, 0.1, 1.0, 0.5
    u0 = lambda x: np.cos(np.pi * x / L) + 0.3
    coeffs = initial_coeffs(u0, L, n_max=60)
    print("a0/2 =", coeffs[0])
    print("exact (fast) at t=0.1, x=0.3 :",
          exact_moment_fast(0.3, 0.1, D, alpha, sigma, L, coeffs))
    print("exact (full) at t=0.1, x=0.3 :",
          exact_moment(0.3, 0.1, D, alpha, sigma, L, u0))
    # Verify recovery of initial condition at t=0.
    xs = np.linspace(0, L, 9)
    err0 = np.max(np.abs(exact_moment(xs, 0.0, D, alpha, sigma, L, u0) - u0(xs)))
    print("max |M(x,0) - u0(x)| =", err0)
