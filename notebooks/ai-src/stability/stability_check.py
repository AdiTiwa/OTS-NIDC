#!/usr/bin/env python3
"""
Von Neumann stability check for OTS-NIDC (p=1) vs plain Forward Euler (FE)
on the scalar family  u_t = D u_xx + alpha*sigma^2 u.

We probe the amplification factors
    G_FE(k)   = 1 - 4 mu s + mu*eta
    G_NIDC(k) = 1 - 4 mu s + mu*eta + (eta/9) s - eta^2/72
where
    mu = D*dt/dx^2            (diffusion number)
    eta = alpha*sigma^2*dx^2/D (reaction number, O(dx^2))
    s  = sin^2(k*dx/2)
and the NIDC correction is the p=1 form from OTS-NIDC.md Eq. 5
(C^(1) is O(dx^4); here written in its von-Neumann (Fourier) form with the
 dt_opt-baked prefactor as given in the paper; i.e. the correction does NOT
 carry an extra free dt).

# NOTE on sign dependence: the correction's effect on the diffusion-limited
# CFL depends on the SIGN of eta = alpha*sigma^2*dx^2/D.  For the moment
# equations of the stochastic heat equation alpha = k(k-1)/2 >= 0, so eta > 0
# (growth).  With the CORRECTED sign of C^(1) (OTS-NIDC.md Eq.5, +),
# for eta > 0 the correction SHRINKS the grid-scale stable region slightly;
# for eta < 0 (decay) it EXPANDS it slightly.  The effect is O(eta)=O(dx^2)
# and tiny either way.  We report the growth case (relevant to the project).

Outputs: stability_plot.png and a textual report.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Parameters (choose dx=1, D=1 to keep mu, eta transparent) -------------
D = 1.0
dx = 1.0
eta = 0.30            # reaction number = alpha*sigma^2 * dx^2 / D  (>0: growth)
dt_opt = dx**2 / (6.0 * D)   # OTS optimal step  (= 1/6 here)
mu_opt = D * dt_opt / dx**2  # = 1/6

# ---- Amplification factors -------------------------------------------------
def G_FE(mu, s):
    b = mu * eta
    return 1.0 - 4.0 * mu * s + b

def G_NIDC(mu, s):
    b = mu * eta
    # p=1 OTS-NIDC correction in Fourier space (SIGN per corrected OTS-NIDC.md Eq.5):
    #   C^(1) = +dt*(alpha sigma^2 dx^2/6) d_x^2 u  + ...
    #   Fourier: d_x^2 u -> -2 s/dx^2, so +dt*(a dx^2/6)*(-2 s/dx^2) = -(eta/9)*s
    #   and    +dt*(a^2 dx^2/(12D)) u         = +eta^2/72
    return 1.0 - 4.0 * mu * s + b - (eta / 9.0) * s + (eta**2) / 72.0

# ---- Grid of wavenumbers ---------------------------------------------------
theta = np.linspace(0.0, np.pi, 600)        # theta = k*dx
x = theta / np.pi                            # x = k*dx/pi  in [0,1]
s = np.sin(theta / 2.0) ** 2

# ---- (a) FE formula check at the representative point ----------------------
# Sanity: G_FE at s=0 should be 1 + eta*mu
mu_test = 0.25
print("Sanity G_FE(s=0) =", G_FE(mu_test, 0.0), " expected 1+eta*mu =", 1 + eta * mu_test)

# ---- (b) Stability boundaries (grid-scale mode, s=1) -----------------------
# FE:  1 - 4 mu + mu*eta >= -1  =>  mu <= 2/(4 - eta)
# NIDC: 1 - 4 mu + mu*eta - eta/9 + eta^2/72 >= -1
#       =>  mu <= (2 - eta/9 + eta^2/72)/(4 - eta)
mu_FE_max = 2.0 / (4.0 - eta)
mu_NIDC_max = (2.0 - eta / 9.0 + eta**2 / 72.0) / (4.0 - eta)
factor = mu_NIDC_max / mu_FE_max   # <1 => slight shrink

print("\n=== Diffusion-limited (grid-scale, s=1) CFL boundary ===")
print(f"  eta (reaction number)        = {eta:.4f}")
print(f"  FE   stable mu <=           = {mu_FE_max:.6f}  (dt <= {mu_FE_max*dx**2/D:.6f})")
print(f"  NIDC stable mu <=           = {mu_NIDC_max:.6f}  (dt <= {mu_NIDC_max*dx**2/D:.6f})")
print(f"  Expansion factor (NIDC/FE)  = {factor:.6f}   ({100*(factor-1):.2f} %)")
print(f"  FE   CFL dt <= dx^2/(2D)    = {dx**2/(2*D):.6f}")
print(f"  dt_opt = dx^2/(6D)          = {dt_opt:.6f}  (sits at mu={mu_opt:.4f} < FE CFL {0.5})")

# ---- (c) Evaluate |G| for the requested dt set -----------------------------
dt_factors = [0.5, 1.0, 1.5, 2.0]
results = []
print("\n=== |G| over k for each scheme / dt (reaction mode k=0 dominates max) ===")
print(f"{'dt/dt_opt':>9} {'mu':>8} {'max|G_FE|':>10} {'max|G_NIDC|':>12} {'FE_grid':>9} {'NIDC_grid':>10}")
for f in dt_factors:
    dt = dt_opt * f
    mu = D * dt / dx**2
    Gfe = G_FE(mu, s)
    Gn = G_NIDC(mu, s)
    max_fe = np.max(np.abs(Gfe))
    max_n = np.max(np.abs(Gn))
    fe_grid = G_FE(mu, 1.0)   # s=1, grid scale
    n_grid = G_NIDC(mu, 1.0)
    results.append((f, mu, max_fe, max_n, fe_grid, n_grid))
    print(f"{f:>9.2f} {mu:>8.4f} {max_fe:>10.4f} {max_n:>12.4f} {fe_grid:>9.4f} {n_grid:>10.4f}")

# ---- (d) Plot --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: |G| vs k*dx/pi for the dt set
ax = axes[0]
colors = ["C0", "C1", "C2", "C3"]
for (f, mu, *_), c in zip(results, colors):
    Gfe = G_FE(mu, s)
    Gn = G_NIDC(mu, s)
    ax.plot(x, np.abs(Gfe), "--", color=c, label=f"FE  dt={f:.1f} dt_opt")
    ax.plot(x, np.abs(Gn), "-", color=c, label=f"NIDC dt={f:.1f} dt_opt")
ax.axhline(1.0, color="k", lw=1.0, ls=":")
ax.set_xlabel(r"$k\Delta x/\pi$")
ax.set_ylabel(r"$|G|$")
ax.set_title(f"Amplification factor vs wavenumber  (eta={eta})")
ax.set_ylim(0.0, 1.6)
ax.legend(fontsize=7, ncol=2, loc="upper right")

# Panel 2: grid-scale amplification G(pi) vs mu, with CFL floors
ax = axes[1]
mu_grid = np.linspace(0.0, 0.8, 400)
Gfe_g = G_FE(mu_grid, 1.0)
Gn_g = G_NIDC(mu_grid, 1.0)
ax.plot(mu_grid, Gfe_g, "b--", lw=2, label="FE  G(k=pi)")
ax.plot(mu_grid, Gn_g, "r-", lw=2, label="NIDC G(k=pi)")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.axhline(-1.0, color="k", lw=0.8, ls=":")
ax.axvline(mu_FE_max, color="b", ls=":", lw=1.2)
ax.axvline(mu_NIDC_max, color="r", ls=":", lw=1.2)
ax.annotate(f"FE CFL\nmu={mu_FE_max:.3f}", (mu_FE_max, 0.6),
            fontsize=8, color="b", ha="right")
ax.annotate(f"NIDC CFL\nmu={mu_NIDC_max:.3f}", (mu_NIDC_max, -0.6),
            fontsize=8, color="r", ha="left")
ax.set_xlabel(r"$\mu = D\Delta t/\Delta x^2$")
ax.set_ylabel(r"$G(k=\pi/\Delta x)$")
ax.set_title("Grid-scale mode: diffusion-limited CFL")
ax.legend(fontsize=8, loc="lower left")

fig.suptitle(
    f"OTS-NIDC p=1 von Neumann stability (eta={eta}, dt_opt={dt_opt:.4f})\n"
    f"NIDC grid-scale CFL shrinks FE by factor {factor:.4f} ({100*(factor-1):.2f}%)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("stability_plot.png", dpi=130)
print("\nSaved stability_plot.png")
