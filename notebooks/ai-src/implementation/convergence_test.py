"""
convergence_test.py
====================
Sweep dx over grid refinements, solve the scalar moment PDE

    u_t = D u_xx + alpha sigma^2 u,   Neumann BCs on [0, L],

with each method, compute the L2 error vs the exact analytic moment at
final time T, and estimate convergence order (log-log slope of error vs dx).

Methods compared (all on the SAME grid, hence same optimal dt = dx^2/6D):
    - OTS-NIDC p=1        (expected ~4th order global accuracy)
    - Forward Euler (p=0) (expected ~2nd order)
    - 4th-order FD        (5-point spatial, plain FE; time error caps it ~2nd)
    - Reference expm       (matrix-exponential, exact in time; 2nd-order spatial)

Outputs:
    - error_vs_dx.png     (log-log error vs dx, one line per method)
    - results.md          (table of errors + measured orders, printed too)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exact_moments import initial_coeffs, exact_moment_fast
from scalar_nidc import solve_ots_nidc, solve_forward_euler
from baselines import (solve_4th_order_fd, solve_reference_expm)


# --- problem parameters ------------------------------------------------
L = 1.0
D = 0.1
alpha = 1.0          # second-moment scalar family (k=2 -> alpha = 1)
sigma = 0.5
T = 0.05
# smooth initial condition -> cosine series converges fast
u0 = lambda x: np.cos(np.pi * x / L) + 0.3
coeffs = initial_coeffs(u0, L, n_max=200)
N_list = [32, 64, 128, 256]      # cells; dx = L/N
dx_list = [L / N for N in N_list]


def l2_error(x, u_num, t_final):
    u_ex = exact_moment_fast(x, t_final, D, alpha, sigma, L, coeffs)
    return np.sqrt(np.mean((u_num - u_ex) ** 2))


def err_method(fn):
    errs = []
    for N in N_list:
        x, u, t_final = fn(L, N, D, alpha, sigma, T, u0)
        errs.append(l2_error(x, u, t_final))
    return np.array(errs)


def order(dx, err):
    """Log-log slope between successive refinement levels."""
    return np.log(err[:-1] / err[1:]) / np.log(dx[:-1] / dx[1:])


def main():
    methods = {
        "OTS-NIDC (p=1)": lambda L, N, D, a, s, T, u0: solve_ots_nidc(L, N, D, a, s, T, u0, p=1),
        "Forward Euler": lambda L, N, D, a, s, T, u0: solve_forward_euler(L, N, D, a, s, T, u0),
        "4th-order FD": lambda L, N, D, a, s, T, u0: solve_4th_order_fd(L, N, D, a, s, T, u0),
        "Reference expm": lambda L, N, D, a, s, T, u0: solve_reference_expm(L, N, D, a, s, T, u0),
    }

    results = {}
    for name, fn in methods.items():
        results[name] = err_method(fn)
        print(f"{name:16s} errors: " + " ".join(f"{e:.3e}" for e in results[name]))

    # Estimate orders
    print("\nConvergence orders (log-log slope between levels):")
    orders = {}
    for name, errs in results.items():
        o = order(np.array(dx_list), errs)
        orders[name] = o
        print(f"  {name:16s}: " + " ".join(f"{v:.2f}" for v in o) +
              f"   mean={o.mean():.2f}")

    # --- plot ---
    plt.figure(figsize=(7, 5))
    colors = {"OTS-NIDC (p=1)": "C0", "Forward Euler": "C1",
              "4th-order FD": "C2", "Reference expm": "C3"}
    markers = {"OTS-NIDC (p=1)": "o", "Forward Euler": "s",
               "4th-order FD": "d", "Reference expm": "*"}
    for name, errs in results.items():
        plt.loglog(dx_list, errs, marker=markers[name], color=colors[name],
                   label=name, linewidth=1.5)
    # reference slopes
    dx = np.array(dx_list)
    base = results["Forward Euler"][0]
    plt.loglog(dx, base * (dx / dx[0]) ** 2, "k--", alpha=0.5, label="2nd order")
    plt.loglog(dx, base * (dx / dx[0]) ** 4, "k:", alpha=0.5, label="4th order")
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$L^2$ error at $T$")
    plt.title("Scalar moment PDE: convergence vs exact analytic moment\n"
              rf"(D={D}, $\alpha$={alpha}, $\sigma$={sigma}, T={T})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_dx.png", dpi=130)
    print("\nSaved error_vs_dx.png")

    # --- results.md ---
    lines = []
    lines.append("# Convergence Test Results: OTS-NIDC vs Baselines\n")
    lines.append(f"**Problem:** `u_t = D u_xx + alpha*sigma^2 u`, "
                 f"Neumann BCs on [0, {L}].\n")
    lines.append(f"- D = {D}, alpha = {alpha} (second-moment scalar family), "
                 f"sigma = {sigma}, T = {T}\n")
    lines.append(f"- Initial condition: `u0(x) = cos(pi x/L) + 0.3`\n")
    lines.append(f"- Optimal time step per level: `dt_opt = dx^2 / (6 D)` "
                 f"(same for every method)\n")
    lines.append(f"- Grids (cells N): {N_list}\n")
    lines.append("## L2 error at final time T\n")
    header = "| Method | " + " | ".join(f"N={N} (dx={L/N:.4f})" for N in N_list) + " |\n"
    lines.append(header)
    lines.append("|" + "---|" * (len(N_list) + 1) + "\n")
    for name, errs in results.items():
        row = "| " + name + " | " + " | ".join(f"{e:.3e}" for e in errs) + " |\n"
        lines.append(row)
    lines.append("\n## Measured convergence order (log-log slope)\n")
    lines.append("| Method | slope(32->64) | slope(64->128) | slope(128->256) | mean |\n")
    lines.append("|---|---|---|---|---|\n")
    for name, o in orders.items():
        row = (f"| {name} | {o[0]:.2f} | {o[1]:.2f} | {o[2]:.2f} | {o.mean():.2f} |\n")
        lines.append(row)
    lines.append("\n## Expected vs observed\n")
    lines.append("- **OTS-NIDC (p=1):** expected ~4th order; observed "
                 f"mean {orders['OTS-NIDC (p=1)'].mean():.2f}.\n")
    lines.append("- **Forward Euler:** expected ~2nd order; observed "
                 f"mean {orders['Forward Euler'].mean():.2f}.\n")
    lines.append("- **4th-order FD + plain FE:** time error from dt_opt caps "
                 "it at ~2nd order (observed "
                 f"{orders['4th-order FD'].mean():.2f}).\n")
    lines.append("- **Reference expm:** time-exact 2nd-order spatial; observed "
                 f"{orders['Reference expm'].mean():.2f}.\n")
    lines.append("\n*Generated by convergence_test.py.*\n")

    with open("results.md", "w") as f:
        f.writelines(lines)
    print("Saved results.md")


if __name__ == "__main__":
    main()
