# notebooks/ai-src — scripts backing the living paper

Source: `OTS-NIDC` repo (code/ and examples/), copied 2026-09-16. Every
**[V]** (verified) number in `paper/main.pdf` was produced by one of these
scripts against an independent exact or high-resolution reference.

## Layout

| Directory | Contents | Paper claims supported |
|---|---|---|
| `implementation/` | scalar p=1 solver (`scalar_nidc.py`), exact cosine-series reference (`exact_moments.py`), baselines incl. Euler–Maruyama (`baselines.py`), dx-sweep driver (`convergence_test.py`), archived output (`results.md`) | scalar 4.01 vs 2.01 measured orders (Table 1); FD negative control |
| `verify/` | coupled M/V harness (`coupled_verify.py`) | 2:1 step ratio [V]; coupled order ≈ 2 — the paper's [C] (contradicted) item |
| `stability/` | von Neumann script (`stability_check.py`) + analysis (`stability_analysis.md`) | CFL shrink factor `1 − η/18`; stability at Δt_opt |
| `examples/` | Regime 1 (D1=3, D2=1, r=3) and Regime 2 (D1=1, D2=4, r=4) convergence tests, runner, worked examples | exact mismatch correction [V]; regime-2 subcycling at integer r |

## Running

Each script is self-contained (numpy/scipy/matplotlib; deterministic seed
12345 where stochastic). Example:

```bash
cd notebooks/ai-src/implementation
python convergence_test.py        # regenerates results.md numbers
```

`examples/test_regime1_D1_3_D2_1.py` resolves `exact_moments` via a
relative path into `../implementation/` (patched from the original
hardcoded path on copy).

Note: the two regime example scripts write figures to `examples/figures/`
relative to the *original* repo layout; `run_tests.py` creates the dir on
demand.
