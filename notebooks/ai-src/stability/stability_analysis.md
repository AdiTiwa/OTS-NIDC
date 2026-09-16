# Von Neumann Stability Analysis — OTS‑NIDC (p=1)

**Document:** Stability sketch for the OTS‑NIDC scheme
**Question addressed (Kevin Chu / Velexi):** *Do the NIDC correction terms
negatively impact the stability of the forward‑Euler time stepping?*

**Short answer:** **No.** For the physically relevant scalar family
(ασ² ≥ 0, i.e. moment growth), the p=1 NIDC correction *very slightly
*shrinks* the grid‑scale (diffusion‑limited) stable region. The shrink
factor is `1 − η/18 + η²/144 ≈ 1 − (ασ²Δx²)/(18D)`, i.e. O(Δx²) and
typically < 2% at practical resolutions. NIDC is comfortably stable at
`Δt_opt = Δx²/(6D)` — in fact it is stable up to a CFL bound that is
≈1.6% *narrower* than plain Forward Euler for the tested parameters, which
is negligible. **The correction does not destabilize the scheme at the
optimal step.**

> **Sign convention note (2026‑07‑09 correction):** an earlier draft of this
> analysis used the *opposite* sign for `C^(1)` and concluded the correction
> *expands* the stable region. The corrected `OTS‑NIDC.md` Eq. 5 (and the
> verified implementation `scalar_nidc.py`) give `C^(1)` with a **+** sign,
> which is what yields the observed 4th‑order convergence. This document now
> follows the corrected sign. The correction's effect on the CFL is small and
> of the same O(Δx²) magnitude either way; only its direction flips.

---

## 1. Setup

Scalar family (OTS‑NIDC §2, Eq. 1):

```
u_t = D u_xx + α σ² u ,     α = k(k−1)/2 ≥ 0 .
```

Discretization — Forward Euler in time, 3‑point centered difference in space
(with `δ_x² u_j = (u_{j+1} − 2u_j + u_{j−1})/Δx²`):

```
u_j^{n+1} = u_j^n + Δt [ D δ_x² u_j^n + ασ² u_j^n ] .
```

### OTS‑NIDC corrected update (p=1, corrected §5, Eq. 5)

`Δt_opt = Δx²/(6D)`. The first‑level correction (itself O(Δx⁴)) is

```
C_j^{(1)} = + (ασ² Δx⁴)/(36 D) δ_x² u_j^n  +  (α²σ⁴ Δx⁴)/(72 D²) u_j^n .
```

The corrected scheme:

```
u_j^{n+1} = u_j^n + Δt_opt [ D δ_x² u_j^n + ασ² u_j^n ] + C_j^{(1)} .
```

---

## 2. von Neumann ansatz

Set `u_j^n = Gⁿ e^{i k j Δx}`. Then

```
δ_x² u_j^n  →  (e^{ikΔx} − 2 + e^{−ikΔx})/Δx² u_j^n
            =  − (2/Δx²) sin²(kΔx/2) u_j^n .
```

Define

```
s   = sin²(kΔx/2) ∈ [0, 1] ,
μ   = D Δt / Δx²               (diffusion number) ,
η   = ασ² Δx² / D     ∈ O(Δx²) (reaction number, ≥ 0 for moment growth) .
```

### 2a. Plain Forward Euler

```
G_FE(k) = 1 − 4 μ s + μ η .                         (FE)
```

- `s = 0` (k = 0, mode of homogeneous growth): `G_FE(0) = 1 + μη > 1`.
  This is the **true physical growth factor** of the moment, not numerical
  instability — every consistent scheme that treats the reaction explicitly
  has `|G(0)|>1` for ασ²>0.
- `s = 1` (k = π/Δx, grid scale): `G_FE(1) = 1 − 4μ + μη`. This mode is the
  one that can cross `G < −1` and cause **oscillatory numerical blow‑up**, so
  it governs the diffusion CFL.

**FE stability (diffusion‑limited):** `|G_FE(1)| ≤ 1` ⇒ `−1 ≤ 1 − 4μ + μη ≤ 1`
⇒ `4μ − μη ≤ 2` ⇒

```
μ ≤ 2 / (4 − η)  ≈  1/2 + η/8 + …          (for small η) .
```

Dropping the small η‑shift (η = O(Δx²) ≪ 1) gives the familiar **grid CFL**
`μ ≤ 1/2`, i.e.

```
Δt ≤ Δx² / (2 D) .                              (FE CFL)
```

The mode `k=0` never imposes a stricter bound because `1+μη > 1` is physical
growth, not a violation of the *numerical* stability criterion for a
diffusion scheme. (Formally, the strict `|G|≤1 ∀k` criterion is only
appropriate for non‑growing problems; for ασ²>0 one examines the
high‑wavenumber / oscillatory modes, which is the standard reading of the
project's "Δt ≤ Δx²/(2D) roughly.")

### 2b. OTS‑NIDC p=1 (corrected sign)

Substitute `Δt = Δt_opt` and add `C_j^{(1)}` in Fourier space. The correction
terms become

```
+ (ασ² Δx⁴)/(36 D) · (−2/Δx²) s   =  − (ασ² Δx²)/(18 D) s
                                    =  − (η/9) s ,
+ (α²σ⁴ Δx⁴)/(72 D²)              =  + η²/72 .
```

Hence the corrected amplification factor is

```
G_NIDC(k) = 1 − 4 μ s + μ η  − (η/9) s + η²/72 .          (NIDC)
```

Compare with (FE): the NIDC correction adds a **k‑dependent term
`−(η/9) s ≤ 0`** (which *lowers* the grid‑scale mode since `s` is largest
there) and a **uniform positive offset `+η²/72`** (which raises every mode
slightly). The net effect on the grid‑scale boundary is a small *shrink*.

#### Effect on the stability boundary (grid scale, s = 1)

```
G_NIDC(1) = 1 − 4μ + μη − η/9 + η²/72 ≥ −1
         ⇒  4μ − μη ≤ 2 − η/9 + η²/72
         ⇒  μ ≤ (2 − η/9 + η²/72) / (4 − η) .
```

**NIDC diffusion‑limited CFL:**

```
μ_NIDC ≤ (2 − η/9 + η²/72) / (4 − η)
       ≈ 1/2 − η/8 − η/9·...
```

**Shrink factor relative to FE:**

```
factor = μ_NIDC,max / μ_FE,max
       = (2 − η/9 + η²/72) / 2
       = 1 − η/18 + η²/144
       ≈ 1 − (ασ² Δx²)/(18 D)        (for small η) .
```

- Because `η = ασ²Δx²/D ≥ 0` for the moment equations, the factor is
  **≤ 1**: the correction *shrinks* the stable region very slightly.
- The shrink is `O(Δx²)` and tiny: at `η = 0.3` (already a coarse,
  large‑η discretization), `factor ≈ 0.984` (−1.6%). As `Δx → 0`,
  `η → 0` and NIDC → FE exactly (`factor → 1`), as it must since the
  correction is itself O(Δx⁴).
- **Sign caveat:** if `η < 0` (a *decaying* reaction, ασ²<0), the
  `−(η/9)s` term becomes positive at the grid scale and the factor becomes
  `1 − η/18 + η²/144 > 1`: a *slight expansion* (still only O(|η|)). The
  project's moment equations all have ασ² ≥ 0, so we are in the
  shrink case.

### 2c. Coupled second‑moment case (tractable)

For the second‑moment system (§9, Eq. V) the diffusion coefficient is the
pair sum `D_ij = D_i + D_j`, so the optimal step is

```
Δt^{(ij)}_opt = Δx² / [6 (D_i + D_j)] .                 (§9.7)
```

The von Neumann amplification factor for the pair‑(i,j) equation under the
same ansatz is formally identical to the scalar one with the replacement
`D → D_ij`:

```
G_V(k)   = 1 − 4 μ_V s + μ_V η_V ,          μ_V = D_ij Δt/Δx² ,
G_V^NIDC(k) = 1 − 4 μ_V s + μ_V η_V − (η_V/9) s + η_V²/72 ,
```

where `η_V = α_V σ_V² Δx² / D_ij` and `α_V, σ_V` are the relevant
reaction/noise couplings for `V^{(ij)}`. The stability boundary is therefore

```
μ_V ≤ 2/(4 − η_V)            (FE / V) ,
μ_V ≤ (2 − η_V/9 + η_V²/72)/(4 − η_V)   (NIDC / V) ,
```

with the **same shrink factor** `1 − η_V/18 + η_V²/144`. Concretely, with
all `D_i = D` the second‑moment step is `Δt_V = Δx²/(12D)` and the diffusion
number at the optimal step is `μ_V = 1/12`, well inside the CFL bound
`1/2`. The NIDC correction again leaves stability essentially unchanged
(slightly narrower). The 2:1 separate‑stepping resolution (§10) keeps each
family on its own OTS step, so no cross‑family CFL coupling is introduced.

---

## 3. Stability‑region comparison table (FE vs NIDC p=1, scalar)

| Quantity | Plain Forward Euler | OTS‑NIDC p=1 | Relation |
|---|---|---|---|
| Amplification `G(k)` | `1 − 4μs + μη` | `1 − 4μs + μη − (η/9)s + η²/72` | NIDC adds O(η)=O(Δx²) terms |
| Grid‑scale bound `μ_max` | `2/(4−η)` | `(2 − η/9 + η²/72)/(4−η)` | NIDC ≤ FE for η≥0 |
| Coarse CFL (`η→0`) | `Δt ≤ Δx²/(2D)` | `≈ Δt ≤ (1−η/9)·Δx²/(2D)` | ~same |
| Shrink factor | 1 | `1 − η/18 + η²/144` | **< 1 for η>0** |
| `Δt_opt = Δx²/(6D)` (`μ=1/6`) | stable (0.1667 < 0.5) | stable | **both inside** |
| Where it first breaks (η=0.3) | `μ≈0.5405` | `μ≈0.5319` | NIDC breaks ~1.6% *earlier* |

Representative numbers for `η = 0.30` (`Δx=ΔD=1` so `Δt_opt=1/6`, coarse grid):

```
μ_FE,max   = 0.540541   ⇒  Δt ≤ 0.5405·Δx²/D
μ_NIDC,max = 0.531869   ⇒  Δt ≤ 0.5319·Δx²/D
factor     = 0.983958   ⇒  −1.60 % narrower stable region for NIDC
```

---

## 4. Numerical check (`stability_check.py`)

The script evaluates `G_FE` and `G_NIDC` on `s = sin²(kΔx/2)`, `kΔx/π ∈ [0,1]`,
for `Δt ∈ {0.5, 1, 1.5, 2}·Δt_opt`, and plots them. Key printed output (η=0.30):

```
dt/dt_opt   mu    max|G_FE|  max|G_NIDC|   FE_grid  NIDC_grid
   0.50   0.0833     1.0250     1.0262      0.6917   0.6596
   1.00   0.1667     1.0500     1.0513      0.3833   0.3513
   1.50   0.2500     1.0750     1.0762      0.0750   0.0429
   2.00   0.3333     1.1000     1.1013     −0.2333  −0.2654
```

Reading the numbers:
- `max|G| > 1` at every row is the **k=0 physical‑growth** contribution
  (`1 + μη`), not numerical instability — see §2a. The relevant numerical
  (oscillatory) stability is read off the **grid‑scale** column
  (`FE_grid = G_FE(1)`, `NIDC_grid = G_NIDC(1)`).
- `FE_grid` crosses `−1` when `μ ≈ 0.5405` and `NIDC_grid` when
  `μ ≈ 0.5319`. **NIDC stays oscillatory‑stable slightly *less* long** — it
  first breaks at `Δt ≈ 0.5319·Δx²/D`, i.e. ~1.6% *before* the FE break
  point and ~3.2× the OTS optimal step.
- At `Δt_opt` (`dt/dt_opt = 1`, `μ = 0.1667`) both grid‑scale values are
  `≈ +0.38/+0.35`, far from the `−1` danger zone. **Confirmed: NIDC is
  stable at `Δt_opt`.**

The plot `stability_plot.png` shows (left) `|G|` vs wavenumber for all four
`Δt` values and (right) the grid‑scale mode `G(k=π/Δx)` vs `μ` with the two
CFL floors marked.

---

## 5. CFL discussion and connection to `Δt_opt`

- **Plain FE CFL:** `Δt ≤ Δx²/(2D)` (from the grid‑scale oscillatory mode,
  dropping the O(Δx²) reaction shift).
- **NIDC effective CFL:** `Δt ≲ (1 − η/9)·Δx²/(2D)` — essentially the same,
  marginally narrower for η > 0.
- **OTS optimal step:** `Δt_opt = Δx²/(6D)`. Since `1/6 < 1/2`, `Δt_opt`
  lies **inside the FE stable region** with a comfortable safety factor of 3.
  The NIDC correction does not move the optimal step and does not push it
  anywhere near the CFL ceiling — it only trims the ceiling by ~1.6%.

---

## 6. CONCLUSION — answering Kevin's question

> *"Do the NIDC correction terms negatively impact the forward‑Euler
> stability?"*

**No — not in any practical sense.** The p=1 NIDC correction terms do not
destabilize forward‑Euler stepping. Their effect on the von Neumann
amplification factor is the addition of `−(η/9) sin²(kΔx/2) + η²/72`, which
for the physically relevant case `η = ασ²Δx²/D ≥ 0` (moment growth) **slightly
*shrinks*** the grid‑scale (diffusion‑limited) stable region by a factor

```
factor = 1 − η/18 + η²/144  ≈  1 − (ασ²Δx²)/(18D)   < 1   (η ≥ 0) .
```

This is an O(Δx²) effect — typically < 2% at practical resolutions — and
vanishes as `Δx → 0`, consistent with the correction being itself O(Δx⁴).
Numerically (η = 0.30): FE breaks at `μ ≈ 0.5405`, NIDC at `μ ≈ 0.5319`,
a shrink of **−1.6%**. Both schemes are rock‑solid at the OTS optimal
step `Δt_opt = Δx²/(6D)` (μ = 1/6 ≈ 0.167, ~3× inside the CFL bound).

If `η < 0` (pure decay), the correction produces a symmetric tiny *expansion*
of the same O(Δx²) magnitude — still negligible and never destabilizing in
any practical sense. **Bottom line: the NIDC correction is stability‑neutral
to first order and marginally *tightening* (slightly narrowing the stable
region) for the growing moment equations of the stochastic heat equation —
but it never threatens stability at the optimal step.**
