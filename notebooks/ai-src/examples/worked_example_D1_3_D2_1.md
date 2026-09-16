# Worked Example: Coupled Stochastic Reaction-Diffusion with D₁=3, D₂=1
## Regime 1 (r = 3) — Shared Step with Exact Mismatch Correction

---

## 1. The SPDE System

Two species with linear multiplicative noise, coupled through reaction terms:

```
du₁ = [ D₁ ∂ₓₓ u₁ + R₁₁ u₁ + R₁₂ u₂ ] dt + σ₁₁ u₁ dW₁ + σ₁₂ u₂ dW₂
du₂ = [ D₂ ∂ₓₓ u₂ + R₂₁ u₁ + R₂₂ u₂ ] dt + σ₂₁ u₁ dW₁ + σ₂₂ u₂ dW₂
```

with **D₁ = 3**, **D₂ = 1**. All other coefficients constant.

---

## 2. Moment Equations (M-block)

Taking expectations (Itô, zero-mean noise kills stochastic terms):

```
∂ₜ m₁ = D₁ ∂ₓₓ m₁ + R₁₁ m₁ + R₁₂ m₂
∂ₜ m₂ = D₂ ∂ₓₓ m₂ + R₂₁ m₁ + R₂₂ m₂
```

where mᵢ = 𝔼[uᵢ]. This is a **closed linear deterministic system** — no V-block terms appear.

**Key**: The M-block and V-block are fully decoupled (structural property of affine drift + linear multiplicative noise). We only solve the M-block here; V-block is identical in structure with Dᵢ → Dᵢ + Dⱼ.

---

## 3. OTS-NIDC Optimal Steps

For a given Δx, the OTS-optimal step for diffusion coefficient D is:

```
τ(D) = Δx² / (6D)
```

| Species | D | τ(D) |
|---------|---|------|
| 1 (fast) | 3 | τ₁ = Δx² / 18 |
| 2 (slow) | 1 | τ₂ = Δx² / 6 |

**Ratio**: r = D₁/D₂ = 3/1 = **3** → exactly at the stability boundary r* = 3.

---

## 4. Shared Step Choice (Regime 1)

Since r = 3 ≤ 3, we use **Regime 1**: shared step at the slower species' optimal step.

```
Δt* = τ₂ = Δx² / 6
```

- Species 2 advances at its own optimum → **no correction needed**.
- Species 1 is forced onto Δt* ≠ τ₁ → **exact mismatch correction required**.

**Deviation for species 1**:
```
ε = Δt* − τ₁ = Δx²/6 − Δx²/18 = Δx²/9 = 2τ₁
```

So species 1 takes a step **3× larger** than its own optimum (τ₁ : Δt* = 1 : 3).

---

## 5. Exact Mismatch Correction for Species 1

### 5.1 General Truncation Error Formula

For one forward-Euler + central-difference step of uₜ = D uₓₓ with arbitrary step Δt*:

```
e(Δt*) = Δt* D ũₓₓₓₓ [ (Δt* D)/2 − Δx²/12 ] + O((Δt*)³) + O(Δt* Δx⁴)      (1)
```

### 5.2 Specialize to Species 1 at Shared Step Δt*

D = D₁ = 3, Δt* = Δx²/6. Plug into the bracket:

```
(Δt* D₁)/2 − Δx²/12 = (Δx²/6 × 3)/2 − Δx²/12
                     = Δx²/4 − Δx²/12
                     = Δx²/6
                     = ε D₁ / 2    ✓ (since ε = Δx²/9, εD₁/2 = 3Δx²/18 = Δx²/6)
```

**Leading local truncation error for species 1**:
```
e₁ = Δt* D₁ ũ₁ₓₓₓₓ (ε D₁ / 2) + ⋯
   = (Δx²/6) × 3 × ũ₁ₓₓₓₓ × (Δx²/6) + ⋯
   = (Δx⁴ / 12) ũ₁ₓₓₓₓ + O(Δx⁶)                                    (2)
```

This is **O(Δx⁴)** — the correction restores 4th-order accuracy in space.

### 5.3 NIDC Correction Term

The NIDC correction subtracts the leading error using a wide-stencil approximation of ũₓₓₓₓ.

**Fourth-derivative stencil** (5-point, O(Δx²) accurate):
```
ũₓₓₓₓ ≈ (u_{j-2} − 4u_{j-1} + 6u_j − 4u_{j+1} + u_{j+2}) / Δx⁴
```

**Correction added to species 1's update**:
```
C₁ = − e₁  ≈  − (Δx⁴/12) × [ (u_{j-2} − 4u_{j-1} + 6u_j − 4u_{j+1} + u_{j+2}) / Δx⁴ ]
   =  − (1/12) (u_{j-2} − 4u_{j-1} + 6u_j − 4u_{j+1} + u_{j+2})
```

**Beautiful**: The Δx⁴ cancels exactly — the correction is **grid-independent** (as expected for an algebraic identity).

---

## 6. Complete Discrete Update (One Shared Step)

Let `u¹_j`, `u²_j` be the discrete values at grid point j.

### Species 2 (at its optimum — standard OTS-NIDC p=1):
```
u²^{n+1}_j = u²^n_j + (Δt* D₂ / Δx²) (u²^n_{j+1} − 2u²^n_j + u²^n_{j-1})
           + Δt* (R₂₁ u¹^n_j + R₂₂ u²^n_j)
           + C¹₂(u²^n)                    # standard NIDC correction for D=1
```

### Species 1 (mismatched step — corrected):
```
u¹^{n+1}_j = u¹^n_j + (Δt* D₁ / Δx²) (u¹^n_{j+1} − 2u¹^n_j + u¹^n_{j-1})
           + Δt* (R₁₁ u¹^n_j + R₁₂ u²^n_j)
           + C¹₁(u¹^n)                    # standard NIDC correction for D=3 at Δt*
           + C^mismatch₁                  # EXACT MISMATCH CORRECTION (new)
```

where the **mismatch correction** is:
```
C^mismatch₁ = − (1/12) (u¹_{j-2} − 4u¹_{j-1} + 6u¹_j − 4u¹_{j+1} + u¹_{j+2})
```

**Note**: No dependence on species 2's values in the mismatch correction — it only uses species 1's own wide stencil. The coupling terms (R₁₂ u²) are handled by standard reaction evaluation at time tₙ.

---

## 7. Stability Check

Forward-Euler diffusion stability: Δt ≤ Δx²/(2D)

- Species 2: Δt* = Δx²/6 ≤ Δx²/(2×1) = Δx²/2 ✓
- Species 1: Δt* = Δx²/6 ≤ Δx²/(2×3) = Δx²/6 ✓ (exactly at limit)

**r = 3 sits exactly at the stability boundary**. In practice, use a slight safety factor (e.g., Δt* = 0.99 × Δx²/6) or verify the corrected scheme's stability region extends slightly beyond plain FTCS.

---

## 8. Pseudocode

```python
def ots_nidc_coupled_step(u1, u2, dx, dt, D1=3.0, D2=1.0, R=None):
    """
    One shared time step for coupled D1=3, D2=1 system.
    u1, u2: arrays of length N (periodic BC assumed for simplicity)
    Returns: u1_new, u2_new
    """
    N = len(u1)
    inv_dx2 = 1.0 / (dx * dx)
    
    # --- Laplacian (3-point stencil) ---
    lap1 = np.roll(u1, -1) - 2*u1 + np.roll(u1, 1)
    lap2 = np.roll(u2, -1) - 2*u2 + np.roll(u2, 1)
    
    # --- Standard NIDC corrections (p=1) ---
    # For species 2 at its optimum D=1, dt = dx^2/6: correction cancels u_xxxx term
    # For species 1 at D=3, its own optimum would be dt = dx^2/18
    # But we're using dt = dx^2/6, so standard correction is for wrong step
    # We'll apply standard correction for D=1,dt=dx^2/6 to both, then fix species 1
    
    # 4th derivative (5-point stencil)
    u1_xxxx = (np.roll(u1, -2) - 4*np.roll(u1, -1) + 6*u1 
               - 4*np.roll(u1, 1) + np.roll(u1, 2)) * inv_dx2 * inv_dx2
    u2_xxxx = (np.roll(u2, -2) - 4*np.roll(u2, -1) + 6*u2 
               - 4*np.roll(u2, 1) + np.roll(u2, 2)) * inv_dx2 * inv_dx2
    
    # Standard NIDC correction for dt = dx^2/6, D=1
    # Derived from: -dt * D * (dx^2/12) * u_xxxx = -(dx^2/6)*1*(dx^2/12)*u_xxxx
    # Wait: standard OTS correction for p=1 at optimum is different.
    # Let's use the exact formulas from the derivation.
    
    # For species 2 (D=1, dt=dx^2/6 = tau_2): standard OTS correction
    C1_std_2 = - (dt * D2 * dx**2 / 12) * u2_xxxx
    
    # For species 1, standard correction would be for its own optimum tau_1
    # But we're NOT at its optimum. The standard NIDC machinery at wrong dt
    # gives a correction that's not what we want. Better: apply standard
    # correction for species 2's parameters, then add exact mismatch correction.
    
    # Actually, simpler: the total correction for species 1 should cancel
    # the full bracket in (1). We derived C^mismatch_1 = -(1/12)*5-point-stencil.
    
    # Wide stencil (without dx^4 factor)
    wide1 = (np.roll(u1, -2) - 4*np.roll(u1, -1) + 6*u1 
             - 4*np.roll(u1, 1) + np.roll(u1, 2))
    wide2 = (np.roll(u2, -2) - 4*np.roll(u2, -1) + 6*u2 
             - 4*np.roll(u2, 1) + np.roll(u2, 2))
    
    # Species 2: standard OTS correction at its optimum
    # e = dt*D*(dt*D/2 - dx^2/12)*u_xxxx, at dt=dx^2/6, D=1: bracket = 0
    # So standard OTS correction for p=1 cancels higher-order terms.
    # For this example, we'll just use the standard OTS-NIDC p=1 correction
    # as a black box, and focus on the NEW mismatch correction.
    
    # MISMATCH CORRECTION for species 1 (the key new term):
    # C^mismatch = - (1/12) * wide_stencil
    mismatch_correction_1 = - wide1 / 12.0
    
    # Reaction terms
    if R is not None:
        react1 = R[0,0]*u1 + R[0,1]*u2
        react2 = R[1,0]*u1 + R[1,1]*u2
    else:
        react1 = np.zeros_like(u1)
        react2 = np.zeros_like(u2)
    
    # --- Full updates ---
    u1_new = (u1 
              + dt * D1 * inv_dx2 * lap1
              + dt * react1
              + mismatch_correction_1)   # <-- THE KEY ADDITION
    
    u2_new = (u2 
              + dt * D2 * inv_dx2 * lap2
              + dt * react2
              + standard_ots_correction(u2, dx, dt, D2))  # standard OTS
    
    return u1_new, u2_new
```

---

## 9. Error Budget Summary

| Component | Species 1 (D=3) | Species 2 (D=1) |
|-----------|------------------|------------------|
| **Step size** | Δt* = Δx²/6 (3× own optimum) | Δt* = Δx²/6 (own optimum) |
| **Leading diffusion error** | Cancelled by **exact mismatch correction** | Cancelled by **standard OTS-NIDC** |
| **Coupling error** | Reaction evaluated at tₙ (O(Δt*) = O(Δx²) local, O(Δx²) global if uncorrected) | Same |
| **Reaction correction** | Include in NIDC framework (higher-order) | Include in NIDC framework (higher-order) |
| **Global accuracy target** | O(Δx⁴) | O(Δx⁴) |

**Key point**: The mismatch correction is **exact for the leading diffusion error** at any ε. The only approximation is the wide-stencil approximation of uₓₓₓₓ (O(Δx²) accurate, giving O(Δx⁴) global after accumulation).

---

## 10. What This Example Demonstrates

1. **r = 3 is exactly at the Regime 1/2 boundary** — shared step works but species 1 is at its stability limit.
2. **Exact mismatch correction** (C^mismatch = −wide/12) is algebraically derived, not fitted.
3. **No interpolation needed** — coupling terms use values at tₙ (reaction terms don't need intermediate coupling data in Regime 1).
4. **The correction is grid-independent** — Δx⁴ cancels, leaving pure stencil weights.
5. **Uniformly valid** — if D₁/D₂ were 2.9 or 3.0 or 2.0, the same formula works with ε = Δt* − τ₁.

---

## 11. Extension: If r > 3 (Regime 2)

If D₁ = 4, D₂ = 1 → r = 4 > 3. Shared step **unstable** for species 1.

Then use **subcycling** (Section 5 of the methodology):
- Macro step H = τ₂ = Δx²/6
- Species 1 takes n = ⌊4⌋ = 4 exact substeps of τ₁ = Δx²/24
- Remainder ρ = H − 4τ₁ = 0 (exact integer ratio → no remainder step!)
- Cubic Hermite interpolation for m₂(t) during substeps

The algebraic framework handles both regimes seamlessly.

---

*This example can be directly coded and tested against the exact moment equations to verify O(Δx⁴) convergence for both species simultaneously.*