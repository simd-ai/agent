# rhoPimpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)

## Value — MUST be consistent with the computed k value

Use `CaseSpec.turbulence_initial_values.omega` when pre-computed and available.

Otherwise compute from k and the hydraulic length scale:

```
omega = sqrt(k) / (Cmu^0.25 × l)
```

where:
- `Cmu = 0.09` (standard kOmegaSST constant)
- `l = 0.07 × D` (turbulent length scale for fully-developed pipe flow; D = hydraulic diameter)

**Example — LN2 pipe flow (continuing from k.md example):**
- k ≈ 1.8e-4, D = 0.025 m → l = 0.07 × 0.025 = 1.75e-3 m
- omega = sqrt(1.8e-4) / (0.09^0.25 × 1.75e-3) ≈ 0.01342 / (0.5477 × 1.75e-3) ≈ 14

For low-speed cryogenic pipe flow, omega is typically **10–30 s⁻¹**.
A value of 100 s⁻¹ paired with a tiny k is inconsistent — the k/omega ratio sets nut = k/omega;
an over-estimated omega gives nut ≈ 0 and under-damps turbulence production.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <omega_value>` |
| outlet | `zeroGradient` |
| wall | `omegaWallFunction` + `value uniform <omega_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
