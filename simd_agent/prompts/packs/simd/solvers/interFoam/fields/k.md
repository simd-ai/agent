# interFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m²/s²)
Generate only when turbulence is active.

Use `CaseSpec.turbulence_initial_values.k` when available.
Formula: `k = 1.5 × (I × |U|)²`, I ≈ 0.05.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <k_value>` |
| outlet | `zeroGradient` |
| wall | `kqRWallFunction` + `value uniform <k_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
