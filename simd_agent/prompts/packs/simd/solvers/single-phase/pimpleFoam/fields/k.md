# pimpleFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m²/s²)

Use `CaseSpec.turbulence_initial_values.k` when available.
Formula: `k = 1.5 × (I × |U|)²`, I ≈ 0.05 for internal flows.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `zeroGradient` or `inletOutlet` |
| wall | `kqRWallFunction` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

```
internalField   uniform <k_value>;

boundaryField
{
    inlet  { type fixedValue; value uniform <k_value>; }
    outlet { type zeroGradient; }
    walls  { type kqRWallFunction; value uniform <k_value>; }
}
```
