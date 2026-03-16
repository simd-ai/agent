# pimpleFoam — 0/p

**Dimensions**: `[0 2 -2 0 0 0 0]` (kinematic pressure, m²/s²)
**NOT Pa** — do not use `[1 -1 -2 0 0 0 0]`

**internalField**: `uniform 0`

## Typical BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` (uniform 0) |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

When no `fixedValue` pressure BC exists, add `pRefCell 0; pRefValue 0;` to the `PIMPLE {}` block in fvSolution.
