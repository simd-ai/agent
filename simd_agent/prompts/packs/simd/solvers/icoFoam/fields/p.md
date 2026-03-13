# icoFoam — 0/p

**Dimensions**: `[0 2 -2 0 0 0 0]` (kinematic pressure, m²/s²)
**NOT Pa** — do not use `[1 -1 -2 0 0 0 0]`
**internalField**: `uniform 0`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` + `value uniform 0` |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

If ALL patches are `zeroGradient` (no `fixedValue p`), add `pRefCell 0; pRefValue 0;` to the `PISO {}` block in `fvSolution`.
