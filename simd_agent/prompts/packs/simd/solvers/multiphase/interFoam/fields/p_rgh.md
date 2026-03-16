# interFoam — 0/p_rgh

**Dimensions**: `[0 2 -2 0 0 0 0]` (kinematic pressure minus hydrostatic head)
**NOT Pa** — do not use `[1 -1 -2 0 0 0 0]`
**internalField**: `uniform 0`

Generate `0/p_rgh` — NOT `0/p`.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` + `value uniform 0` |
| wall | `fixedFluxPressure` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
