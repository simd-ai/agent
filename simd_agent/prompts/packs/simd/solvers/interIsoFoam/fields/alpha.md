# interIsoFoam — 0/alpha.<phase1Name>

**Dimensions**: `[0 0 0 0 0 0 0]` (volume fraction)
**internalField**: `uniform 0` or `uniform 1` depending on initial condition.

File name uses exact phase name from CaseSpec (e.g. `0/alpha.water`).

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <alpha_value>` |
| outlet | `zeroGradient` |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
