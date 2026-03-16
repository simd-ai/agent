# compressibleMultiphaseInterFoam — 0/alphas

**ALWAYS required** — composite alpha field for N-phase solver.
**Dimensions**: `[0 0 0 0 0 0 0]`
**internalField**: `uniform 0`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `zeroGradient` |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
