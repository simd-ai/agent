# compressibleMultiphaseInterFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]`
**internalField**: `uniform (0 0 0)`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform (<vx> <vy> <vz>)` |
| outlet | `zeroGradient` |
| wall | `noSlip` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
