# compressibleInterFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]`
Generate only when turbulence is active.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <k_value>` |
| outlet | `zeroGradient` |
| wall | `kqRWallFunction` + `value uniform <k_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
