# compressibleInterIsoFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]`

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <omega_value>` |
| outlet | `zeroGradient` |
| wall | `omegaWallFunction` + `value uniform <omega_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
