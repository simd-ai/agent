# compressibleMultiphaseInterFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]`

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <epsilon_value>` |
| outlet | `zeroGradient` |
| wall | `epsilonWallFunction` + `value uniform <epsilon_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
