# pimpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)

Use `CaseSpec.turbulence_initial_values.omega` when available.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `zeroGradient` |
| wall | `omegaWallFunction` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
