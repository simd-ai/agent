# pimpleFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)

Use `CaseSpec.turbulence_initial_values.epsilon` when available.
Formula: `ε = Cμ^0.75 × k^1.5 / L`, Cμ = 0.09.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `zeroGradient` |
| wall | `epsilonWallFunction` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
