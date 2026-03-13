# interFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)
Generate only when turbulence is active.

Formula: `ε = Cμ^0.75 × k^1.5 / L`, Cμ = 0.09.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <epsilon_value>` |
| outlet | `zeroGradient` |
| wall | `epsilonWallFunction` + `value uniform <epsilon_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
