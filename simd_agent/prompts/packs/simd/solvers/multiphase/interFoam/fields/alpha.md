# interFoam — 0/alpha.<phase1Name>

**Dimensions**: `[0 0 0 0 0 0 0]` (dimensionless volume fraction)
**internalField**: depends on case — `uniform 0` (all phase2) or `uniform 1` (all phase1)

File name: `0/alpha.<phase1Name>` — use exact phase name from CaseSpec (e.g. `alpha.water`).
NEVER use `alpha.phase1` unless the phase name is literally `phase1`.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <alpha_value>` (0 or 1) |
| outlet | `zeroGradient` |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
