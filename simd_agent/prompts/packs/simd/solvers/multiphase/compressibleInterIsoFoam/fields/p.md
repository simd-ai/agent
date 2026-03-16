# compressibleInterFoam — 0/p

**Dimensions**: `[1 -1 -2 0 0 0 0]` (Pa — absolute pressure)
**internalField**: `uniform <operating_pressure>` (Pa) — same value as 0/p_rgh

compressibleInterFoam reads `0/p` at startup with MUST_READ. **This file is REQUIRED**.
Missing it causes: `FOAM FATAL ERROR: cannot find file "0/p"`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` + `value uniform <p_outlet>` |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

## Notes

- Use the same operating pressure for internalField as in `0/p_rgh`.
- `p` and `p_rgh` are related by p = p_rgh + ρ·g·h. At t=0 (before gravity acts), initialize both to the same value.
- The outlet fixedValue should match the configured operating/outlet pressure.
