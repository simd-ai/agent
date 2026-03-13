# compressibleInterFoam — 0/p_rgh

**Dimensions**: `[1 -1 -2 0 0 0 0]` (Pa — compressibleInterFoam uses DIMENSIONAL pressure)
**Note**: interFoam (incompressible) uses `[0 2 -2 0 0 0 0]` — do NOT use that here.
**internalField**: `uniform <operating_pressure>` (Pa) — same value as 0/p

compressibleInterFoam reads BOTH `0/p_rgh` and `0/p` at startup. Both MUST be generated.
- `p_rgh` = p − ρ·g·h (modified pressure)
- `p` = absolute pressure (generates separately — see fields/p.md)

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` + `value uniform <p_outlet>` |
| wall | `fixedFluxPressure` + `value uniform <p_internal>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

## Notes

- Use operating pressure (e.g. 101325 Pa or configured value) for internalField and outlet.
- `fixedFluxPressure` on walls allows the pressure gradient to adjust to the flux boundary condition.
- Set `0/p` internalField = `0/p_rgh` internalField (both equal to operating pressure at t=0).
