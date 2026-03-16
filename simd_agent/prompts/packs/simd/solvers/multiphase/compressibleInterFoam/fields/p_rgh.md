# compressibleInterFoam — 0/p_rgh

**Dimensions**: `[1 -1 -2 0 0 0 0]`

`p_rgh` is the gravity-modified pressure field: p_rgh = p − ρ·g·h.

**Note**: interFoam (incompressible) uses `[0 2 -2 0 0 0 0]` — do NOT use that here.

compressibleInterFoam reads BOTH `0/p_rgh` and `0/p` at startup. Both MUST be generated.

## Initialisation
- `internalField uniform <initialDomainPressure>;`

Use a separate initial-domain pressure value when available.
If the user provides only an operating or reference pressure, that value may be used as the initial field.

## Boundary rules
- inlet with imposed mass-flow or velocity flux:
  - `type fixedFluxPressure;`
  - `value uniform <initialDomainPressure>;`
- outlet with specified static pressure:
  - `type fixedValue;`
  - `value uniform <outletPressure>;`
- wall:
  - `type fixedFluxPressure;`
  - `value uniform <initialDomainPressure>;`
- frontAndBack:
  - `type empty;`
- symmetry:
  - `type symmetry;`

## Constraints
- Keep `p_rgh` consistent with gravity and the prescribed flux boundary conditions.
- Use the specified outlet pressure when the user provides one.
- Set `0/p` internalField equal to `0/p_rgh` internalField at t=0.
