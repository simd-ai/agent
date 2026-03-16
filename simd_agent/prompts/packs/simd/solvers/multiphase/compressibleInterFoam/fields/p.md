# compressibleInterFoam — 0/p

**Dimensions**: `[1 -1 -2 0 0 0 0]`

Absolute pressure field. compressibleInterFoam reads `0/p` at startup with MUST_READ. **This file is REQUIRED** — missing it causes `FOAM FATAL ERROR: cannot find file "0/p"`.

## Initialisation
- `internalField uniform <initialDomainPressure>;`

## Boundary rules
- inlet:
  - `type zeroGradient;`
- outlet:
  - `type fixedValue;`
  - `value uniform <outletPressure>;`
- wall:
  - `type zeroGradient;`
- frontAndBack:
  - `type empty;`
- symmetry:
  - `type symmetry;`

## Constraints
- Keep `p` consistent with `p_rgh`, gravity treatment, and the specified outlet condition.
- Use the same internalField value as `0/p_rgh` (both equal to domain pressure at t=0).
- The outlet fixedValue must match the configured outlet pressure.
