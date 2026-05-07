# Outlet pressure from total pressure (`totalPressure`)

## When to use
Use this when the user specifies:
- total pressure
- stagnation pressure
- wording like `p0`

This is not the same as specifying static outlet pressure.

## Purpose
This BC computes the patch static pressure from the provided total pressure and the local flow state.

## UI fields to expose
- `totalPressure` (stored as `p0`, in Pa)
- flow regime hint:
  - `subsonicIncompressible`
  - `subsonicCompressible`
  - `transonicCompressible`
  - `supersonicCompressible`
- optional advanced:
  - `rhoField`
  - `psiField`
  - `gamma`

## OpenFOAM mapping
Primary BC family: `type: totalPressure`

Common entries:
- `p0`
- `value`

Optional / conditional:
- `U`
- `phi`
- `rho`
- `psi`
- `gamma`

## Planner rule
If the user explicitly requests outlet total pressure:
- choose this BC family directly
- do not downgrade to simple fixed static pressure
- only use plain `fixedValue` on pressure when the user asked for static pressure, not total pressure

## Solver-awareness rule
The patch agent should detect solver class:
- incompressible: only `p0` and `value` needed
- compressible: may also need `rho`, `psi`, or `gamma`
- transonic/supersonic: include additional mode-specific entries

## Output policy
The UI should show:
- `pressureMode = total`
- `uiPrimaryInput = p0`

The final merged spec should preserve that this is a total-pressure condition, not just a raw static pressure value.
