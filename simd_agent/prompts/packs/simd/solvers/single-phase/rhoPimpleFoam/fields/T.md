# rhoPimpleFoam — 0/T

**Dimensions**: `[0 0 0 1 0 0 0]` (Kelvin)
**NEVER generate `0/h` or `0/e`** — thermo initialises energy from `0/T`.

## internalField

**CRITICAL**: Use `inlet_temperature` from the case spec. NEVER default to 300 K.

Reason: icoPolynomial EOS → ρ(T) = a0 + a1·T. For LN2 (inlet=77K): ρ = 1167.9 − 4.7×T.
At T=300K: ρ = −242 kg/m³ (negative) → SIGFPE on iteration 0, before any limiter runs.
The internalField sets the initial density field; a wrong value here causes immediate divergence.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <inlet_temperature>` |
| outlet | `zeroGradient` |
| wall (adiabatic / no heat transfer) | `zeroGradient` |
| wall (user-specified isothermal temperature) | `fixedValue` + `value uniform <wall_temperature>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

## Adiabatic vs isothermal wall — CRITICAL distinction

**Do NOT convert "no heat transfer" into a fixedValue wall BC.**

If the user says the wall is adiabatic, insulated, or there is no heat transfer → use `zeroGradient`.
If the user explicitly specifies a wall temperature (e.g. "wall at 400 K") → use `fixedValue`.

Setting wall T to `fixedValue` equal to the fluid temperature is not equivalent to adiabatic.
It forces the solver to maintain that exact temperature at the wall face every time step,
which actively suppresses any natural temperature variation near the wall (even from numerical noise)
and can produce an artificial Neumann-Dirichlet conflict. Use `zeroGradient` for adiabatic walls.

## CRITICAL — EOS ceiling constraint on fixedValue temperatures

When the EOS is `icoPolynomial`, `ρ(T) = a0 + a1·T` becomes zero at `T_ceiling = a0/|a1|`.
**Any fixedValue BC temperature above this ceiling gives negative density → SIGFPE in turbulence model.**

`fvOptions limitTemperature` only clamps internal cell values — **boundary face values are NEVER clamped by limitTemperature.**
A wall fixedValue of 400K still evaluates to ρ < 0 at that face even if fvOptions says max=200K.

**Rule**: If `fv_options_eos_t_ceiling` is provided and a wall_temperature exceeds it, cap the fixedValue to `fv_options_eos_t_ceiling × 0.9` for the BC.
Write a comment in the file explaining the cap.
