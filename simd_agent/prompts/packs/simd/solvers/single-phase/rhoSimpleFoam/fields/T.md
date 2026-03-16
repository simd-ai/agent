# rhoSimpleFoam — 0/T

**Dimensions**: `[0 0 0 1 0 0 0]` (Kelvin — NOT Celsius)
**This is the ONLY energy initial condition file to generate.**
Do NOT generate `0/h` or `0/e` — the thermo package initialises the energy field from `0/T` at startup.

## internalField

**CRITICAL**: Use `inlet_temperature` from the case spec. NEVER default to 300 K.

Reason: icoPolynomial EOS → ρ(T) = a0 + a1·T. For LN2 (inlet=77K): ρ = 1167.9 − 4.7×T.
At T=300K: ρ = −242 kg/m³ (negative) → SIGFPE on iteration 0, before any limiter runs.
The internalField sets the initial density field; a wrong value here causes immediate divergence.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <T_inlet_K>` |
| outlet | `zeroGradient` | |
| wall (isothermal) | `fixedValue` | `value uniform <T_wall_K>` |
| wall (adiabatic) | `zeroGradient` | |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Rules

- Values in **Kelvin** — if user provides Celsius, convert: K = °C + 273.15
- Use `inlet_temperature` and `wall_temperature` from CaseSpec when available
- Default wall BC is `zeroGradient` (adiabatic) when wall temperature is not specified

## CRITICAL — EOS ceiling constraint on fixedValue temperatures

When the EOS is `icoPolynomial`, `ρ(T) = a0 + a1·T` becomes zero at `T_ceiling = a0/|a1|`.
**Any fixedValue BC temperature above this ceiling gives negative density → SIGFPE in turbulence model.**

`fvOptions limitTemperature` only clamps internal cell values — **boundary face values are NEVER clamped by limitTemperature.**

**Rule**: If `fv_options_eos_t_ceiling` is provided and a wall_temperature exceeds it, cap the fixedValue to `fv_options_eos_t_ceiling × 0.9`. Write a comment explaining the cap.
