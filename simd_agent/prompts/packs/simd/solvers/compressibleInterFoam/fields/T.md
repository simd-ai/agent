# compressibleInterFoam — 0/T

**Dimensions**: `[0 0 0 1 0 0 0]`

## Initialisation
Temperature must always be explicitly initialised with a positive value (Kelvin).

**CRITICAL**: Use `inlet_temperature` from the case spec. NEVER default to 300 K.

Reason: icoPolynomial EOS → ρ(T) = a0 + a1·T. If you write 300K for LN2 (inlet=77K):
ρ = 1167.9 − 4.7×300 = −242 kg/m³ (negative) → SIGFPE on iteration 0.

When the user does not provide a separate initial bulk/domain temperature:
- use the inlet temperature as the default initial field

Therefore:
- `internalField uniform <inletTemperature>;`  ← must equal `inlet_temperature` from spec

## Boundary rules
- inlet:
  - `type fixedValue;`
  - `value uniform <inletTemperature>;`
- outlet:
  - `type zeroGradient;`
- wall:
  - if a wall temperature is specified:
    - `type fixedValue;`
    - `value uniform <wallTemperature>;`
  - otherwise:
    - `type zeroGradient;`
- frontAndBack:
  - `type empty;`
- symmetry:
  - `type symmetry;`

## Constraints
- Always generate an explicit positive `internalField`.
- Use user-provided thermal data when available.
- Keep the field consistent with the selected thermodynamic model.
