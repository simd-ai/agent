# compressibleMultiphaseInterFoam — 0/T

**Dimensions**: `[0 0 0 1 0 0 0]` (Kelvin)

## internalField

**CRITICAL**: Use `inlet_temperature` from the case spec. NEVER default to 300 K.
For cryogenic fluids (LN2, LH2, LOX, LHe): a wrong internalField (e.g. 300K) gives
negative density via icoPolynomial EOS → SIGFPE on iteration 0.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <inlet_temperature>` |
| outlet | `zeroGradient` |
| wall (adiabatic) | `zeroGradient` |
| wall (fixed temp) | `fixedValue` + `value uniform <wall_temperature>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
