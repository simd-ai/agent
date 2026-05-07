# rhoPimpleFoam — 0/p

**Dimensions**: `[1 -1 -2 0 0 0 0]` (Pa) — absolute static pressure
**NOT kinematic** — do not use `[0 2 -2 0 0 0 0]`
**internalField**: `uniform <outlet_pressure>` — MUST match the outlet `fixedValue` exactly to avoid SIGFPE on iteration 1

## BC types

| Patch role | BC type |
|---|---|
| inlet | `zeroGradient` |
| outlet | `fixedValue` — use configured `operating_pressure` from CaseSpec (NOT hardcoded 101325 unless that is the actual value) |
| wall | `zeroGradient` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
