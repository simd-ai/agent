# rhoSimpleFoam — 0/p

**Dimensions**: `[1 -1 -2 0 0 0 0]` (Pa — absolute static pressure)
**NOT kinematic**: do NOT use `[0 2 -2 0 0 0 0]` or values like `0` or `1`

## internalField

**CRITICAL**: `internalField` MUST be initialized to the **same value as the outlet fixedValue pressure**.
A mismatch causes a pressure discontinuity on iteration 1 → SIGFPE in GAMGSolver.

```
internalField   uniform <outlet_pressure>;   // MUST match outlet fixedValue exactly
```

- If the outlet is at 400000 Pa → `internalField uniform 400000;`
- If the outlet is at 101325 Pa → `internalField uniform 101325;`
- Read `operating_pressure` from CaseSpec (do NOT default to 101325 unless that is the actual outlet pressure)

Using `uniform 0` or any value that doesn't match the outlet causes divergence or SIGFPE.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `zeroGradient` | Pressure-free inlet when velocity/flow-rate is specified |
| outlet | `fixedValue` | `value uniform <outlet_pressure>;` — use configured operating_pressure |
| wall | `zeroGradient` | |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Standard outlet template
```
outlet
{
    type            fixedValue;
    value           uniform <outlet_pressure>;   // Pa — use operating_pressure from CaseSpec
}
```

## Rules

- Dimensions are **Pa** (`[1 -1 -2 0 0 0 0]`), never kinematic (`[0 2 -2 0 0 0 0]`)
- `internalField` MUST equal the outlet `fixedValue` pressure — never differ from it
- Every patch in `CaseSpec.patch_names` must appear in `boundaryField`
