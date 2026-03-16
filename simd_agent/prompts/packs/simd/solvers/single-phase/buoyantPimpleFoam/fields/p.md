# buoyantSimpleFoam — 0/p

**Derived/calculated field** — NOT directly solved. The solver reconstructs `p` from
`p_rgh` at each iteration: `p = p_rgh + rho*(g·x)`.

## CRITICAL: ALL patches MUST use `type calculated`

```
dimensions      [1 -1 -2 0 0 0 0];   // Pa — absolute pressure
internalField   uniform <operating_pressure>;  // same value as 0/p_rgh internalField

boundaryField
{
    // EVERY patch uses "calculated" — no exceptions
    inlet
    {
        type    calculated;
        value   $internalField;
    }
    outlet
    {
        type    calculated;
        value   $internalField;
    }
    hotWall
    {
        type    calculated;
        value   $internalField;
    }
    coldWall
    {
        type    calculated;
        value   $internalField;
    }
    // ... repeat for ALL patches in patch_names
}
```

## Why `calculated` on every patch

buoyantSimpleFoam reconstructs `p` internally from `p_rgh`. Any non-`calculated` BC
type on `0/p` would conflict with the solver's internal pressure update and produce
incorrect results or a runtime error.

This is the ONLY field in the case where ALL patches unconditionally use `type calculated`.
