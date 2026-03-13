# simpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)

## internalField

Use `CaseSpec.turbulence_initial_values.omega` when available.
Formula: `ω = k^0.5 / (Cμ^0.25 × L)` where Cμ = 0.09, L ≈ 0.07 × D_h.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `zeroGradient` |
| wall | `omegaWallFunction` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

```
internalField   uniform <omega_value>;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform <omega_value>;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            omegaWallFunction;
        value           uniform <omega_value>;
    }
}
```
