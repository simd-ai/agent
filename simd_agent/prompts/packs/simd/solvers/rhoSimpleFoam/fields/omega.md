# rhoSimpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)
**Only for**: kOmegaSST, kOmega turbulence models

## internalField

Use pre-computed value from `CaseSpec.turbulence_initial_values.omega` when available.
Formula: `ω = k^0.5 / (Cμ^0.25 × L)`
where Cμ = 0.09, L ≈ 0.07 × hydraulic_diameter (or characteristic length from CaseSpec).

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <omega_value>` |
| outlet | `zeroGradient` | |
| wall | `omegaWallFunction` | `value uniform <omega_value>` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Template

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
