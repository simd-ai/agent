# rhoSimpleFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)
**Only for**: kEpsilon turbulence model

## internalField

Use pre-computed value from `CaseSpec.turbulence_initial_values.epsilon` when available.
Formula: `ε = Cμ^0.75 × k^1.5 / L`
where Cμ = 0.09, L ≈ 0.07 × hydraulic_diameter.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <epsilon_value>` |
| outlet | `zeroGradient` | |
| wall | `epsilonWallFunction` | `value uniform <epsilon_value>` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Template

```
internalField   uniform <epsilon_value>;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform <epsilon_value>;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            epsilonWallFunction;
        value           uniform <epsilon_value>;
    }
}
```
