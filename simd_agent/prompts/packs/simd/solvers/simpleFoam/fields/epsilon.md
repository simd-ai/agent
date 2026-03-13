# simpleFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)

## internalField

Use `CaseSpec.turbulence_initial_values.epsilon` when available.
Formula: `ε = Cμ^0.75 × k^1.5 / L` where Cμ = 0.09.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `zeroGradient` |
| wall | `epsilonWallFunction` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

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
