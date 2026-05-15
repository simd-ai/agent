# simpleFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)

## internalField

Use `CaseSpec.turbulence_initial_values.epsilon` when available.
Formula: `ε = Cμ^0.75 × k^1.5 / L` where Cμ = 0.09.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `inletOutlet` | `inletValue uniform <epsilon_value>` — prevents negative epsilon from backflow |
| wall | `epsilonWallFunction` |
| symmetry | `symmetry` |
| symmetryPlane | `symmetryPlane` |
| empty (2D planar) | `empty` — no `value`, just `type empty;` |
| wedge (2D axi) | `wedge` — no `value`, just `type wedge;` |

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
        type            inletOutlet;
        inletValue      uniform <epsilon_value>;
        value           uniform <epsilon_value>;
    }
    walls
    {
        type            epsilonWallFunction;
        value           uniform <epsilon_value>;
    }
}
```

## Critical rules

- `epsilon` MUST be ≥ 1e-6 — values below this cause division-by-zero in turbulence model
- Formula: `ε = Cμ^0.75 × k^1.5 / L` with Cμ = 0.09
- Outlet MUST use `inletOutlet` (not `zeroGradient`) to prevent negative epsilon from backflow → SIGFPE
```
