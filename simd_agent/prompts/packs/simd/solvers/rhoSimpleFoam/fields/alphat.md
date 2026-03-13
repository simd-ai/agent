# rhoSimpleFoam — 0/alphat

**Dimensions**: `[1 -1 -1 0 0 0 0]` (kg/m/s — turbulent thermal diffusivity)
**Only for**: compressible solvers with turbulence AND energy active
**internalField**: `uniform 0`

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `calculated` | `value uniform 0` |
| outlet | `calculated` | `value uniform 0` |
| wall | `compressible::alphatWallFunction` | `Prt 0.85; value uniform 0;` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

Use the fully-qualified name `compressible::alphatWallFunction` — not just `alphatWallFunction`.

## Template

```
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    walls
    {
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }
}
```
