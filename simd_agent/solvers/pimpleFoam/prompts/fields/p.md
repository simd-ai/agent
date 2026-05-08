# pimpleFoam — 0/p

**Dimensions**: `[0 2 -2 0 0 0 0]` (kinematic pressure, m^2/s^2)
**NOT Pa**: do NOT use `[1 -1 -2 0 0 0 0]` or values like 101325

## internalField

`uniform 0` is standard for incompressible. The absolute value doesn't matter — only gradients do.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `zeroGradient` | When velocity is prescribed at inlet |
| outlet | `fixedValue` | `value uniform 0;` — reference pressure |
| wall | `zeroGradient` | |
| symmetry | `symmetry` | |
| symmetryPlane | `symmetryPlane` | |
| empty (2D planar) | `empty` | No `value` — just `type empty;` |
| wedge (2D axi) | `wedge` | No `value` — just `type wedge;` |

## pRef in closed domains

When NO patch has a `fixedValue` pressure BC, the pressure matrix is singular.
Add to `fvSolution PIMPLE {}` block:
```
pRefCell    0;
pRefValue   0;
```

## Template

```
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
}
```
