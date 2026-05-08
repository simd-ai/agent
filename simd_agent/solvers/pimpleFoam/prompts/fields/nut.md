# pimpleFoam — 0/nut

**Dimensions**: `[0 2 -1 0 0 0 0]` (m^2/s)
**internalField**: `uniform 0`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `nutkWallFunction` + `value uniform 0` |
| symmetry | `symmetry` |
| symmetryPlane | `symmetryPlane` |
| empty (2D planar) | `empty` — no `value`, just `type empty;` |
| wedge (2D axi) | `wedge` — no `value`, just `type wedge;` |

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
        type            nutkWallFunction;
        value           uniform 0;
    }
}
```
