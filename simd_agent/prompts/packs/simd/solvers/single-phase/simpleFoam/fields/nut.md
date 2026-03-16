# simpleFoam — 0/nut

**Dimensions**: `[0 2 -1 0 0 0 0]` (m²/s)
**internalField**: `uniform 0`

## BC types

| Patch role | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `nutkWallFunction` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

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
