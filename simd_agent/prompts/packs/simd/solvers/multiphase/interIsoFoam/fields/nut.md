# interIsoFoam — 0/nut

**Dimensions**: `[0 2 -1 0 0 0 0]`
**internalField**: `uniform 0`
Generate only when turbulence is active.

## BC types

| Patch | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `nutkWallFunction` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
