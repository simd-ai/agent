# pimpleFoam — 0/nut

**Dimensions**: `[0 2 -1 0 0 0 0]` (m²/s)
**internalField**: `uniform 0`

## BC types

| Patch | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `nutkWallFunction` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
