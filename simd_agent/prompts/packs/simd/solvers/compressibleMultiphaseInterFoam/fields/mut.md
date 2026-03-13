# compressibleMultiphaseInterFoam — 0/mut

**Dimensions**: `[1 -1 -1 0 0 0 0]` (dynamic turbulent viscosity, Pa·s)
**internalField**: `uniform 0`

Use `mut` (dynamic), NOT `nut`. Generate only when turbulence is active.

## BC types

| Patch | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `mutkWallFunction` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
