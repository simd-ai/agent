# rhoPimpleFoam — 0/alphat

**Dimensions**: `[1 -1 -1 0 0 0 0]` (turbulent thermal diffusivity)
**internalField**: `uniform 0`

Generate ONLY when turbulence AND energy are both active.

## BC types

| Patch | BC type |
|---|---|
| inlet | `calculated` + `value uniform 0` |
| outlet | `calculated` + `value uniform 0` |
| wall | `compressible::alphatWallFunction` + `Prt 0.85;` + `value uniform 0` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
