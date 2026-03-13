# Solver: icoFoam — Identity & Global Rules

**Algorithm**: PISO (transient)
**Compressible**: no (incompressible, Newtonian, laminar)
**Pressure field**: `p` — kinematic pressure, m²/s², `[0 2 -2 0 0 0 0]`
**Turbulence**: NONE
**Energy**: NONE

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `transportProperties` |
| `0/` | `U`, `p` |

**Never generate**: `0/T`, `0/k`, `0/omega`, `0/epsilon`, `0/nut`, `constant/turbulenceProperties`, `constant/g`

---

## Global critical rules

1. icoFoam is LAMINAR — never add turbulence files.
2. `constant/transportProperties` contains ONLY `nu` (kinematic viscosity).
3. Use `PISO {}` block — NOT SIMPLE or PIMPLE.
4. `0/U` and `0/p` must include ALL mesh patches with exact names from CaseSpec.
5. 2D meshes: any `empty` patch must be `type empty` in every `0/*` file.
6. Time settings are physical (`deltaT` in seconds, `endTime` in seconds).
7. `startFrom startTime; startTime 0;` — never `latestTime`.
8. `application icoFoam;` in `controlDict` — never a sibling solver.
