# Solver: interIsoFoam — Identity & Global Rules

**Algorithm**: PIMPLE + isoAdvection (transient, two-phase VOF, geometric interface)
**Compressible**: no
**Pressure field**: `p_rgh` — kinematic pressure minus hydrostatic head, `[0 2 -2 0 0 0 0]`
**Energy**: NONE — do NOT generate `0/T`
**Gravity**: REQUIRED — always generate `constant/g`

Same required files as interFoam. Key difference: alpha interface advection uses **isoAdvection** (geometric VOF) for a sharper interface than classic MULES alone.

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `g`, `phaseProperties`, `turbulenceProperties` |
| `0/` | `U`, `p_rgh`, `alpha.<phase1Name>`, turbulence fields if active |

**Never generate**: `0/T`, `0/p`, `constant/transportProperties`

---

## Global critical rules

1. Generate `0/p_rgh` (NOT `0/p`).
2. Always generate `constant/g`.
3. Never generate `0/T`.
4. Do NOT write `solver isoAdvector;` — isoAdvection is controlled via reconstruction parameters.
5. Alpha controls go in `alphaControls {}` or `isoAdvection {}` block — NOT inside `solvers {}`.
6. Use `div(phi,alpha) Gauss vanLeer;` and `div(phirb,alpha) Gauss linear;`.
7. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
8. `startFrom startTime; startTime 0;` — never `latestTime`.
9. Every mesh patch must appear in ALL `0/*` files; `empty` patches → `type empty`.
