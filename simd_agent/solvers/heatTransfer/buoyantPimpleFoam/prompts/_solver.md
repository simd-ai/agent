# Solver: buoyantPimpleFoam — Identity & Global Rules

**Algorithm**: PIMPLE (transient)
**Compressible**: yes — heRhoThermo; density varies with temperature and pressure
**Pressure fields**: TWO files are REQUIRED:
  - `p_rgh` — dynamic modified pressure (primary solved variable), Pa, `[1 -1 -2 0 0 0 0]`
  - `p` — absolute static pressure (calculated/derived), Pa, `[1 -1 -2 0 0 0 0]`
  - Relationship: `p = p_rgh + rho*(g·x)` — reconstructed each time step
**Energy**: `he` — sensibleEnthalpy; temperature field `0/T` drives density via EOS
**Gravity**: `constant/g` REQUIRED — buoyancy drives transient density currents

**Use case**: Transient natural convection (fire smoke spread, room ventilation transients,
heated cavity fill, sloshing heated fluid), moving parts with heat transfer, oscillating
thermal plumes, building fire dynamics.

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `thermophysicalProperties`, `turbulenceProperties`, `g` |
| `0/` | `U`, `p_rgh`, `p`, `T`, and turbulence fields matching the selected model |

**Also generate** `0/alphat` when turbulence is active.

**Never generate**: `0/rho`, `0/h`, `0/e`

---

## Critical: p vs p_rgh

Same as buoyantSimpleFoam:
- `p_rgh` is the SOLVED variable.
- `p` is CALCULATED — all patches must use `type calculated`.
- `fixedFluxPressure` on walls and closed boundaries for `p_rgh`.
- `fixedValue` at pressure outlets for `p_rgh`.

---

## Global critical rules

1. Every mesh patch in `patch_names` MUST appear in EVERY `0/*` field file.
2. `application` in `controlDict` MUST equal `buoyantPimpleFoam`.
3. `0/p` uses `type calculated` on ALL patches.
4. `constant/g` MUST be generated.
5. `fluxRequired` in fvSchemes MUST list `p_rgh`.
6. `residualControl` in PIMPLE block MUST use `p_rgh`.
7. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
8. NO `div(phid,p)` in fvSchemes.
9. `endTime` and `deltaT` in PHYSICAL SECONDS (not iteration counter).
10. `nOuterCorrectors ≥ 2` — required for p_rgh–ρ–T coupling (PIMPLE, not PISO).
11. `adjustTimeStep yes; maxCo 0.5;` recommended for stability.
12. When turbulence active: generate `0/alphat` with `compressible::alphatWallFunction`.
13. `0/T` internalField MUST equal ambient/initial temperature, NOT 300 K default.
