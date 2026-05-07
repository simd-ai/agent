# Solver: rhoSimpleFoam — Identity & Global Rules

**Algorithm**: SIMPLE (steady-state)
**Compressible**: yes
**Pressure field**: `p` — absolute static pressure, Pa, `[1 -1 -2 0 0 0 0]`
**Energy**: `he` — thermo package transports sensibleEnthalpy (`h`) or sensibleInternalEnergy (`e`)
**Gravity**: no `constant/g` for rhoSimpleFoam

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution`, `fvOptions` |
| `constant/` | `thermophysicalProperties`, `turbulenceProperties` |
| `0/` | `U`, `p`, `T`, and turbulence fields matching the selected model |

**Never generate**: `0/rho`, `0/h`, `0/e`, `constant/g`
- `0/rho` — density is `thermo.rho()` at runtime, not a file
- `0/h` / `0/e` — thermo initialises energy from `0/T`; providing `0/h` causes `Negative initial temperature T0` crashes

---

## Global critical rules

1. Every mesh patch listed in `patch_names` MUST appear in every `0/*` field file.
2. `application` in `controlDict` MUST equal `rhoSimpleFoam` — never a sibling solver.
3. `0/p` is absolute pressure in **Pa**. Do not use kinematic units (m²/s²).
4. `0/T` is temperature in **Kelvin**.
5. GAMG smoother MUST be `GaussSeidel` — never `DIC` (causes SIGFPE, exit 136).
6. Energy variable (`h` or `e`) must be consistent across `thermophysicalProperties`, `fvSchemes` div entries, `fvSolution` regex groups, and `residualControl`.
7. When turbulence + energy are both active, generate `0/alphat` with `compressible::alphatWallFunction` on walls.
8. `startFrom startTime; startTime 0;` in `controlDict` — never `latestTime`.
9. `controlDict` `endTime` = `max_iterations` (iteration counter); `deltaT 1`.
10. Do NOT invent fields or patches not listed in CaseSpec.
