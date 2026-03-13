# Solver: rhoPimpleFoam — Identity & Global Rules

**Algorithm**: PIMPLE (transient)
**Compressible**: yes
**Pressure field**: `p` — absolute static pressure, Pa, `[1 -1 -2 0 0 0 0]`
**Energy**: `he` — thermo package transports sensibleEnthalpy (`h`) or sensibleInternalEnergy (`e`)
**Gravity**: no `constant/g` for rhoPimpleFoam

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
- `0/alphat` — MUST be generated when turbulence AND energy are both active

---

## Global critical rules

1. Every mesh patch listed in `patch_names` MUST appear in every `0/*` field file.
2. `application` in `controlDict` MUST equal `rhoPimpleFoam` — never a sibling solver.
3. `0/p` is absolute pressure in **Pa**. Do not use kinematic units (m²/s²).
4. `0/T` is temperature in **Kelvin**. Never generate `0/h` or `0/e`.
5. GAMG smoother MUST be `GaussSeidel` — never `DIC` (causes SIGFPE, exit 136).
6. Energy variable (`h` or `e`) must be consistent across `thermophysicalProperties`, `fvSchemes` div entries, `fvSolution` regex groups, and `residualControl`.
7. When turbulence + energy are both active, generate `0/alphat` with `compressible::alphatWallFunction` on walls.
8. `startFrom startTime; startTime 0;` in `controlDict` — never `latestTime`.
9. `controlDict` `endTime` and `deltaT` are physical time (seconds), not iteration count.
10. `rho` solver entry is MANDATORY in `fvSolution/solvers` — the solver reads it at runtime even though `0/rho` is not a file.
11. Do NOT invent fields or patches not listed in CaseSpec.
