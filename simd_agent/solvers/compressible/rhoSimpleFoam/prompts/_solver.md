# Solver: rhoSimpleFoam — Identity & Global Rules

**Algorithm**: SIMPLE (steady-state)
**Compressible**: yes
**Pressure field**: `p` — absolute static pressure, Pa, `[1 -1 -2 0 0 0 0]`
**Energy**: `he` — thermo package transports sensibleEnthalpy (`h`) or sensibleInternalEnergy (`e`)
**Gravity**: no `constant/g` for rhoSimpleFoam

---

## Numerical profiles — two distinct setups

`rhoSimpleFoam` is configured under one of two **thermo profiles**, picked
deterministically from the fluid + temperature signature in the validated config:

| Signal in config | → Profile |
|---|---|
| Fluid name matches `LN2 / LH2 / LOX / LHe / cryogenic` | **cryogenic** |
| Any inlet temperature < 200 K | **cryogenic** |
| Liquid (ρ > 200 kg/m³) with heat transfer | **cryogenic** |
| Anything else (air, room-temp gas, no explicit liquid) | **gas** |

### Profile A — Gas (perfectGas / hePsiThermo)
- Standard rhoSimpleFoam textbook setup (Foundation `angledDuct` tutorial)
- `consistent yes` (SIMPLEC) allowed on good meshes
- `div(phi,U) bounded Gauss linearUpwindV grad(U)` for low/moderate Mach
- `grad(U) cellLimited Gauss linear 1` (stability margin)
- Relaxation: `U 0.7, p 0.3, h 0.5, turb 0.7`
- `SIMPLE { rhoMin 0.1; rhoMax 10.0; }` safety bounds always emitted
- `transonic yes` when Mach > 0.5

### Profile B — Cryogenic (icoPolynomial / heRhoThermo)
- Conservative, EOS-ceiling-aware setup for LN2/LH2/LOX/LHe
- `consistent no` (standard SIMPLE — SIMPLEC over-corrects at low T)
- `div(phi,U) bounded Gauss upwind`
- `grad(U) Gauss linear` (no cellLimited — interacts badly with stiff h-ρ coupling)
- Relaxation: `U 0.5, p 0.3, h 0.05, turb 0.5`
- `nNonOrthogonalCorrectors ≥ 2`
- EOS ceiling clamps on `0/T` wall fixedValue and `fvOptions limitTemperature.max`

`controlDict` and `0/*` files are profile-agnostic. Profile affects mainly
`fvSolution`, `fvSchemes`, and the `thermophysicalProperties` EOS choice.

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
