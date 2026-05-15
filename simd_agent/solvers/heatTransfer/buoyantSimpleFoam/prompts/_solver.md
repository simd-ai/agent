# Solver: buoyantSimpleFoam — Identity & Global Rules

**Algorithm**: SIMPLE (steady-state)
**Compressible**: yes — heRhoThermo; density varies with temperature and pressure
**Pressure fields**: TWO files are REQUIRED:
  - `p_rgh` — dynamic modified pressure (primary solved variable), Pa, `[1 -1 -2 0 0 0 0]`
  - `p` — absolute static pressure (calculated/derived), Pa, `[1 -1 -2 0 0 0 0]`
  - Relationship: `p = p_rgh + rho*(g·x)` — the solver reconstructs p at each iteration
**Energy**: `he` — sensibleEnthalpy; temperature field `0/T` drives density via EOS
**Gravity**: `constant/g` REQUIRED — buoyancy source term `−rho*g` drives flow

**Use case**: Natural convection (HVAC, heated rooms, electronic cooling, chimney effect),
gravity-driven density stratification, low-speed heat transfer where gravity matters.

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `thermophysicalProperties`, `turbulenceProperties`, `g` |
| `0/` | `U`, `p_rgh`, `p`, `T`, and turbulence fields matching the selected model |

**Also generate** `0/alphat` when turbulence is active (compressible::alphatWallFunction).

**Never generate**: `0/rho`, `0/h`, `0/e`
- `0/rho` — computed from heRhoThermo at runtime
- `0/h` / `0/e` — thermo initialises energy from `0/T`

---

## Critical: p vs p_rgh

`p_rgh` is the variable actually solved by SIMPLE.
`p` is a DERIVED field — every patch MUST use `type calculated`.
DO NOT put `fixedValue`, `zeroGradient`, or any other non-calculated type on `0/p` patches.

```
// 0/p — ALL patches must be "calculated"
boundaryField
{
    inlet  { type calculated; value $internalField; }
    outlet { type calculated; value $internalField; }
    walls  { type calculated; value $internalField; }
}
```

`p_rgh` uses:
- `fixedFluxPressure` on walls and no-flow boundaries (maintains flux consistency with buoyancy)
- `fixedValue` at pressure outlets (absolute Pa, e.g. 100000)
- `fixedFluxPressure` at velocity inlets

---

## Critical: p_rgh internalField

Set to operating pressure in Pa (e.g. `uniform 100000` for 1 atm).
Do NOT use 0 — this is absolute pressure, and perfectGas EOS needs a non-zero reference.

---

## Global critical rules

1. Every mesh patch in `patch_names` MUST appear in EVERY `0/*` field file.
2. `application` in `controlDict` MUST equal `buoyantSimpleFoam`.
3. `0/p` uses `type calculated` on ALL patches — never fixedValue or zeroGradient.
4. `0/p_rgh` uses `fixedFluxPressure` on walls/inlets, `fixedValue` at pressure outlets.
5. `constant/g` MUST be generated — omitting it causes fatal IO error at startup.
6. `fluxRequired` in fvSchemes MUST list `p_rgh` (not `p`).
7. `residualControl` in SIMPLE block MUST use `p_rgh` (not `p`).
8. `relaxationFactors.fields` MUST include `p_rgh` (not `p`).
9. GAMG smoother MUST be `GaussSeidel` — never `DIC` (causes SIGFPE).
10. Do NOT generate `div(phid,p)` in fvSchemes — buoyantSimpleFoam does NOT use it.
11. `endTime` = `max_iterations` (iteration counter); `deltaT 1`.
12. When turbulence active + energy: generate `0/alphat` with `compressible::alphatWallFunction`.
13. `0/T` internalField MUST equal `inlet_temperature` (NOT 300 K default).
