# Solver: compressibleInterIsoFoam — Identity & Global Rules

**Algorithm**: PIMPLE + isoAdvection (transient, two-phase VOF, compressible, non-isothermal, geometric interface reconstruction)
**Pressure fields**: `p_rgh` (modified pressure) AND `p` (absolute pressure) — BOTH required
**Energy**: YES — generate `0/T` (Kelvin)
**Gravity**: REQUIRED — always generate `constant/g`
**Key difference from compressibleInterFoam**: Uses isoAdvector geometric reconstruction for sharper interfaces (thin films, fast impacts, droplets). Otherwise same physics.
**Turbulence**: uses `nut` and `alphat` (same as other modern OF solvers)

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `g`, `thermophysicalProperties` (base), `thermophysicalProperties.<phase1Name>`, `thermophysicalProperties.<phase2Name>`, `turbulenceProperties` |
| `0/` | `U`, `p_rgh`, `p`, `T`, `alpha.<phase1Name>`, `k`/`omega`/`epsilon` (if turbulent), `nut`, `alphat` |

**CRITICAL**: Generate BOTH `0/p` and `0/p_rgh`:
- `0/p_rgh` — modified pressure = p − ρ·g·h (dimensions `[1 -1 -2 0 0 0 0]`)
- `0/p` — absolute pressure (same dimensions `[1 -1 -2 0 0 0 0]`, same BCs as p_rgh)
  OpenFOAM reads `0/p` at startup with `MUST_READ`. Missing it causes "cannot find file 0/p" fatal error.

**Never generate**: `0/h`, `0/e`, `0/mut`, `constant/transportProperties`, `system/fvOptions`

---

## Phase naming (CRITICAL)

Use phase names from CaseSpec (`alpha_fields`). Examples:
- LN2 boiloff: `liquidNitrogen` and `nitrogenVapour`
- LH2: `liquidHydrogen` and `hydrogenVapour`
- LOX: `liquidOxygen` and `oxygenVapour`
- LHe: `liquidHelium` and `heliumVapour`
- Water/air: `water` and `air`

---

## ThermophysicalProperties file structure (CRITICAL)

Three-file approach — same as compressibleInterFoam:

### 1. `constant/thermophysicalProperties` (base — MINIMAL)
```
phases ( <phase1Name> <phase2Name> );
pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  <sigma>;
```
**DO NOT put thermoType or mixture here.**

### 2. `constant/thermophysicalProperties.<phase1Name>` (liquid)
Full `thermoType {}` + `mixture {}` with icoPolynomial EOS (for cryogenic liquids) or rhoConst.

### 3. `constant/thermophysicalProperties.<phase2Name>` (vapour)
Full `thermoType {}` + `mixture {}` with perfectGas EOS.

---

## isoAdvection-specific fvSolution settings

```
"alpha.<phase1Name>.*"
{
    nAlphaCorr          1;
    nAlphaSubCycles     1;
    cAlpha              1;
    MULESCorr           yes;
    nLimiterIter        8;
    alphaApplyPrevCorr  yes;

    reconstructionScheme isoAlpha;   // KEY DIFFERENCE from compressibleInterFoam
}
```

---

## Global critical rules

1. Generate BOTH `0/p_rgh` AND `0/p` — compressibleInterIsoFoam reads both at startup.
2. Always generate `0/T` and `constant/g`.
3. `pcorr` solver block is REQUIRED in `fvSolution`.
4. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
4b. **PIMPLE `nOuterCorrectors` MUST be ≥ 2** — icoPolynomial liquid has zero acoustic compressibility. PISO mode (nOuterCorrectors 1) causes Co to explode and T to crash negative within 5 timesteps. Use `nNonOrthogonalCorrectors 1` and `momentumPredictor yes`.
4c. **`maxCo 0.5` in controlDict** — NOT 1.0.
5. `divSchemes default` MUST be `Gauss linear` — NEVER `none`, `Gauss upwind`, or `bounded Gauss upwind`. In OF 2406, `none` causes "attempt to read beyond EOF". `Gauss linear` is safe for all types. List BOTH `div(rhoPhi,he)` AND `div(rhoPhi,h)`.
5. Use `nut` (kinematic) not `mut` — modern OF 2406 ESI uses `nut`.
6. Base `thermophysicalProperties` contains ONLY `phases`, `pMin`, `sigma`. All thermo goes in per-phase files.
7. `startFrom startTime; startTime 0;` — never `latestTime`.
8. Every mesh patch must appear in ALL `0/*` files; `empty` patches → `type empty`.
9. **NEVER generate `system/fvOptions`** — `limitTemperature` calls `he()` on `twoPhaseMixtureThermo` which does not implement it → `FOAM FATAL ERROR: Not implemented` at startup.
