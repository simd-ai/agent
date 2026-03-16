# Solver: compressibleMultiphaseInterFoam — Identity & Global Rules

**Algorithm**: PIMPLE + MULES (transient, N-phase VOF, compressible, non-isothermal)
**Pressure fields**: `p_rgh` (modified pressure) AND `p` (absolute pressure) — BOTH required
**Energy**: YES — generate `0/T` (Kelvin)
**Gravity**: REQUIRED — always generate `constant/g`
**Phases**: N ≥ 3 (three or more phases)
**Alpha fields**: `0/alphas` (composite, always required) + `0/alpha.<phaseName>` per phase
**Turbulence**: uses `nut` and `alphat` (same as other modern OF solvers)

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `g`, `thermophysicalProperties` (base), `thermophysicalProperties.<phaseName>` (one per phase), `turbulenceProperties` |
| `0/` | `U`, `p_rgh`, `p`, `T`, `alphas`, `alpha.<phaseName>` (one per phase), `k`/`omega`/`epsilon` (if turbulent), `nut`, `alphat` |

**CRITICAL**: Generate BOTH `0/p` and `0/p_rgh`:
- `0/p_rgh` — modified pressure = p − ρ·g·h (dimensions `[1 -1 -2 0 0 0 0]`)
- `0/p` — absolute pressure (same dimensions `[1 -1 -2 0 0 0 0]`, same BCs as p_rgh)
  OpenFOAM reads `0/p` at startup with `MUST_READ`. Missing it causes "cannot find file 0/p" fatal error.

**Never generate**: `0/h`, `0/e`, `0/mut`, `constant/transportProperties`

---

## Phase naming (CRITICAL)

- Use phase names from CaseSpec (`phase_names`). N must be ≥ 3.
- Generate `0/alpha.<phaseName>` for EVERY phase listed.
- `0/alphas` is ALWAYS required — this is the composite alpha field (sum of all phase alphas).
- Per-phase thermo: `constant/thermophysicalProperties.<phaseName>` (one file per phase).

---

## ThermophysicalProperties file structure (CRITICAL)

One base file + one file per phase:

### Base: `constant/thermophysicalProperties`
```
phases ( <phase1> <phase2> <phase3> ... );
pMin   [1 -1 -2 0 0 0 0]  10000;
sigma12  [1  0 -2 0 0 0 0]  <sigma_12>;  // between phase1 and phase2
sigma13  [1  0 -2 0 0 0 0]  <sigma_13>;  // between phase1 and phase3
```

### Per-phase: `constant/thermophysicalProperties.<phaseName>`
Full `thermoType {}` + `mixture {}` for each phase.
- Liquid phases: use `icoPolynomial` EOS (or `rhoConst` for incompressible approximation)
- Gas phases: use `perfectGas` EOS

---

## fvSolution — alpha solver blocks

```
"alpha.*"     { nAlphaCorr 1; nAlphaSubCycles 1; cAlpha 1; MULESCorr yes; ... }
"alphas.*"    { nAlphaCorr 1; nAlphaSubCycles 1; cAlpha 1; MULESCorr yes; ... }
```
Both `alpha.*` AND `alphas.*` blocks are required.

---

## Global critical rules

1. Generate BOTH `0/p_rgh` AND `0/p` — compressibleMultiphaseInterFoam reads both at startup.
2. Always generate `0/T`, `constant/g`, and `0/alphas`.
3. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
3b. **PIMPLE `nOuterCorrectors` MUST be ≥ 2** — icoPolynomial liquid has zero acoustic compressibility. PISO mode causes Co to explode and T to crash negative within 5 timesteps. Use `nNonOrthogonalCorrectors 1` and `momentumPredictor yes`.
3c. **`maxCo 0.5` in controlDict** — NOT 1.0.
4. `divSchemes default` MUST be `Gauss linear` — NEVER `none`, `Gauss upwind`, or `bounded Gauss upwind`. In OF 2406, `none` causes "attempt to read beyond EOF". `Gauss linear` is safe for all types. List BOTH `div(rhoPhi,he)` AND `div(rhoPhi,h)`.
4. Use `nut` (kinematic) not `mut` — modern OF 2406 ESI uses `nut`.
5. Base `thermophysicalProperties` contains ONLY `phases`, `pMin`, `sigma`. All thermo goes in per-phase files.
6. `startFrom startTime; startTime 0;` — never `latestTime`.
7. Every mesh patch must appear in ALL `0/*` files; `empty` patches → `type empty`.
8. **NEVER generate `system/fvOptions`** — `limitTemperature` calls `he()` on `twoPhaseMixtureThermo` which does not implement it → `FOAM FATAL ERROR: Not implemented` at startup.
