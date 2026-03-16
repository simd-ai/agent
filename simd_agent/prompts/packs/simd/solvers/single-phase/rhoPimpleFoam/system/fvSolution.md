# rhoPimpleFoam — system/fvSolution

## Key rules

- `rho` diagonal solver entry is MANDATORY — the solver looks it up at runtime even though `0/rho` is not a file.
- Energy field in regex: use `h` or `e` to match `thermoType.energy` — not both. Only include turbulence fields that actually exist (e.g. kOmegaSST → k, omega; kEpsilon → k, epsilon — never include both omega AND epsilon).
- Final blocks: repeat solver settings explicitly — do NOT use `$"(U|h|…)"` alias syntax (OpenFOAM cannot dereference regex-named entries via `$` and will crash with "Found ERROR but expected }"). `$p` and `$rho` alias syntax IS safe for simple (non-regex) keys.
- `nOuterCorrectors > 1` → under-relaxation required for stability.

## HARD RULES — pressure solver and PIMPLE

**NEVER use GAMG for pressure in rhoPimpleFoam.** GAMG crashes in `GAMGSolver::scale` at cold
start regardless of heat transfer setting. The pressure matrix is underdetermined before velocity
is established. This is a hard constraint — do NOT use GAMG.

**ALWAYS use PBiCGStab + DILU for p and pFinal.**

**NEVER use `momentumPredictor no`** — always set `momentumPredictor yes`.

**NEVER generate `nOuterCorrectors 50` as a startup default.** Use:
- `nOuterCorrectors 2` when all BC temperatures are within 20 K of each other (near-isothermal)
- `nOuterCorrectors 50` with `residualControl` ONLY when ΔT > 50 K across BCs (h-ρ stiffness)

## Two distinct configurations

### A. Isothermal / near-isothermal (all BC temperatures within ~20 K)

Use nOuterCorrectors 2. The h-ρ stiffness that requires 50 outer correctors does NOT apply
when temperature barely varies — the energy equation converges trivially.

### B. Large temperature gradient (ΔT > 50 K across BCs)

Use nOuterCorrectors 50 with residualControl to handle h-ρ stiffness — see details below.

## nNonOrthogonalCorrectors — CRITICAL for compressible heat transfer

**NEVER use 0 when heat transfer is active or density varies significantly.**
With 0 non-orthogonal correctors, the pressure Laplacian ignores mesh skewness.
For compressible flows with large density gradients this causes mass-conservation errors
that accumulate each time step and eventually diverge.

| Condition | Value |
|---|---|
| Structured orthogonal mesh, isothermal | 0 |
| Any non-orthogonality OR heat transfer | 1 |
| Cryogenic (LN2/LH2/LOX) or T span > 100 K | 2 |

## ρ-h stiffness — CRITICAL for cryogenic startup

**Root cause of Courant runaway on first time step**:

In the PIMPLE outer loop, the sequence is: solve-h → update-ρ → solve-p → correct-U/phi → repeat.
With only 3 outer correctors and no `nEnergyCorrectors`, this happens every outer iteration:
1. The pressure correction changes phi significantly (cold start: p has to build from zero → residual = 1.0)
2. On the next outer iteration, h sees completely new coefficients from the changed phi → h initial residual resets to 1.0
3. After 3 outer iterations, the h-ρ state is still inconsistent → velocity over-correction → Co = 200+

**Fixes applied**:
1. `h` solver uses `relTol 0` (not `relTol 0.1`). With `relTol 0.1`, h exits after 1 iteration (0.01 < 0.1×1.0 → satisfied). This leaves h far from converged. `relTol 0` forces h to reach absolute tolerance every outer iteration.
2. `residualControl` + `nOuterCorrectors 50`: PIMPLE iterates until p, U, h all converge. For first time step (hard): many iterations. For established flow: exits in 3-5 iterations.
3. `bounded Gauss upwind` for div(phi,h) when ΔT > 100 K (see fvSchemes.md) — prevents enthalpy overshoots that cause limitTemperature to clamp 50%+ of cells, making h ill-conditioned.

## Template A — Isothermal / no heat transfer

Use this when `heat_transfer = false` or ΔT ≈ 0 (wall at same temperature as fluid, no energy gradient).

```
solvers
{
    p
    {
        // PBiCGStab safer than GAMG for isothermal single-phase startup.
        // GAMG can hit zero-pivot in GAMGSolver::scale on first time step when the
        // pressure matrix is underdetermined (cold start, no established flow).
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0;
    }

    rho
    {
        solver      diagonal;
        tolerance   1e-12;
        relTol      0;
    }
    rhoFinal    { $rho; relTol 0; }

    "(U|k|omega|alphat)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|k|omega|alphat)Final"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   yes;
    transonic           no;

    // 2 outer correctors is sufficient for isothermal single-phase flow.
    // nOuterCorrectors 50 is ONLY needed when the energy equation is active
    // (h-rho coupling stiffness). Without heat transfer, 50 outer correctors
    // wastes runtime and can amplify numerical noise.
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
}

relaxationFactors
{
    fields      { p 0.3; rho 0.1; }
    equations   { U 0.5; k 0.5; omega 0.5; alphat 0.5; }
}
```

## Template B — Heat transfer / cryogenic (ΔT > 50 K, energy equation active)

Use this when `heat_transfer = true` or a significant temperature difference exists across the domain.

```
solvers
{
    // PBiCGStab+DILU is MANDATORY for rhoPimpleFoam — GAMG crashes in GAMGSolver::scale
    // at cold start regardless of heat transfer setting (see hard rule above).
    p
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0;
    }

    rho
    {
        solver      diagonal;
        tolerance   1e-12;
        relTol      0;
    }
    rhoFinal    { $rho; relTol 0; }

    // h MUST have relTol 0 — CRITICAL for PIMPLE outer loop stability.
    // With relTol 0.1, the h solver exits after 1 iteration (residual drops from 1.0
    // to ~0.01, which is < 0.1×1.0 → relTol satisfied). This leaves h far from
    // converged. On the next outer PIMPLE iteration, phi has changed → h residual
    // resets to 1.0 → solver exits again after 1 iteration. The outer loop oscillates
    // all 50 iterations without converging → Co blows up → SIGFPE.
    // With relTol 0, h must reach the absolute tolerance (1e-6) every outer iteration.
    h   // replace with 'e' if thermoType.energy = sensibleInternalEnergy
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
    hFinal
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }

    // Build regex from turbulence fields only (k+omega OR k+epsilon — never both).
    "(U|k|omega|alphat)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|k|omega|alphat)Final"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   yes;   // always yes — see hard rule; no momentum predictor destabilises startup
    transonic           no;

    nOuterCorrectors    50;    // high ceiling; residualControl exits early once converged
    nCorrectors         1;
    nNonOrthogonalCorrectors 2;   // 2 for heat transfer / cryogenic; 1 for mild gradients; 0 only for isothermal orthogonal

    residualControl
    {
        // CRITICAL: each entry MUST be a sub-dictionary — NOT a plain scalar.
        // Plain scalars (p 5e-3;) crash with: "Residual data for p must be specified as a dictionary"
        // PIMPLE exits once ALL fields are below tolerance. For first timestep: many iterations.
        // For established flow: exits in 3-5. relTol 0 means only absolute tolerance is checked.
        p
        {
            tolerance   5e-3;
            relTol      0;
        }
        U
        {
            tolerance   5e-3;
            relTol      0;
        }
        h
        {
            tolerance   5e-3;
            relTol      0;
        }
    }
}

relaxationFactors
{
    // CRYOGENIC / HEAT TRANSFER (icoPolynomial, ΔT > 100 K): use conservative values below.
    // High relaxation (0.7–0.9) causes turbulence blow-up in the first time step:
    //   - k/omega go negative at near-wall cells (omegaWallFunction gives ω_wall ≈ 60000 s⁻¹
    //     vs bulk ω ≈ 10 s⁻¹; destruction >> production → k < 0 → nut < 0 → velocity diverges)
    //   - h oscillates → limitTemperature clamps 90%+ of cells → density field collapses
    fields      { p 0.1; rho 0.05; }
    equations   { U 0.3; h 0.2; k 0.3; omega 0.3; alphat 0.5; }
}
```
