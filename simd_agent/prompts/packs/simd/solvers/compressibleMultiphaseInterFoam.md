# Solver: compressibleMultiphaseInterFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible · N-phase (N≥3) VOF · Non-isothermal  
**Pressure field**: `p_rgh` (dimensions `[1 -1 -2 0 0 0 0]`) — MUST have `0/p_rgh`  
**Energy equation**: ✅ YES — MUST generate `0/T`  
**Gravity file**: ✅ ALWAYS generate `constant/g` (solver reads gravity via `readGravitationalAcceleration.H`)  
**Alpha fields**: ✅ `0/alphas` (always) + one `0/alpha.<phaseName>` per phase  
**Thermophysical**: ✅ Base file + separate thermo per phase  

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application compressibleMultiphaseInterFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | Transient + VOF div schemes + energy div schemes |
| `system/fvSolution` | PIMPLE + alpha/MULES controls + solvers for p_rgh, U, T |
| `0/U` | Velocity |
| `0/p_rgh` | Modified pressure `[1 -1 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` — ALWAYS required |
| `0/alphas` | Multi-phase composite field — ALWAYS required for this solver |
| `0/alpha.<phaseName>` | ONE file per phase |
| `constant/g` | Gravity vector — ALWAYS |
| `constant/thermophysicalProperties` | Base: `phases`, `sigmas` (per pair), `pMin` |
| `constant/thermophysicalProperties.<phaseName>` | One per phase |
| `constant/turbulenceProperties` | Only if turbulent; otherwise laminar |

> ⚠️ **No** `constant/transportProperties` — `phases`, `sigmas`, and `pMin` all live in the **base** `constant/thermophysicalProperties`, not in `transportProperties`.

**If turbulence enabled**: generate `0/k`, `0/omega` or `0/epsilon` (match chosen model), and `0/nut` or `0/mut` per distro convention. If laminar: do **NOT** generate any turbulence fields.

## constant/thermophysicalProperties (base) template

```
FoamFile { class dictionary; object thermophysicalProperties; }

phases ( water oil air );   // list ALL phase names here

pMin  pMin  [1 -1 -2 0 0 0 0]  10000;   // Pa

// Surface tension for each phase pair
sigmas
(
    (water oil)   0.05
    (water air)   0.07
    (oil   air)   0.02
);
```

## constant/thermophysicalProperties.\<phaseName\> template

One file per phase. Use a thermo model consistent with the phase type:

- **Liquid phase**: `heRhoThermo` + `rhoConst` (or other liquid EOS)
- **Gas phase**: `hePsiThermo` + `perfectGas`

```
thermoType
{
    type            heRhoThermo;     // or hePsiThermo for gas
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState rhoConst;        // or perfectGas for gas phase
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie          { nMoles 1; molWeight <...>; }
    equationOfState { rho <...>; }   // omit for perfectGas
    thermodynamics  { Cp <...>; Hf 0; }
    transport       { mu <...>; Pr <...>; }
}
```

## fvSolution alpha template (per phase)

```
"alphas.*"
{
    nAlphaCorr      2;
    nAlphaSubCycles 1;
    cAlpha          1;
    MULESCorr       yes;
    nLimiterIter    3;
}

"alpha.*"
{
    nAlphaCorr      2;
    nAlphaSubCycles 1;
    cAlpha          1;
    MULESCorr       yes;
    nLimiterIter    3;
}

p_rgh
{
    solver      GAMG;
    tolerance   1e-7;
    relTol      0.01;
    smoother    DIC;
}
p_rghFinal
{
    $p_rgh;
    relTol 0;
}

U
{
    solver      smoothSolver;
    smoother    symGaussSeidel;
    tolerance   1e-6;
    relTol      0;
}

T
{
    solver      smoothSolver;
    smoother    symGaussSeidel;
    tolerance   1e-8;
    relTol      0;
}

PIMPLE
{
    momentumPredictor           no;
    nOuterCorrectors            1;
    nCorrectors                 2;
    nNonOrthogonalCorrectors    0;
}
```

## Critical rules

1. **`p_rgh` NOT `p`** — MUST_READ. `0/T` ALWAYS required.
2. `constant/g` ALWAYS required.
3. **No `constant/transportProperties`** — `phases`, `sigmas`, and `pMin` belong in the **base** `constant/thermophysicalProperties`.
4. Base `constant/thermophysicalProperties` (phases, sigmas per pair, pMin) + per-phase files are ALL required.
5. **Always generate `0/alphas`** for this solver (required by tutorials and implementation).
6. Generate one `0/alpha.<phaseName>` file for EVERY phase.
7. Alpha values must sum to 1 across all phases at every cell.
8. Generate one `constant/thermophysicalProperties.<phaseName>` for EVERY phase.
9. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
10. Patch names MUST match mesh boundary exactly. Empty patches (2D) → `{ type empty; }` in ALL `0/*` files.
