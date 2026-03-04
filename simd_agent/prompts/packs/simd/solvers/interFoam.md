# Solver: interFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · Two-phase VOF (Volume of Fluid)  
**Pressure field**: `p_rgh` (pressure minus hydrostatic contribution, dimensions `[1 -1 -2 0 0 0 0]`)  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Gravity file**: ✅ ALWAYS generate `constant/g` (even if gravity = false → use `(0 0 0)`)  
**Alpha field**: ✅ `0/alpha.<phase1Name>` — name MUST match config phases (e.g. `alpha.water`, `alpha.liquid`)

## Phase naming (CRITICAL)

- Let `phase1Name` and `phase2Name` come from `config.phases` (preferred).
- If not provided, default to `(water air)`.
- Alpha file MUST be `0/alpha.<phase1Name>`.
- Do NOT invent `alpha.phase1` unless `phase1Name == "phase1"`.

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application interFoam;` · `startFrom startTime; startTime 0;` · `adjustTimeStep yes;` |
| `system/fvSchemes` | VOF div schemes for alpha (vanLeer); `interpolationSchemes { default linear; }` only |
| `system/fvSolution` | `PIMPLE { }` with alpha controls (`nAlphaCorr`, `nAlphaSubCycles`, `cAlpha`) inside |
| `0/U` | Velocity — all patches |
| `0/p_rgh` | Modified pressure `[1 -1 -2 0 0 0 0]` — all patches |
| `0/alpha.<phase1Name>` | Volume fraction `[0 0 0 0 0 0 0]` — 0=phase2, 1=phase1 |
| `constant/g` | Gravity vector — ALWAYS required |
| `constant/transportProperties` | Two-phase: phases, nu, rho, sigma (surface tension) |
| `constant/turbulenceProperties` | ONLY if turbulent; otherwise laminar |
| `0/k`, `0/omega`, `0/nut` | ONLY if turbulence enabled (e.g. kOmegaSST) |

## constant/g template

```
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);    // or (0 0 0) if gravity=false
```

## constant/transportProperties template (two-phase)

```
phases (water air);   // use actual phase1Name phase2Name from config

water
{
    transportModel  Newtonian;
    nu              1e-6;     // kinematic viscosity [m2/s]
    rho             1000;     // density [kg/m3]
}

air
{
    transportModel  Newtonian;
    nu              1.48e-5;  // air kinematic viscosity
    rho             1;
}

sigma           0.07;  // surface tension [N/m]  — set to 0 if not applicable
```

## controlDict time-step control (recommended)

interFoam is sensitive to the interface Courant number. Prefer automatic time-stepping:

```
adjustTimeStep  yes;
maxCo           1;
maxAlphaCo      0.5;   // often tighter than maxCo for stability
maxDeltaT       <cap>; // problem-dependent
```

Do NOT hardcode `maxCo 0.9` without also setting `maxAlphaCo`.

## fvSolution template

```
solvers
{
    "pcorr.*"   { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }
    p_rgh       { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.05; }
    p_rghFinal  { $p_rgh; relTol 0; }
    U           { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0; }
    k           { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0; }
    omega       { $k; }
}

PIMPLE
{
    momentumPredictor        yes;
    nOuterCorrectors         1;
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;

    // alpha algorithm controls (NOT a linear-solver block)
    nAlphaCorr      2;
    nAlphaSubCycles 1;
    cAlpha          1;
}
```

> **Note**: `nAlphaCorr`, `nAlphaSubCycles`, and `cAlpha` belong **inside** the `PIMPLE` (or `PISO`) algorithm block — NOT as a linear solver entry under `solvers { }`.

## fvSchemes template (VOF-specific)

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                                     none;
    div(rhoPhi,U)                               Gauss linearUpwind grad(U);
    div(phi,alpha)                              Gauss vanLeer;    // bounded alpha transport
    div(phirb,alpha)                            Gauss linear;     // smoother interface compression
    div(((rho*nuEff)*dev2(T(grad(U)))))         Gauss linear;
    // include turbulence terms only if turbulence enabled:
    // div(phi,k)     Gauss linearUpwind grad(k);
    // div(phi,omega) Gauss linearUpwind grad(omega);
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
```

## 2D mesh — empty patches

If the mesh has a patch with `type empty` (e.g. `frontAndBack`), that patch MUST appear as `{ type empty; }` in `0/U`, `0/p_rgh`, and `0/alpha.<phase1Name>`. Use the exact mesh patch name — never invent one.

## Critical rules

1. **`p_rgh` NOT `p`** — interFoam reads `p_rgh`. `0/p_rgh` is required; `0/p` is optional (post-processing only).
2. `constant/g` is ALWAYS required, even when gravity=false (set value to `(0 0 0)`).
3. Alpha field name MUST match the first phase name from config: `alpha.water`, `alpha.liquid`, etc.
4. `div(phi,alpha)` MUST use `Gauss vanLeer` for bounded alpha transport.
5. `div(phirb,alpha)` should use `Gauss linear` (smoother interface option — NOT vanLeer).
6. `interpolationSchemes` block is `{ default linear; }` only — do NOT add `interface interfaceCompression`.
7. Alpha algorithm controls (`nAlphaCorr`, `nAlphaSubCycles`, `cAlpha`) go inside the `PIMPLE`/`PISO` block, NOT as a linear-solver entry.
8. Use `adjustTimeStep yes;` with both `maxCo` and `maxAlphaCo` in `controlDict`.
9. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
10. For inlet alpha: `alpha.<phase1Name> = 1` if pure phase1; `0` if pure phase2.
