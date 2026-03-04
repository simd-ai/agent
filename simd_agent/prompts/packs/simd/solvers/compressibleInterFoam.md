# Solver: compressibleInterFoam  ·  OpenFOAM v2406

**Type**: Transient · Two-phase VOF · Compressible · Non-isothermal  
**Pressure field**: `p_rgh` (dimensions `[1 -1 -2 0 0 0 0]`) — MUST have `0/p_rgh`  
**Energy equation**: ✅ YES — MUST generate `0/T`  
**Gravity file**: ✅ ALWAYS generate `constant/g` (even if `(0 0 0)`)  
**Alpha field**: ✅ `0/alpha.<phase1Name>` — name follows phase1 name (e.g. `alpha.water`)  
**Thermophysical**: ✅ Base file + separate thermo per phase  

## Phase naming (CRITICAL)

Use phase names from config if provided. Let `phase1Name = config.phases[0]`, `phase2Name = config.phases[1]`.  
If phases are not provided, default to `(water air)`.

- Alpha field: `0/alpha.<phase1Name>` — do **NOT** hardcode `alpha.phase1` unless phase1Name literally is `phase1`.
- Thermo files follow the same naming.

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application compressibleInterFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | VOF + compressible convective schemes; no nonstandard `interpolationSchemes` |
| `system/fvSolution` | PIMPLE + alpha controls + `pcorr` solver block |
| `0/U` | Velocity |
| `0/p_rgh` | Modified pressure `[1 -1 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` — ALWAYS required |
| `0/alpha.<phase1Name>` | Volume fraction |
| `constant/g` | Gravity vector — ALWAYS |
| `constant/thermophysicalProperties` | Base: `phases`, `sigma`, `pMin` |
| `constant/thermophysicalProperties.<phase1Name>` | Thermo for phase 1 |
| `constant/thermophysicalProperties.<phase2Name>` | Thermo for phase 2 |
| `constant/turbulenceProperties` | If turbulent; otherwise laminar |

> ⚠️ **No** `constant/transportProperties` for this solver — `sigma` and `pMin` live in the **base** `constant/thermophysicalProperties`.

**Optional** (do NOT require unless config says):
- `0/p` (some cases write for convenience)
- `0/rho` (READ_IF_PRESENT in some implementations)
- Turbulence fields depending on model (see Turbulence section below)

## constant/thermophysicalProperties (base) template

```
FoamFile { class dictionary; object thermophysicalProperties; }

phases ( <phase1Name> <phase2Name> );

pMin  pMin  [1 -1 -2 0 0 0 0]  10000;   // Pa
sigma sigma [1  0 -2 0 0 0 0]  0.07;    // N/m
```

## constant/thermophysicalProperties.\<phaseName\> template

```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;        // typical for gas phase
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie          { nMoles 1; molWeight <...>; }
    thermodynamics  { Cp <...>; Hf 0; }
    transport       { mu <...>; Pr <...>; }
}
```

For liquid-like phase: `equationOfState rhoConst;` and add `rho <value>;` inside `equationOfState { }`.  
Keep the thermo package consistent with `compressibleInterFoam` expectations.

## fvSolution template

```
solvers
{
    "alpha.<phase1Name>.*"
    {
        nAlphaCorr          1;
        nAlphaSubCycles     1;
        cAlpha              1;
        MULESCorr           yes;
        nLimiterIter        8;
        alphaApplyPrevCorr  yes;
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

    pcorr
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
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

    // if turbulence enabled, include k/omega or k/epsilon etc.
}

PIMPLE
{
    momentumPredictor           no;
    nOuterCorrectors            1;
    nCorrectors                 2;
    nNonOrthogonalCorrectors    0;
}
```

## fvSchemes template

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                         none;
    div(rhoPhi,U)                   Gauss linearUpwind grad(U);
    div(phi,alpha)                  Gauss vanLeer;
    div(phirb,alpha)                Gauss linear;
    div(rhoPhi,T)                   Gauss linearUpwind grad(T);
    div(rhoPhi,K)                   Gauss linearUpwind grad(K);
    div(phi,p)                      Gauss linearUpwind grad(p);
    div(phi,k)                      Gauss linearUpwind grad(k);
    div(phi,omega)                  Gauss linearUpwind grad(omega);
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes    { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes       { default corrected; }
fluxRequired        { default no; p_rgh; pcorr; alpha.<phase1Name>; }
wallDist            { method meshWave; }
```

> ⚠️ Do **NOT** add `interface interfaceCompression` under `interpolationSchemes` — that is nonstandard and will cause errors.

## Turbulence fields

- If `turbulenceProperties` says **laminar**: do **NOT** generate `k`, `omega`, `epsilon`, `mut`, or `nut`.
- If **RAS/LES** enabled: generate only what the selected model needs.
- For compressible turbulence: use **`0/mut`** (µt) rather than `0/nut` — compressible models use dynamic viscosity.

## Critical rules

1. **`p_rgh` NOT `p`** — solver reads `p_rgh` as MUST_READ. `0/T` is ALWAYS required.
2. `constant/g` is ALWAYS required.
3. **No `constant/transportProperties`** — `sigma` and `pMin` belong in `constant/thermophysicalProperties` (base).
4. Provide a **base** `constant/thermophysicalProperties` (phases, sigma, pMin) **plus** per-phase files.
5. Alpha field name **follows phase1 name**: `0/alpha.<phase1Name>` (e.g. `0/alpha.water`).
6. `pcorr` solver block is required in `fvSolution`.
7. Alpha controls (`nAlphaCorr`, `MULESCorr`, etc.) live inside the alpha field solver block — not as a separate linear-solver block.
8. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
9. Patch names MUST match mesh boundary exactly. Empty patches (2D) → `{ type empty; }` in ALL `0/*` files.
