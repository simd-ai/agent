# compressibleInterIsoFoam — system/fvSolution

## PIMPLE settings — CRITICAL

Same rules as compressibleInterFoam:
- `nOuterCorrectors` MUST be ≥ 2 — zero acoustic compressibility (icoPolynomial) requires outer iterations to couple T → ρ → p → U. `nOuterCorrectors 1` (PISO) → velocity divergence within 2-3 timesteps.
- `nNonOrthogonalCorrectors` MUST be ≥ 1.
- `momentumPredictor yes` — bounds U growth each outer iteration.
- `pcorr` block REQUIRED — looked up at runtime by CorrectPhi.H.

## isoAdvection alpha — KEY DIFFERENCE from compressibleInterFoam

- `reconstructionScheme isoAlpha` (or `plicRDF` for sharper interfaces) — this is what distinguishes compressibleInterIsoFoam.
- The alpha block still needs `solver`/`smoother`/`tolerance` for the pre-solve step.

## Final blocks — CRITICAL

NEVER use `$U` alias in Final blocks. `U` is defined inside a regex pattern key; `$U` cannot resolve it. Always repeat solver settings explicitly.

## Template

```
solvers
{
    "alpha.<phase1Name>.*"
    {
        // Linear solver (required for pre-solve step)
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;

        // isoAdvection controls
        nAlphaCorr          1;
        nAlphaSubCycles     2;
        cAlpha              1;
        MULESCorr           yes;
        nLimiterIter        8;
        alphaApplyPrevCorr  yes;

        reconstructionScheme isoAlpha;   // KEY: distinguishes from compressibleInterFoam
    }

    p_rgh
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-7;
        relTol      0.01;
    }
    p_rghFinal
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-8;
        relTol      0;
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
        relTol      0.1;
        nSweeps     1;
    }
    UFinal
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-7;
        relTol      0;
        nSweeps     1;
    }

    T
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-6;
        relTol      0.1;
        nSweeps     1;
    }
    TFinal
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-7;
        relTol      0;
        nSweeps     1;
    }

    "(k|omega|epsilon)"
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-6;
        relTol      0.1;
        nSweeps     1;
    }
    "(k|omega|epsilon)Final"
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-7;
        relTol      0;
        nSweeps     1;
    }
}

PIMPLE
{
    // CRITICAL: nOuterCorrectors >= 2 — see rules above
    momentumPredictor           yes;
    nOuterCorrectors            2;
    nCorrectors                 2;
    nNonOrthogonalCorrectors    1;
}
```
