# compressibleInterFoam — system/fvSolution

## PIMPLE settings — CRITICAL

- `nOuterCorrectors` MUST be ≥ 2 — compressibleInterFoam uses icoPolynomial (ρ = f(T) only, zero acoustic compressibility). Without outer iterations the T → ρ → p → U coupling is unresolved per step, causing Co to explode within 2-3 timesteps and T to crash negative.
- `nOuterCorrectors 1` → PISO mode → fatal velocity divergence for liquid-dominant cases.
- `nNonOrthogonalCorrectors` MUST be ≥ 1 — real meshes have non-orthogonality; 0 leaves uncorrected laplacian error in p_rgh.
- `momentumPredictor yes` — required so U is bounded by an explicit solve each outer iteration. With `no`, U accumulates uncorrected pressure-gradient errors.
- `pcorr` solver block is REQUIRED — looked up at runtime by CorrectPhi.H.

## Alpha solver — CRITICAL

The alpha block needs BOTH MULES controls AND a linear solver (solver/smoother/tolerance). Without the linear solver, the alpha pre-solve step fails silently and MULES gets a bad initial flux.

Use `nAlphaSubCycles 2` — sub-cycling alpha within each timestep improves stability at moderate Co numbers.

## Final blocks — CRITICAL

NEVER use `$"(U|T|…)"` or `$U` alias syntax in Final blocks. OpenFOAM cannot dereference regex-named entries via `$`. This causes `Found ERROR but expected }` fatal IO error. **Always repeat solver settings explicitly.**

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

        // MULES controls
        nAlphaCorr          1;
        nAlphaSubCycles     2;
        cAlpha              1;
        MULESCorr           yes;
        nLimiterIter        8;
        alphaApplyPrevCorr  yes;
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
