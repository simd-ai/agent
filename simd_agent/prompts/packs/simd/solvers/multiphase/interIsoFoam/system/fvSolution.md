# interIsoFoam — system/fvSolution

**isoAdvection alpha controls go in a separate `alphaControls {}` or `isoAdvection {}` block** — NOT inside `solvers {}`.
Do NOT use `solver isoAdvector;`.

```
solvers
{
    p_rgh
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
        coarsestLevelCorr
        {
            solver      smoothSolver;
            smoother    symGaussSeidel;
            nSweeps     8;
            tolerance   1e-9;
            relTol      0;
        }
    }
    p_rghFinal { $p_rgh; relTol 0; }

    "(U|k|omega|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    // Do NOT use $U here — there is no plain 'U' key in solvers{}, only the
    // regex key "(U|k|omega|epsilon)". OpenFOAM cannot dereference a regex-named
    // entry via $ and will crash. Repeat the solver settings explicitly.
    "(U|k|omega|epsilon)Final"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   yes;
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 0;
}

// isoAdvection reconstruction controls
isoAdvection
{
    nAlphaCorr          1;
    nAlphaSubCycles     1;
    cAlpha              1;

    reconstructionScheme plicRDF;
    vof2IsoTol          1e-8;
    surfCellTol         1e-6;
    nAlphaBounds        3;
    snapTol             1e-12;
    clip                true;
}
```
