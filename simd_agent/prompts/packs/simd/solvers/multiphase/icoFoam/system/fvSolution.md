# icoFoam — system/fvSolution

**Must use `PISO {}` block** — not SIMPLE or PIMPLE.

Add `pRefCell 0; pRefValue 0;` inside `PISO {}` ONLY if there is no `fixedValue` pressure BC anywhere (all-Neumann system).

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.05;
        coarsestLevelCorr
        {
            solver      smoothSolver;
            smoother    symGaussSeidel;
            nSweeps     8;
            tolerance   1e-9;
            relTol      0;
        }
    }
    pFinal
    {
        $p;
        relTol 0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0;
    }
}

PISO
{
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;

    // Uncomment only if ALL pressure BCs are zeroGradient (no fixedValue p):
    // pRefCell  0;
    // pRefValue 0;
}
```
