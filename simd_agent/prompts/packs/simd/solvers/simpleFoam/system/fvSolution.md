# simpleFoam — system/fvSolution

**Algorithm block**: `SIMPLE`

## Solvers

- **p**: `GAMG` with `GaussSeidel` smoother (preferred over PCG for robustness on meshes with non-orthogonality)
- **U, turbulence**: `smoothSolver` with `symGaussSeidel`

## SIMPLE block

```
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    // Add pRefCell/pRefValue ONLY when no fixed-value pressure BC exists:
    // pRefCell        0;
    // pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega)" 1e-3;
    }
}
```

`consistent yes` enables SIMPLEC — more stable convergence.

`pRefCell 0; pRefValue 0;` is needed when the domain has no `fixedValue` pressure BC
(e.g. all outlets are `zeroGradient`). With an outlet at fixed pressure, omit these.

## Relaxation factors

Conservative (safe for most meshes):
```
relaxationFactors
{
    equations
    {
        U       0.7;
        ".*"    0.7;
    }
}
```

Aggressive (faster convergence, good mesh required):
```
relaxationFactors
{
    equations
    {
        U       0.9;
        ".*"    0.9;
    }
}
```

## Complete template (from official pitzDaily tutorial)

```
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
        coarsestLevelCorr
        {
            solver      smoothSolver;
            smoother    symGaussSeidel;
            nSweeps     8;
            tolerance   1e-9;
            relTol      0;
        }
    }

    "(U|k|epsilon|omega|f|v2)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega)" 1e-3;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.9;
    }
}
```
