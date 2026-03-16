# buoyantSimpleFoam — system/fvSolution

**Algorithm block**: `SIMPLE`

## Key differences from rhoSimpleFoam

- Solved pressure is `p_rgh`, not `p` — all solver entries, residualControl, and
  relaxationFactors MUST reference `p_rgh`.
- `pRefCell` / `pRefValue` may be needed when no absolute pressure BC is set
  (e.g. closed domain with only fixedFluxPressure boundaries).

## Solver settings

- **p_rgh**: use `GAMG` with smoother `GaussSeidel`. Never `DIC` — causes SIGFPE.
  Include `coarsestLevelCorr` with `smoothSolver+symGaussSeidel`.
- **U, h, turbulence, alphat**: use `smoothSolver` with `symGaussSeidel`.
- Use regex group `"(U|h|k|omega|epsilon|alphat)"` for all equation fields.

## Complete template

```
solvers
{
    rho
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0.1;
    }

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

    "(U|h|k|omega|epsilon|alphat)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|h|k|omega|epsilon|alphat)Final"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }
}

SIMPLE
{
    momentumPredictor   no;
    nNonOrthogonalCorrectors 0;
    // pRefCell / pRefValue — needed only when NO patch has a fixedValue p_rgh BC
    // (e.g. closed cavity with all fixedFluxPressure boundaries)
    // pRefCell    0;
    // pRefValue   0;
    residualControl
    {
        p_rgh   { tolerance 1e-3; relTol 0; }
        U       { tolerance 1e-4; relTol 0; }
        h       { tolerance 1e-4; relTol 0; }
        k       { tolerance 5e-3; relTol 0; }
        omega   { tolerance 5e-3; relTol 0; }
        epsilon { tolerance 5e-3; relTol 0; }
    }
}

relaxationFactors
{
    fields    { rho 1.0; p_rgh 0.7; }
    equations { U 0.3; h 0.7; k 0.3; omega 0.3; epsilon 0.3; alphat 0.5; }
}
```

## Notes

- `rho 1.0` relaxation: buoyantSimpleFoam explicitly calls `rho.relax()`.
- `h 0.7` relaxation: typical for moderate ΔT cases.
  If ΔT > 300 K (e.g. fire scenarios), consider reducing to 0.3–0.5.
- `residualControl` entries MUST be sub-dictionaries `{ tolerance X; relTol 0; }`,
  NOT plain scalars — plain scalars crash with "Residual data must be specified as a dictionary".

## Checklist

- [ ] `SIMPLE {}` block (not PIMPLE, not PISO)
- [ ] `p_rgh` uses GAMG + GaussSeidel (not DIC)
- [ ] `residualControl` references `p_rgh` not `p`
- [ ] `relaxationFactors.fields` has `p_rgh` not `p`
- [ ] `rho` solver entry present
- [ ] Each residualControl entry is a dict `{ tolerance X; relTol 0; }`
