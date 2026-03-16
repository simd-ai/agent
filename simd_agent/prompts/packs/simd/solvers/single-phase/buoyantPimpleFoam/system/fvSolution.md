# buoyantPimpleFoam — system/fvSolution

**Algorithm block**: `PIMPLE`

## Key differences from rhoPimpleFoam

- Solved pressure is `p_rgh` — ALL solver entries, residualControl, relaxationFactors
  MUST reference `p_rgh`.
- `nOuterCorrectors ≥ 2` REQUIRED: the ρ–T–p_rgh coupling in buoyant flows requires
  at least 2 outer PIMPLE loops to converge the density-driven momentum source.
  Using `nOuterCorrectors 1` (PISO mode) leaves the buoyancy coupling unresolved.

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
        tolerance       1e-8;
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
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|h|k|omega|epsilon|alphat)Final"
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   yes;
    nOuterCorrectors    2;      // MINIMUM 2 for buoyancy coupling; increase to 3-5 if unstable
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    residualControl
    {
        p_rgh   { tolerance 1e-4; relTol 0; }
        U       { tolerance 1e-4; relTol 0; }
        h       { tolerance 1e-4; relTol 0; }
    }
}

relaxationFactors
{
    fields    { rho 1.0; p_rgh 0.7; }
    equations { U 0.5; h 0.7; k 0.5; omega 0.5; epsilon 0.5; alphat 0.5; }
}
```

## Notes

- `nOuterCorrectors 2` is the minimum. For strong natural convection (Ra > 10⁸) or
  fires/large ΔT, use 3–5.
- `h relTol 0` in the equation solver: buoyancy resets h residual each outer loop;
  using `relTol 0.1` would exit after 1 iteration when relTol is satisfied relative
  to the reset residual, leaving h under-converged.
- `residualControl` entries MUST be sub-dictionaries `{ tolerance X; relTol 0; }`.

## Checklist

- [ ] `PIMPLE {}` block (not SIMPLE)
- [ ] `nOuterCorrectors ≥ 2`
- [ ] `p_rgh` uses GAMG + GaussSeidel
- [ ] `residualControl` uses `p_rgh` not `p`
- [ ] `relaxationFactors.fields` has `p_rgh` not `p`
- [ ] `rho` solver entry present
- [ ] Each residualControl entry is a dict `{ tolerance X; relTol 0; }`
