# buoyantSimpleFoam — system/fvSolution

**Algorithm block**: `SIMPLE`

## Valid OpenFOAM linear solvers, preconditioners, and smoothers

Only use names from these tables. Any other name causes a fatal IO error at runtime.

### Linear solvers

| Keyword | Matrix type | Description |
|---|---|---|
| `PCG` | symmetric only | Preconditioned conjugate gradient for symmetric pressure matrices. |
| `PBiCGStab` | symmetric + asymmetric | Stabilised preconditioned bi-conjugate gradient; recommended general-purpose solver. |
| `smoothSolver` | symmetric + asymmetric | Iterative solver that uses a smoother (must specify `smoother` keyword). |
| `GAMG` | symmetric + asymmetric | Geometric-algebraic multi-grid; fastest for pressure on good meshes. |
| `diagonal` | any | Direct diagonal solver for explicit systems (e.g. rho). |

### Preconditioners (for PCG / PBiCGStab)

| Keyword | Matrix type | Description |
|---|---|---|
| `DIC` | symmetric | Diagonal incomplete-Cholesky; use with PCG for symmetric pressure. |
| `FDIC` | symmetric | Faster DIC with caching; drop-in replacement for DIC. |
| `DILU` | asymmetric | Diagonal incomplete-LU; use with PBiCGStab for velocity/turbulence. |
| `diagonal` | any | Simple diagonal preconditioning; cheap but weak. |
| `GAMG` | any | Multi-grid as preconditioner; expensive but powerful for ill-conditioned systems. |
| `none` | any | No preconditioning. |

### Smoothers (for smoothSolver / GAMG)

| Keyword | Description |
|---|---|
| `GaussSeidel` | Gauss-Seidel; most reliable default smoother for GAMG and smoothSolver. |
| `symGaussSeidel` | Symmetric Gauss-Seidel; better for smoothSolver on asymmetric equation systems. |
| `DIC` | Diagonal incomplete-Cholesky used as smoother; can improve convergence on bad matrices. |
| `DICGaussSeidel` | DIC followed by Gauss-Seidel post-smoothing; best for difficult symmetric systems. |

## Key differences from rhoSimpleFoam

- Solved pressure is `p_rgh`, not `p` — all solver entries, residualControl, and
  relaxationFactors MUST reference `p_rgh`.
- `pRefCell` / `pRefValue` may be needed when no absolute pressure BC is set
  (e.g. closed domain with only fixedFluxPressure boundaries).

## Solver settings

- **p_rgh**: use `GAMG` with smoother `GaussSeidel`. Never `DIC` — causes SIGFPE.
  Include `nCoarsestCells  20;` and `coarsestLevelCorr` with `PBiCGStab; preconditioner none;`.
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
        nCoarsestCells  20;
        coarsestLevelCorr
        {
            solver          PBiCGStab;
            preconditioner  none;
            tolerance       1e-9;
            relTol          0;
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
    nNonOrthogonalCorrectors 1;
    // pRefCell / pRefValue — needed only when NO patch has a fixedValue p_rgh BC
    // (e.g. closed cavity with all fixedFluxPressure boundaries)
    // pRefCell    0;
    // pRefValue   0;
    residualControl
    {
        p_rgh   1e-3;
        U       1e-4;
        h       1e-4;
        k       5e-3;
        omega   5e-3;
        epsilon 5e-3;
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
- SIMPLE's `simpleControl` reads residualControl with `absTolOnly=true` —
  entries are plain scalars (e.g. `p_rgh 1e-3;`). This is the canonical format.
  The dictionary format (`p { tolerance X; relTol 0; }`) is for PIMPLE only.

## Checklist

- [ ] `SIMPLE {}` block (not PIMPLE, not PISO)
- [ ] `p_rgh` uses GAMG + GaussSeidel (not DIC)
- [ ] `residualControl` references `p_rgh` not `p`
- [ ] `relaxationFactors.fields` has `p_rgh` not `p`
- [ ] `rho` solver entry present
- [ ] residualControl entries are plain scalars (e.g. `p_rgh 1e-3;`) — SIMPLE format
