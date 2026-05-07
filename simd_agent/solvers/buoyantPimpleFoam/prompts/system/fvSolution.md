# buoyantPimpleFoam — system/fvSolution

**Algorithm block**: `PIMPLE`

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
        nCoarsestCells  500;
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
