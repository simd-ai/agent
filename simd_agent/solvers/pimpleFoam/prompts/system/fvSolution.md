# pimpleFoam — system/fvSolution

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

## PIMPLE block parameters

| Parameter | Typical value | Effect |
|---|---|---|
| `nOuterCorrectors` | 1-3 | Outer PIMPLE iterations per time step. 1 = pure PISO (explicit). 2-3 allows larger deltaT. |
| `nCorrectors` | 1-3 | Inner pressure-velocity corrector iterations. 2 is standard. |
| `nNonOrthogonalCorrectors` | 1-2 | Extra pressure solves for non-orthogonal meshes. 1 is the safe default, 2 for highly skewed. |
| `momentumPredictor` | yes/no | `yes` for high Re; `no` for very low Re or if diverging |

With `nOuterCorrectors > 1`, under-relaxation (< 1) is applied each outer iteration.
With `nOuterCorrectors 1` (pure PISO), relaxation factors should be `1` (no under-relaxation needed).

## Relaxation factors

```
// With nOuterCorrectors 1 (PISO mode) — no relaxation needed:
relaxationFactors { equations { ".*" 1; } }

// With nOuterCorrectors > 1 (PIMPLE mode) — under-relax:
relaxationFactors { equations { U 0.7; ".*" 0.7; } }
```

## Pressure solver selection (handled by validator)

The validator selects the pressure solver based on mesh quality:
- **Good/moderate mesh** (non-ortho < 50 deg): GAMG with GaussSeidel + coarsestLevelCorr PBiCGStab+DIC
- **Poor/unknown mesh** (non-ortho >= 50 deg): PBiCGStab with DIC preconditioner

GAMG crashes in `GAMGSolver::scale` on non-orthogonal meshes because coarse-level
agglomeration creates degenerate matrices. PBiCGStab+DIC is slower but always stable.

## HARD RULES — no compressible contamination

- **NEVER add a `rho` solver entry** — pimpleFoam has no rho equation.
- **NEVER add `alphat` to the solver regex** — that is a compressible thermal diffusivity field.
- **NEVER use `transonic yes`** — transonic flag is for compressible solvers only.
- Turbulence regex MUST only include fields that are actually generated. For laminar: just `U`. For turbulent: build from active fields only (k+omega OR k+epsilon — never both).

## residualControl — CRITICAL format

PIMPLE's `pimpleControl` REQUIRES each residualControl entry to be a **sub-dictionary**.
Include ALL solved fields — p, U, and turbulence (k, omega or k, epsilon):
```
residualControl
{
    p   { tolerance 1e-4; relTol 0; }
    U   { tolerance 1e-4; relTol 0; }
    k   { tolerance 1e-3; relTol 0; }
    omega { tolerance 1e-3; relTol 0; }
}
```

**NEVER use plain scalars** like `p 1e-4;` — they crash with:
`"Residual data for p must be specified as a dictionary"`

This is different from SIMPLE, which accepts plain scalars.

## pFinal / UFinal blocks — CRITICAL

PIMPLE solvers need `Final` solver entries for the last outer corrector iteration.
`$p` alias is safe because `p` is a plain (non-regex) key.
**NEVER use `$"(U|k|omega)"` syntax** — OpenFOAM cannot dereference regex-named entries via `$`.
Repeat solver settings explicitly in Final blocks.

**Note**: fvSolution is generated deterministically by the validator — not by the LLM.
Any LLM-generated version is replaced.

## Complete template — Turbulent (kOmegaSST)

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
        nCoarsestCells  20;
        coarsestLevelCorr
        {
            solver          PBiCGStab;
            preconditioner  DIC;
            tolerance       1e-9;
            relTol          0;
        }
    }
    pFinal
    {
        $p;
        relTol          0;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    "(U|k|omega)Final"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    momentumPredictor   yes;

    residualControl
    {
        p   { tolerance 1e-4; relTol 0; }
        U   { tolerance 1e-4; relTol 0; }
        k   { tolerance 1e-3; relTol 0; }
        omega { tolerance 1e-3; relTol 0; }
    }
}

relaxationFactors
{
    equations { U 0.7; ".*" 0.7; }
}
```

## 2D / 3D notes

- No structural changes to fvSolution for 2D vs 3D — same algorithm, solvers, and relaxation
- `pRefCell 0; pRefValue 0;` — include in PIMPLE block when no fixedValue pressure BC exists
