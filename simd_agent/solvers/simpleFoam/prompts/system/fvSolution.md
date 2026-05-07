# simpleFoam — system/fvSolution

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

## Solvers

- **p**: `GAMG` with `GaussSeidel` smoother. MUST include `nCoarsestCells 500;` and
  `coarsestLevelCorr` with `PBiCGStab; preconditioner none;` — prevents SIGFPE on
  unstructured tet meshes where GAMG over-agglomerates to degenerate coarsest-level
  matrices with zero diagonals.
- **U, turbulence**: `smoothSolver` with `symGaussSeidel`

## SIMPLE block

The SIMPLE block configuration depends on **mesh quality** — the `use_simplec` and
`n_non_ortho_correctors` values in the case spec are derived from real OpenFOAM
`checkMesh` metrics.

### SIMPLEC decision (`consistent` keyword)

- If `use_simplec` is `true`: include `consistent yes;` — enables SIMPLEC for faster
  convergence (typically 3-4x fewer iterations).
- If `use_simplec` is `false` (default): do NOT include the `consistent` keyword at all.
  Standard SIMPLE is used. This is the safe default for unknown or poor-quality meshes.

**Why**: SIMPLEC modifies the pressure equation via H1 correction. On meshes with high
non-orthogonality (>65°) or skewness (>0.8), the modified matrix creates near-zero
entries in GAMG coarse levels → SIGFPE in GAMGSolver::scale.

### Non-orthogonal correctors

Use the `n_non_ortho_correctors` value from case spec:
- `0` — perfectly orthogonal structured grids (non-orthogonality < 5°)
- `1` — general meshes (non-orthogonality < 40°)
- `2` — highly non-orthogonal meshes (non-orthogonality ≥ 40°)

### Template (use_simplec = true, good mesh)

```
SIMPLE
{
    nNonOrthogonalCorrectors {{n_non_ortho_correctors}};
    consistent      yes;
    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega)" 1e-3;
    }
}
```

### Template (use_simplec = false, poor/unknown mesh)

```
SIMPLE
{
    nNonOrthogonalCorrectors {{n_non_ortho_correctors}};

    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega)" 1e-3;
    }
}
```

SIMPLE's `simpleControl` reads residualControl with `absTolOnly=true` — entries
are **plain scalars** (e.g. `p 1e-4;`). This is the canonical format from all
official tutorials (pitzDaily, motorBike, windAroundBuildings).

Do NOT use the dictionary format (`p { tolerance 1e-4; relTol 0; }`) — that
format is for PIMPLE only.

`pRefCell 0; pRefValue 0;` — ALWAYS include. It is harmless when a `fixedValue` pressure
BC exists, but prevents a singular pressure matrix crash when it doesn't (closed domain).
Omitting it risks a fatal error that is hard to diagnose.

## Relaxation factors

CRITICAL: Always include BOTH `fields` and `equations` blocks.

- `fields { p 0.3; }` — pressure field relaxation. Even with SIMPLEC, including
  `p 0.3` is the safe default for automated cases on unknown meshes.
- `equations { U 0.7; k 0.5; omega 0.5; epsilon 0.5; }` — U at 0.7, turbulence at 0.5.
  Turbulence equations need tighter damping (0.5) than velocity (0.7) for stability.
  NEVER use 0.9 for any field — it causes SIGFPE on real-world meshes.
  Do NOT rely on `".*" 0.7` catch-all for turbulence — always list k/omega/epsilon explicitly.

```
relaxationFactors
{
    fields
    {
        p       0.3;
    }
    equations
    {
        U       0.7;
        k       0.5;
        omega   0.5;
        epsilon 0.5;
    }
}
```

## Complete template

```
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
        nCoarsestCells  500;
        coarsestLevelCorr
        {
            solver          PBiCGStab;
            preconditioner  DIC;
            tolerance       1e-9;
            relTol          0;
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
    nNonOrthogonalCorrectors {{n_non_ortho_correctors}};
    // Include "consistent yes;" ONLY if use_simplec is true
    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega)" 1e-3;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.5;
        omega           0.5;
        epsilon         0.5;
    }
}
```

## Pressure solver selection (handled by validator)

The validator selects the pressure solver based on mesh quality:
- **Good/moderate mesh** (non-ortho < 50°): GAMG with GaussSeidel + coarsestLevelCorr PBiCGStab+DIC
- **Poor/unknown mesh** (non-ortho ≥ 50°): PBiCGStab with DIC preconditioner

GAMG crashes in `GAMGSolver::scale` on non-orthogonal meshes because coarse-level
agglomeration creates degenerate matrices. PBiCGStab+DIC is slower but always stable.

**Note**: fvSolution is generated deterministically by the validator — not by the LLM.
Any LLM-generated version is replaced.

## 2D / 3D notes

- No structural changes to fvSolution for 2D vs 3D — same algorithm, solvers, and relaxation
- `pRefCell` / `pRefValue` — always include in both 2D and 3D
