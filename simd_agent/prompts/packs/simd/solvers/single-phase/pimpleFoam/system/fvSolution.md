# pimpleFoam — system/fvSolution

**Algorithm block**: `PIMPLE`

## PIMPLE block parameters

| Parameter | Typical value | Effect |
|---|---|---|
| `nOuterCorrectors` | 1–3 | Outer PIMPLE iterations per time step. 1 = pure PISO (explicit). 2–3 allows larger deltaT. |
| `nCorrectors` | 1–3 | Inner pressure-velocity corrector iterations. 2 is standard. |
| `nNonOrthogonalCorrectors` | 0–2 | Extra pressure solves for non-orthogonal meshes. 0 for structured, 1–2 for skewed. |
| `momentumPredictor` | yes/no | `yes` for high Re; `no` for very low Re or if diverging |

With `nOuterCorrectors > 1`, under-relaxation (< 1) is applied each outer iteration.
With `nOuterCorrectors 1`, relaxation factors should be `1` (no under-relaxation needed).

## Relaxation factors

```
// With nOuterCorrectors 1 (PISO mode) — no relaxation needed:
relaxationFactors { equations { ".*" 1; } }

// With nOuterCorrectors > 1 (PIMPLE mode) — under-relax:
relaxationFactors { equations { U 0.7; ".*" 0.7; } }
```

## HARD RULES — no compressible contamination

- **NEVER add a `rho` solver entry** — pimpleFoam has no rho equation.
- **NEVER add `alphat` to the solver regex** — that is a compressible thermal diffusivity field.
- **NEVER use `transonic yes`** — transonic flag is for compressible solvers only.
- Turbulence regex MUST only include fields that are actually generated. For laminar: just `U`. For turbulent: build from active fields only (k+omega OR k+epsilon — never both).

## Template A — Laminar

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    UFinal
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
    nNonOrthogonalCorrectors 0;
    momentumPredictor   yes;

    residualControl
    {
        U   1e-4;
        p   1e-4;
    }
}

relaxationFactors
{
    equations { U 0.7; ".*" 0.7; }
}
```

## Template B — Turbulent (kOmegaSST / kEpsilon)

Adjust the regex to match the active turbulence model — k+omega for kOmegaSST, k+epsilon for kEpsilon.
`$p` alias is safe here because `p` is a plain (non-regex) key.

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
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
    pFinal
    {
        $p;
        relTol          0;
    }

    // kOmegaSST: "(U|k|omega)"   kEpsilon: "(U|k|epsilon)"
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    // Repeat settings explicitly — do NOT use $U (no plain 'U' key to dereference)
    "(U|k|omega)Final"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 0;
    momentumPredictor   yes;

    residualControl
    {
        U   1e-4;
        p   1e-4;
    }
}

relaxationFactors
{
    equations { U 0.7; ".*" 0.7; }
}
```

Note: `pFinal { $p; relTol 0; }` alias is valid — `p` is a plain key. Do NOT use `$"(U|k|omega)"` syntax for regex keys (OpenFOAM cannot dereference regex-named entries via `$`).
