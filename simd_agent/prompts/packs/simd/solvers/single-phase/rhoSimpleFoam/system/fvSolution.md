# rhoSimpleFoam — system/fvSolution

**Algorithm block**: `SIMPLE`

## rho solver entry

`rhoSimpleFoam` assigns `rho = thermo.rho()` and calls `rho.relax()` inside the
pressure loop. It does **not** call `rhoEqn.solve()`. A `solvers { rho {} }` entry
is therefore **not required** and must **not** be generated. The `relaxationFactors.fields { rho 0.05; }`
entry **is** used by `rho.relax()` and must be present.

Note: if the auto-injector adds a `rho { solver diagonal; }` block in validation,
that is harmless — but do not generate it yourself.

## Solver settings

- **p**: use `GAMG` with smoother `GaussSeidel`. **Never** use `DIC` — causes SIGFPE (exit 136).
  Always include `coarsestLevelCorr` with `smoothSolver+symGaussSeidel` inside the `p` block.
  The default GAMG coarsest-level solver is PCG+DIC; when the flow diverges and the coarsest
  agglomeration matrix becomes ill-conditioned (zero/negative diagonal), DIC computes a
  reciprocal diagonal of ±∞ → SIGFPE in `DICPreconditioner::calcReciprocalD`.
- **U, h/e, turbulence, alphat**: use `smoothSolver` with `symGaussSeidel`, or `GAMG` with `GaussSeidel`.
- Use a regex group `"(U|h|k|omega|epsilon|alphat)"` to keep the file concise.
  Replace `h` with `e` if `thermoType.energy = sensibleInternalEnergy`.

## SIMPLE block

```
SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent      no;
    residualControl
    {
        p       1e-4;
        U       1e-4;
        h       1e-4;   // replace with "e" if sensibleInternalEnergy
        k       1e-4;
        omega   1e-4;
        epsilon 1e-4;
    }
}
```

**`consistent no` (standard SIMPLE, NOT SIMPLEC)**: SIMPLEC (`consistent yes`) causes the
second non-orthogonal pressure-correction to restart at residual ≈ 1.0 (it over-shoots
the first correction). With standard SIMPLE the corrections are conservative and the
second GAMG solve starts near zero — dramatically reducing `div(phi)`.

**`nNonOrthogonalCorrectors 2`** (not 0 or 1): applies 3 total pressure corrections per
SIMPLE iteration. Each corrector reduces `div(phi)` by ~10x. With 0 correctors the
continuity error `sum local` can reach 1.5e+07 on the first iteration; with 2 correctors
it drops to ~1e+03 — below the threshold where `h × div(phi)` causes artificial cooling.

## Relaxation factors

```
relaxationFactors
{
    fields      { p 0.3; rho 0.05; }
    equations   { U 0.5; h 0.05; k 0.5; omega 0.5; epsilon 0.5; alphat 0.5; }
}
```

CRITICAL: `h 0.05` (not 0.3 or 0.5). The mechanism: with a non-divergence-free phi in
early iterations, the energy RHS has a large artificial cooling term `h × div(phi)`.
Since h is large-negative for cryogenic/sub-ambient cases (T < 298.15 K → h < 0), this
creates massive artificial cooling — T drops to near-zero in 1–2 iterations even with
h=0.3. With h=0.05, T barely moves from the initial condition while velocity converges,
preventing the omega/k/nut divergence cascade.

`rho 0.05` is the relaxation for `rho.relax()` — required even though 0/rho is not generated.

## Complete template

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
        coarsestLevelCorr
        {
            solver      smoothSolver;
            smoother    symGaussSeidel;
            nSweeps     8;
            tolerance   1e-9;
            relTol      0;
        }
    }
    pFinal { $p; relTol 0; }

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
    nNonOrthogonalCorrectors 2;
    consistent      no;
    residualControl
    {
        p       1e-4;
        U       1e-4;
        h       1e-4;
        k       1e-4;
        omega   1e-4;
        epsilon 1e-4;
    }
}

relaxationFactors
{
    fields      { p 0.3; rho 0.05; }
    equations   { U 0.5; h 0.05; k 0.5; omega 0.5; epsilon 0.5; alphat 0.5; }
}
```

## Checklist

- [ ] `SIMPLE {}` block present
- [ ] `p` uses GAMG with `GaussSeidel` smoother (NOT `DIC`)
- [ ] `p` block includes `coarsestLevelCorr { solver smoothSolver; smoother symGaussSeidel; }`
- [ ] Regex group covers all active fields (h or e, turbulence fields, alphat if energy+turbulence)
- [ ] `relaxationFactors.fields { rho 0.05; }` present
- [ ] No `solvers { rho {} }` entry generated
- [ ] `residualControl` covers p, U, energy field, turbulence fields
