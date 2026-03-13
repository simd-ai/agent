# Solver: simpleFoam  ·  OpenFOAM v2406

**Type**: Steady-state · Incompressible · SIMPLE (laminar/RAS/LES optional)  
**Pressure field**: `p` (kinematic pressure), dimensions `[0 2 -2 0 0 0 0]`  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Gravity**: ❌ No `constant/g`

---

## Prime directive

Generate a **syntactically correct, internally consistent** OpenFOAM case from
`validated_config`.  
**Do NOT refuse or redirect** because configuration looks unusual.  Apply
conservative defaults below.  Physical realism is the user's responsibility.

---

## Required files

### system/

| File | Notes |
|------|-------|
| `system/controlDict` | `application simpleFoam;` · `deltaT 1;` · `endTime = <max_iterations>` |
| `system/fvSchemes` | Steady schemes; robust divSchemes; include `wallDist` only when turbulence enabled |
| `system/fvSolution` | MUST include `SIMPLE {}` and `relaxationFactors` |

### 0/

| File | Notes |
|------|-------|
| `0/U` | Velocity — all patches |
| `0/p` | Kinematic pressure `[0 2 -2 0 0 0 0]` — all patches |
| `0/k` | Only if turbulence model uses k (RAS/LES) |
| `0/omega` | Only for kOmegaSST |
| `0/epsilon` | Only for kEpsilon |
| `0/nut` | Only if turbulence enabled (RAS/LES); walls use `nutkWallFunction` |

### constant/

| File | Notes |
|------|-------|
| `constant/transportProperties` | Must define `nu [0 2 -1 0 0 0 0] <value>;` |
| `constant/turbulenceProperties` | Always generate (`simulationType laminar/RAS/LES`) |

**Never generate**: `0/T` · `constant/thermophysicalProperties` · `constant/g`

---

## controlDict (steady)

Treat `endTime` as iteration counter (pseudo-time):

```
startFrom startTime;
startTime 0;
stopAt    endTime;
deltaT    1;
endTime   <max_iterations>;
```

Never use `startFrom latestTime`.

---

## Pressure reference (pRefCell / pRefValue)

Include `pRefCell`/`pRefValue` inside `SIMPLE{}` **only** when the pressure
system is all-Neumann (every `p` patch is `zeroGradient` / `symmetry` / `empty`).  
If any patch uses `fixedValue` for `p` (common at an outlet), omit them.

---

## fvSolution template

Rules:
- `p` uses `GAMG` with `smoother GaussSeidel` — never `DIC` (causes SIGFPE on exit code 136).
- Transport equations (`U`, turbulence) use `smoothSolver` or `PBiCGStab`.
- Use **regex groups** to avoid brittle `$k` / `$U` aliasing across inconsistently
  defined entries.
- Do NOT add `nut` to `solvers` or `relaxationFactors/equations` — it is derived
  by the turbulence model, not solved as a transport equation.

```
solvers
{
    p
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-6;
        relTol      0.1;
    }
    pFinal  { $p; relTol 0; }

    "(U|k|omega|epsilon)"
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-6;
        relTol      0.1;
    }
    "(U|k|omega|epsilon)Final"
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-6;
        relTol      0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;

    // Include ONLY if needed (all-Neumann pressure system):
    // pRefCell  0;
    // pRefValue 0;

    residualControl
    {
        p       1e-4;
        U       1e-4;

        // Only include entries for turbulence fields that are actually generated
        k       1e-4;
        omega   1e-4;
        epsilon 1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p   0.3;
    }
    equations
    {
        U       0.7;

        // Only include if those turbulence equations are active
        k       0.7;
        omega   0.7;
        epsilon 0.7;
    }
}
```

> **Final block**: repeat solver settings explicitly — do NOT use `$"(U|k|…)"` alias
> syntax; OpenFOAM cannot dereference regex-named entries via `$` and will crash.

---

## fvSchemes template (robust)

Using `default bounded Gauss upwind` prevents crashes from missing div scheme
names that turbulence models or solver options may introduce internally.
The important entries are then overridden explicitly for better accuracy.

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                               bounded Gauss upwind;

    div(phi,U)                            bounded Gauss linearUpwind grad(U);

    // turbulence convection (only meaningful if those fields exist)
    div(phi,k)                            bounded Gauss limitedLinear 1;
    div(phi,omega)                        bounded Gauss limitedLinear 1;
    div(phi,epsilon)                      bounded Gauss limitedLinear 1;

    // viscous stress term — must use dev2, not dev
    div((nuEff*dev2(T(grad(U)))))         Gauss linear;
}

laplacianSchemes    { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes       { default corrected; }

// Include wallDist ONLY when turbulence is enabled
wallDist            { method meshWave; }
```

> If you prefer `default none` for strictness, you MUST explicitly include a
> scheme for every divergence term used by the solver and any active turbulence
> model.  The `default bounded Gauss upwind` approach is safer for codegen.

---

## Boundary condition constraints

- Use EXACT patch names from `simulation_config.mesh.patches[].name` (case-sensitive).
- Every patch must appear in every `0/*` field file.
- Respect mesh constraint patches:
  - `empty` patches → `{ type empty; }` in all `0/*` fields
  - `symmetry` patches → `{ type symmetry; }`
- For `0/nut` (when turbulence enabled): walls use `nutkWallFunction`, inlets/outlets
  use `calculated uniform 0;`.

---

## flowRateInletVelocity (incompressible)

When the inlet is driven by a **volumetric** flow rate (m³/s):

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  <value_in_m3_per_s>;
    value               uniform (0 0 0);
}
```

When the inlet is driven by a **mass** flow rate (kg/s):

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    <value_in_kg_per_s>;
    // rhoInlet  <density>;  ← only include if provided in validated_config
    value           uniform (0 0 0);
}
```

> **CRITICAL**: `value uniform (0 0 0)` is a patch-initialisation placeholder only —
> it must NOT contain the flow rate number.
> OpenFOAM fatals without exactly one of `massFlowRate` or `volumetricFlowRate`.
> Only add `rhoInlet` or `rho` if the user explicitly specified them in the config.

---

## Critical rules

1. Pressure `p` is **kinematic** (m²/s²). If the config provides absolute pressure,
   divide by fluid density to convert.
2. Do NOT generate `0/T`, `constant/thermophysicalProperties`, or `constant/g`.
3. `constant/turbulenceProperties` must always be generated; set
   `simulationType laminar;` for laminar cases.
4. Generate `0/nut` only if turbulence is enabled (RAS/LES).
5. Viscous stress term must use `dev2(T(grad(U)))` — not `dev(T(grad(U)))`.
6. Only include `pRefCell`/`pRefValue` when pressure has no `fixedValue` patch.
7. `startFrom startTime; startTime 0;` in `controlDict` — never `latestTime`.
8. `controlDict` `endTime` = `max_iterations` (iteration counter); `deltaT 1`.
9. Do NOT add `nut` to `relaxationFactors` or `fvSolution/solvers`.
10. Every mesh patch (including `empty` patches for 2D) must appear in every `0/*` field file.
