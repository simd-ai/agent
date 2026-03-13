# Solver: pimpleFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · PIMPLE (merged SIMPLE+PISO)  
**What it solves**: Incompressible Newtonian flow; turbulence model can be `laminar` / `RAS` / `LES` (from config).  
**Pressure field**: `p` (kinematic pressure), dimensions `[0 2 -2 0 0 0 0]`  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Gravity**: ❌ No `constant/g` (use buoyant solvers for buoyancy)

---

## Required files

### system/
| File | Notes |
|------|------|
| `system/controlDict` | `application pimpleFoam;` · transient physical time (`deltaT`, `endTime`) |
| `system/fvSchemes` | transient schemes + convection + viscous term |
| `system/fvSolution` | MUST contain `PIMPLE {}` and solver entries |

### 0/
| File | Notes |
|------|------|
| `0/U` | Velocity — all patches |
| `0/p` | Kinematic pressure — all patches |
| `0/k` | Only if turbulence model uses k (RAS/LES) |
| `0/omega` | Only for kOmegaSST |
| `0/epsilon` | Only for kEpsilon |
| `0/nut` | Only if turbulence enabled (RAS/LES) |

### constant/
| File | Notes |
|------|------|
| `constant/transportProperties` | Must define kinematic viscosity `nu` |
| `constant/turbulenceProperties` | Always generate (set `simulationType laminar/RAS/LES`) |

**Never generate**: `0/T` · `constant/thermophysicalProperties` · `constant/g`

---

## controlDict template (transient)

Rule: `endTime` and `deltaT` are physical time controls (seconds), not iteration count.

```
application      pimpleFoam;

startFrom        startTime;
startTime        0;
stopAt           endTime;

deltaT           <solver.delta_t>;      // seconds
endTime          <solver.endTime>;      // seconds

writeControl     timeStep;
writeInterval    <solver.write_interval>;

purgeWrite       0;
runTimeModifiable true;

// Optional for robustness
adjustTimeStep   yes;
maxCo            0.9;
```

Never use `startFrom latestTime`.

---

## constant/transportProperties (incompressible)

Use ONE consistent syntax:

```
transportModel   Newtonian;
nu               [0 2 -1 0 0 0 0] <nu_value>;
```

---

## constant/turbulenceProperties

Always generate this file. Choose `simulationType` from config:

- laminar case:
```
simulationType laminar;
```

- RAS case:
```
simulationType RAS;
RAS
{
    RASModel        <modelName>;   // e.g. kOmegaSST or kEpsilon
    turbulence      on;
    printCoeffs     on;
}
```

- LES case:
```
simulationType LES;
LES
{
    LESModel        <modelName>;
    turbulence      on;
    printCoeffs     on;
}
```

---

## fvSolution template (robust)

Rules:
- `p` uses `GAMG` with `GaussSeidel` smoother.
- Transport fields (`U`, turbulence) use `smoothSolver` or `PBiCGStab`.
- Prefer regex groups to avoid alias mistakes.

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.05;
    }
    pFinal
    {
        $p;
        relTol 0;
    }

    "(U|k|omega|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|k|omega|epsilon)Final"
    {
        $U;
        relTol 0;
    }
}

PIMPLE
{
    nOuterCorrectors         1;
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;

    // Only needed if pressure is all-Neumann (no fixedValue p anywhere)
    // pRefCell  0;
    // pRefValue 0;
}
```

Notes:
- Under-relaxation is often unnecessary for transient PIMPLE; include `relaxationFactors` only if you are intentionally damping divergence.

---

## fvSchemes template (robust for codegen)

The viscous stress divergence term must use `dev2(T(grad(U)))`.

Recommended robust scheme choice:

```
ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default Gauss linear;
}

divSchemes
{
    default                       bounded Gauss upwind;

    div(phi,U)                    bounded Gauss linearUpwind grad(U);

    // turbulence convection (only relevant if the fields exist)
    div(phi,k)                    bounded Gauss limitedLinear 1;
    div(phi,omega)                bounded Gauss limitedLinear 1;
    div(phi,epsilon)              bounded Gauss limitedLinear 1;

    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }

// Include wallDist ONLY when turbulence enabled (RAS/LES with walls)
wallDist
{
    method meshWave;
}
```

If you insist on `divSchemes { default none; }`, you must define every div term that appears (more brittle).

---

## Boundary condition constraints (runtime correctness)

- Use EXACT patch names from `simulation_config.mesh.patches[].name` (case-sensitive).
- Every patch must appear in every generated `0/*` file.
- Respect constraint patches:
  - `empty` → `{ type empty; }`
  - `symmetry`/`symmetryPlane` → `{ type symmetry; }`

Turbulence field BC notes (when turbulence enabled):
- Walls: `k`, `omega/epsilon`, `nut` should use appropriate wall functions.
- Non-walls: use `zeroGradient` or `fixedValue` depending on patch role.

---

## flowRateInletVelocity (incompressible transient)

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
> never put the flow rate here. OpenFOAM fatals without exactly one of
> `massFlowRate` or `volumetricFlowRate`.
> Only add `rhoInlet` or `rho` if the user explicitly specified them in the config.

---

## Critical rules

1. MUST generate `0/p` and `0/U` (pimpleFoam reads them at startup).
2. MUST include `PIMPLE {}` in `fvSolution`.
3. Generate turbulence fields (`0/k`, `0/omega`, `0/epsilon`, `0/nut`) only when turbulence is enabled and the chosen model requires them.
4. Do NOT generate buoyancy/thermo files (`constant/g`, `constant/thermophysicalProperties`, `0/T`) for `pimpleFoam`.
5. Only include `pRefCell/pRefValue` when pressure is all-Neumann (no `fixedValue` on any patch).
6. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
