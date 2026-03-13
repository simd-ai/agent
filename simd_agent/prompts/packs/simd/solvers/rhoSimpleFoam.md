# Solver: rhoSimpleFoam  ·  OpenFOAM v2406

**Type**: Steady-state · Compressible · SIMPLE (pressure-based)  
**Pressure field**: `p` — absolute static pressure in Pa, dimensions `[1 -1 -2 0 0 0 0]`  
**Energy**: solver advances `he = thermo.he()` (field name is `h` or `e`, depending on `thermoType.energy`).  
Always provide `0/T` in Kelvin.  
Do NOT generate `0/h` or `0/e` — we standardise on `0/T` as the only user-specified thermal IC/BC field.  
**Gravity**: No `constant/g` for rhoSimpleFoam.

---

## Prime directive

Generate a **syntactically correct, internally consistent** OpenFOAM case from
`validated_config`.  
**Do NOT refuse or redirect** because a fluid choice, EOS, or temperature range
looks unusual.  If something is ambiguous, apply the conservative defaults in
this file.  Physical realism is the user's responsibility.

---

## Required files

### system/

| File | Notes |
|------|-------|
| `system/controlDict` | `application rhoSimpleFoam;` · `deltaT 1;` · `endTime = <max_iterations>` |
| `system/fvSchemes` | Steady schemes; robust divSchemes; `wallDist` if turbulent |
| `system/fvSolution` | `SIMPLE {}` block; solvers for `p`, `U`, energy (`h` or `e`), turbulence |

### 0/

| File | Notes |
|------|-------|
| `0/U` | Velocity — all patches |
| `0/p` | Absolute pressure in Pa (e.g. `101325`) |
| `0/T` | Temperature in Kelvin — thermo initialises `he` from `T` at startup |
| `0/k`, `0/omega`, `0/nut` | If kOmegaSST |
| `0/k`, `0/epsilon`, `0/nut` | If kEpsilon |
| `0/alphat` | When turbulence AND energy are both active |
| `system/fvOptions` | **REQUIRED** — always generate with `limitTemperature` to prevent negative-T divergence |

### constant/

| File | Notes |
|------|-------|
| `constant/thermophysicalProperties` | REQUIRED |
| `constant/turbulenceProperties` | Always generate (`simulationType` = `laminar` / `RAS` / `LES`) |

### Files you must NEVER generate for this solver

- `0/rho` — density is derived from `thermo.rho()` at runtime.
- `0/h` / `0/e` — thermo initialises the energy field from `0/T` internally.
  Providing `0/h` causes `Negative initial temperature T0` crashes.
- `constant/g` — not needed for `rhoSimpleFoam`.

---

## Thermophysical model selection (CONFIG-DRIVEN)

Build `constant/thermophysicalProperties` from `validated_config` only.

### Step 1 — honour explicit config

If `validated_config` (or `validated_config.physics`) already specifies
`thermoType`, `equationOfState`, or `transport`, use those values **as-is**.
Do not override them.

### Step 2 — conservative defaults when config is silent

| Fluid condition | Default thermoType | Default EOS | Default transport |
|---|---|---|---|
| compressible gas | `hePsiThermo` | `perfectGas` | `const` (or `sutherland` if config says so) |
| liquid with heat transfer OR cryogenic (T < 200 K) | `heRhoThermo` | `icoPolynomial` | `const` |
| isothermal liquid (no heat transfer) | `heRhoThermo` | `rhoConst` | `const` |

**NEVER use `rhoConst` for cryogenic liquids (LN2, LH2, LOX) or when temperature varies significantly.**
For `icoPolynomial`: `rhoCoeffs<8> (a0 a1 0 0 0 0 0 0)` where ρ(T) = a0 + a1·T.
Typical slopes: LN2/LOX (77–120 K) −4.7 kg/m³/K; LH2 (<35 K) −0.7; water/oil >250 K −0.5.
Compute `a0 = ρ_inlet − a1 × T_inlet` from CaseSpec values.

Use `thermo hConst` unless `validated_config` supplies JANAF coefficients, in
which case use `janaf`.

### Step 3 — energy field name

The `energy` keyword controls which variable name is transported:

| `energy` setting in `thermoType{}` | Energy variable | Use in fvSchemes / fvSolution |
|---|---|---|
| `sensibleEnthalpy` | `h` | `div(phi,h)`, entries named `h` |
| `sensibleInternalEnergy` | `e` | `div(phi,e)`, entries named `e` |

Default when not otherwise specified: `sensibleEnthalpy` → `h`.

### thermophysicalProperties templates

**Gas (perfectGas) — `hePsiThermo`**:
```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;           // ← MUST be 'thermo' (not 'thermodynamics')
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.97; }
    thermodynamics { Cp 1005; Hf 0; }  // ← MUST be 'thermodynamics' (not 'thermo')
    transport   { mu 1.8e-5; Pr 0.713; }
}
```

**Cryogenic/temperature-varying liquid — `heRhoThermo` + `icoPolynomial`**:
⚠️  `icoPolynomial` requires `transport=polynomial` + `thermo=hPolynomial`.
`const`+`hConst`+`icoPolynomial` → "Unknown fluidThermo type" fatal error.
```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       polynomial;      // MUST be polynomial
    thermo          hPolynomial;     // MUST be hPolynomial
    equationOfState icoPolynomial;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.97; }
    thermodynamics                   // hPolynomial: CpCoeffs<8> + Hf + Sf (NOT plain Cp)
    {
        Hf              0;
        Sf              0;
        CpCoeffs<8>     (2042 0 0 0 0 0 0 0);
    }
    transport                        // polynomial: muCoeffs + kappaCoeffs (NOT mu/Pr)
    {
        muCoeffs<8>     (1.58e-4 0 0 0 0 0 0 0);
        kappaCoeffs<8>  (0.323 0 0 0 0 0 0 0);  // kappa = mu*Cp/Pr
    }
    equationOfState
    {
        rhoCoeffs<8>    (1169.9 -4.7 0 0 0 0 0 0);  // LN2: a0=ρ−a1*T, a1=−4.7 kg/m³/K
    }
}
```

**Isothermal liquid — `heRhoThermo` + `rhoConst`** (only when T is constant):
```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 18.0; }
    thermodynamics { Cp 4182; Hf 0; }
    transport   { mu 1e-3; Pr 7.0; }
    equationOfState { rho 1000; }
}
```

---

## fvSolution template (steady SIMPLE)

`rhoSimpleFoam` assigns `rho = thermo.rho()` and calls `rho.relax()` inside the
pressure loop — it does **not** call `rhoEqn.solve()`.  A `solvers { rho {} }`
entry is therefore **not required**.  Including it is harmless, but omitting it is
correct.  The `relaxationFactors.fields { rho 0.05; }` entry **is** used by
`rho.relax()` and must be kept.

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;   // NEVER DIC — crashes with SIGFPE on divergence
        tolerance       1e-6;
        relTol          0.1;
    }
    pFinal  { $p; relTol 0; }

    // h or e — match the energy keyword in thermophysicalProperties
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
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p       1e-4;
        U       1e-4;
        h       1e-6;   // use "e" instead if energy = sensibleInternalEnergy
        k       1e-4;
        omega   1e-4;
        epsilon 1e-4;
    }
}

relaxationFactors
{
    fields      { p 0.3; rho 0.05; }
    equations   { U 0.7; h 0.5; e 0.5; k 0.7; omega 0.7; epsilon 0.7; }
}
```

> **Energy field in regex**: replace `h` with `e` in both the regex group and
> `residualControl` if `thermoType.energy` is `sensibleInternalEnergy`.

> **Final block**: repeat solver settings explicitly — do NOT use `$"(U|h|…)"` alias
> syntax; OpenFOAM cannot dereference regex-named entries via `$` and will crash.

---

## fvSchemes template (robust)

The energy equation in `rhoSimpleFoam` uses solver-internal temporary fields
(`K`, `Ekp`) and may include a pressure-work term (`div(phid,p)`) depending
on the build and options.

**Why `bounded Gauss upwind` as default:** OpenFOAM with `default none` will
crash with a fatal "cannot find scheme" error for any div term not explicitly
listed — including internal solver fields like `Ekp` or turbulence model
auxiliaries. The `bounded` keyword enforces conservative (flux-limited) transport,
which is appropriate for steady compressible flows and eliminates unboundedness.
The critical terms (`U`, `h`, `K`, turbulence) are then overridden with
higher-order schemes.

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    // Safe default: bounded upwind covers any internal term the solver requests
    // (Ekp, MRF momentum sources, etc.) without crashing on missing entries.
    default                             bounded Gauss upwind;

    div(phi,U)                          bounded Gauss linearUpwind grad(U);

    // energy convection — emit ONLY the one matching thermoType.energy
    div(phi,h)                          bounded Gauss upwind;   // sensibleEnthalpy
    // div(phi,e)                       bounded Gauss upwind;   // sensibleInternalEnergy

    // kinetic energy and pressure-work terms — REQUIRED for compressible energy eqn
    div(phi,K)                          bounded Gauss upwind;
    div(phi,Ekp)                        bounded Gauss upwind;
    div(phid,p)                         Gauss limitedLinear 1;

    // turbulence convection (include only fields that actually exist)
    div(phi,k)                          bounded Gauss linearUpwind grad(k);
    div(phi,omega)                      bounded Gauss linearUpwind grad(omega);
    div(phi,epsilon)                    bounded Gauss linearUpwind grad(epsilon);

    // viscous stress — must use dev2, not dev
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// Include wallDist ONLY when turbulence is enabled (RAS/LES)
wallDist { method meshWave; }
```

> Emit only one of `div(phi,h)` / `div(phi,e)` — whichever matches the `energy`
> setting. Do not emit both.
>
> `div(phi,K)` and `div(phid,p)` are **mandatory** for `rhoSimpleFoam`. Omitting
> them with `default none` causes a fatal crash. They are harmless with
> `default bounded Gauss upwind` but should always be listed explicitly.

---

## Numerical safety net (fvOptions)

**Always generate `system/fvOptions`** — temperature can diverge to negative values during early SIMPLE iterations. Use this template:

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvOptions;
}

temperatureLimiter
{
    type            limitTemperature;
    active          yes;
    selectionMode   all;
    min             1;        // K — numerical floor, not a physical constraint
    max             100000;   // K — effectively unlimited
}
```

---

## flowRateInletVelocity (compressible)

For `rhoSimpleFoam`, the density field `rho` is derived from the thermo package at
runtime — it is NOT a file.  At **iteration 0**, `rho` has not yet been computed, so
OpenFOAM needs a fallback density value to convert kg/s → m³/s.

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    <value_in_kg_per_s>;   // MANDATORY — the actual kg/s value
    // rho      rho;       ← only if provided in validated_config
    // rhoInlet <density>; ← only if provided in validated_config
    value           uniform (0 0 0);        // placeholder only — NOT the flow rate
}
```

For **volumetric** flow rate (m³/s) with rhoSimpleFoam:

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  <value_in_m3_per_s>;
    value               uniform (0 0 0);
}
```

> **CRITICAL rules**:
> - EXACTLY ONE of `massFlowRate` or `volumetricFlowRate` must be present.
> - `rho` and `rhoInlet` are **optional** — only include them if the user's config provided them.
>   Do NOT invent or add them unless they appear in the BC table.
> - For volumetric flow: never include `rho` or `rhoInlet`.
> - `value uniform (0 0 0)` is a placeholder — NEVER put the flow rate here.
> - `massFlowRate 0` will cause the simulation to diverge immediately (SIGFPE).

---

## Critical rules (solver-runtime correctness)

1. `0/p` is absolute pressure in **Pa**.  Do not use kinematic (m²/s²) values.
2. `0/T` is temperature in **Kelvin**.  This is the only energy IC file you generate.
3. **Never generate `0/h` or `0/e`** — see note above on Negative Temperature crashes.
4. Energy variable (`h` or `e`) must be consistent across `thermophysicalProperties`,
   `fvSchemes` div entries, `fvSolution` regex groups, and `residualControl`.
5. **GAMG smoother must be `GaussSeidel`** — never `DIC` (causes SIGFPE on exit code 136).
6. When turbulence + energy are both active, generate `0/alphat` with
   `compressible::alphatWallFunction` on walls.
7. Always generate `constant/turbulenceProperties` (set `simulationType` to
   `laminar`, `RAS`, or `LES` as appropriate).
8. Every mesh patch must appear in every `0/*` field file with a valid BC.
9. `startFrom startTime; startTime 0;` in `controlDict` — never `latestTime`.
10. `controlDict` `endTime` = `max_iterations` (iteration counter); `deltaT 1`.
11. Prefer `divSchemes default Gauss upwind;` to avoid missing solver-internal
    div names (`Ekp`, `K`, MRF work term).
