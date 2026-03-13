# Solver: rhoPimpleFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible (pressure-based) · PIMPLE  
**Pressure field**: `p` — absolute static pressure in Pa, dimensions `[1 -1 -2 0 0 0 0]`  
**Energy**: thermo-driven; always initialised from `0/T` (Kelvin).

---

## Prime directive

Generate a **syntactically correct, internally consistent** OpenFOAM case from
`validated_config`.  
**Do NOT refuse or redirect** because a fluid choice, EOS, or temperature range
looks unusual.  If something is ambiguous, apply the conservative defaults in
this file.  Physical realism is the user's responsibility.

---

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application rhoPimpleFoam;` |
| `system/fvSchemes` | Euler ddt; div schemes for U and energy (e or h); wallDist if turbulent |
| `system/fvSolution` | `PIMPLE {}` block; solvers for `p`, `rho`, and transport fields |
| `0/U` | Velocity — all patches |
| `0/p` | Absolute pressure (Pa) |
| `0/T` | Temperature (K) |
| `constant/thermophysicalProperties` | REQUIRED — see selection rules below |
| `constant/turbulenceProperties` | Required when turbulence model is active |
| `0/k`, `0/omega`, `0/nut`, `0/alphat` | kOmegaSST + energy active |
| `0/k`, `0/epsilon`, `0/nut`, `0/alphat` | kEpsilon + energy active |
| `system/fvOptions` | **REQUIRED** — always generate with `limitTemperature` to prevent negative-T divergence |

### Files you must NEVER generate for this solver

- `0/rho` — derived from `thermo.rho()` at runtime (`NO_READ`).
- `0/h` / `0/e` — the thermo package initialises the energy field from `0/T`
  internally.  Providing `0/h` causes `Negative initial temperature T0` because
  OpenFOAM back-converts h→T against its own reference state.
- `constant/g` — not needed for `rhoPimpleFoam` (use `buoyantPimpleFoam` for
  buoyancy-driven flows).

> **`rho` solver entry**: even though `0/rho` is not a file, `rhoPimpleFoam`
> solves `rhoEqn` internally and looks up `rho` in `fvSolution/solvers` at
> runtime.  Omitting it causes
> `"Entry 'rho' not found in dictionary system/fvSolution/solvers"`.
> Always include the diagonal entry (see fvSolution template below).

> **`0/alphat`**: MUST be generated whenever turbulence AND energy are both
> active.  Use `compressible::alphatWallFunction` on wall patches.

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
| compressible gas | `hePsiThermo` | `perfectGas` | `const` (or `sutherland`) |
| liquid with heat transfer OR cryogenic (T < 200 K) | `heRhoThermo` | `icoPolynomial` | `const` |
| isothermal liquid (no heat transfer) | `heRhoThermo` | `rhoConst` | `const` |

**NEVER use `rhoConst` for cryogenic liquids (LN2, LH2, LOX) or when temperature varies significantly.**
For `icoPolynomial`: `rhoCoeffs<8> (a0 a1 0 0 0 0 0 0)` where ρ(T) = a0 + a1·T.
Typical slopes: LN2/LOX (77–120 K) −4.7 kg/m³/K; LH2 (<35 K) −0.7; water/oil >250 K −0.5.
Compute `a0 = ρ_inlet − a1 × T_inlet` from CaseSpec values.

Use `thermo hConst` unless `validated_config` supplies JANAF coefficients, in
which case use `janaf`.

### Step 3 — energy field name

The `energy` keyword controls which variable name the solver transports:

| `energy` setting in `thermoType{}` | Energy variable | Use in fvSchemes / fvSolution |
|---|---|---|
| `sensibleInternalEnergy` | `e` | `div(phi,e)`, regex `(U\|e\|…)` |
| `sensibleEnthalpy` | `h` | `div(phi,h)`, regex `(U\|h\|…)` |

Default when not otherwise specified: `sensibleEnthalpy` → `h`.

### thermophysicalProperties templates

**Gas (perfectGas)**:
```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;            // or sutherland
    thermo          hConst;           // ← MUST be 'thermo' (not 'thermodynamics')
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.97; }
    thermodynamics { Cp 1005; Hf 0; }   // ← MUST be 'thermodynamics' (not 'thermo')
    transport   { mu 1.8e-5; Pr 0.72; }
}
```

**Cryogenic/temperature-varying liquid (icoPolynomial)**:
⚠️  `icoPolynomial` is ONLY valid with `transport=polynomial` + `thermo=hPolynomial`.
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
        // a0 = ρ_inlet − a1*T_inlet; LN2 at 77K, ρ=808: a0=1169.9, a1=−4.7
        rhoCoeffs<8>    (1169.9 -4.7 0 0 0 0 0 0);
    }
}
```

> **Key naming**: inside `thermoType{}` the keyword is `thermo  hConst;`
> Inside `mixture{}` the sub-dict is `thermodynamics { Cp …; }` — not `thermo`.

---

## fvSolution template

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;   // NEVER DIC — crashes with SIGFPE on divergence
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal      { $p; relTol 0; }

    rho
    {
        solver      diagonal;
        tolerance   1e-12;
        relTol      0;
    }
    rhoFinal    { $rho; relTol 0; }

    // h or e — match the energy keyword in thermophysicalProperties
    "(U|h|k|omega|epsilon|alphat)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|h|k|omega|epsilon|alphat)Final"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0;
    }
}

PIMPLE
{
    momentumPredictor   yes;
    transonic           no;
    nOuterCorrectors    2;
    nCorrectors         1;
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    // Under-relaxation helps transient stability with nOuterCorrectors > 1.
    // Remove or set to 1.0 for pure transient runs with nOuterCorrectors 1.
    fields      { p 0.3; rho 0.05; }
    equations   { U 0.9; h 0.7; k 0.9; omega 0.9; epsilon 0.9; }
}
```

> **Energy field in regex**: use `h` or `e` to match the `energy` setting.
> If the model uses `sensibleInternalEnergy`, replace `h` with `e` in the
> regex groups above.

> **Final block**: repeat solver settings explicitly — do NOT use `$"(U|h|…)"` alias
> syntax; OpenFOAM cannot dereference regex-named entries via `$` and will crash.

---

## fvSchemes template (robust)

**Why `bounded Gauss upwind` as default instead of `none`:** OpenFOAM with
`default none` crashes fatally on any div term not explicitly listed — including
internal transient fields like `Ekp` or any turbulence model auxiliary. Using
`bounded Gauss upwind` as a safe fallback prevents these crashes while the
important terms are overridden explicitly with higher-order schemes.

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    // Safe default: bounded upwind covers any internal term the solver requests
    // without crashing on missing entries.
    default                             bounded Gauss upwind;

    div(phi,U)                          bounded Gauss linearUpwind grad(U);

    // energy convection — emit ONLY the one matching thermoType.energy
    div(phi,h)                          bounded Gauss linearUpwind grad(h);   // sensibleEnthalpy
    // div(phi,e)                       bounded Gauss linearUpwind grad(e);   // sensibleInternalEnergy

    // kinetic energy and pressure-work terms — REQUIRED for compressible energy eqn
    div(phi,K)                          bounded Gauss linearUpwind grad(K);
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

> Emit **only one** of `div(phi,h)` / `div(phi,e)` — whichever matches the
> `energy` setting. Do not emit both.
>
> `div(phi,K)` and `div(phid,p)` are **mandatory** for `rhoPimpleFoam`. Omitting
> them with `default none` causes a fatal crash. They are harmless with
> `default bounded Gauss upwind` but should always be listed explicitly.

---

## Numerical safety net (fvOptions)

**Always generate `system/fvOptions`** — temperature can diverge to negative values
during early PIMPLE iterations due to numerical overshoot on boundary interfaces.
Use this template:

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

## flowRateInletVelocity (compressible transient)

For `rhoPimpleFoam`, density is derived from the thermo package — `0/rho` is not a
file.  At **iteration 0**, `rho` has not yet been computed, so `rhoInlet` provides
the fallback density for converting kg/s → m³/s.

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    <value_in_kg_per_s>;   // MANDATORY — actual kg/s value
    // rho      rho;       ← only if provided in validated_config
    // rhoInlet <density>; ← only if provided in validated_config
    value           uniform (0 0 0);        // placeholder only — NOT the flow rate
}
```

For **volumetric** flow rate (m³/s):

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
4. **`rho` solver entry is mandatory** in `fvSolution/solvers` — see fvSolution template.
5. **GAMG smoother must be `GaussSeidel`** — never `DIC` (causes SIGFPE on exit code 136).
6. Energy variable (`h` or `e`) must be consistent across `thermophysicalProperties`,
   `fvSchemes` div entries, and `fvSolution` regex groups.
7. When turbulence + energy are both active, generate `0/alphat` with
   `compressible::alphatWallFunction` on walls.
8. Every mesh patch must appear in every `0/*` field file with a valid BC.
9. `startFrom startTime; startTime 0;` in `controlDict` — never `latestTime`.
10. `controlDict` `endTime` must equal the value provided in `### endTime` of the task.
