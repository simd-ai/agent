# Solver: rhoPimpleFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible · PIMPLE (pressure-based compressible)  
**Pressure field**: `p` (absolute static pressure in Pa, dimensions `[1 -1 -2 0 0 0 0]`)  
**Energy equation**: ✅ YES — the solver transports `he = thermo.he()` internally (`h` or `e`).  
  ⚠️ **Do NOT generate `0/h` or `0/e`.** The thermo package initialises the energy field from `0/T` at startup — providing `0/h` causes "Negative initial temperature" because back-conversion from h→T is reference-sensitive.  
  **Always provide `0/T` with temperature boundary conditions (in Kelvin).**  
**Gravity file**: ❌ No (use buoyantPimpleFoam if buoyancy-driven natural convection needed)  

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application rhoPimpleFoam;` · deltaT = solver.delta_t · endTime = solver.endTime |
| `system/fvSchemes` | Euler ddt · upwind divSchemes; include `div(phi,e)` OR `div(phi,h)` matching thermo energy setting |
| `system/fvSolution` | `PIMPLE { }` block; solvers for p/pFinal and regex group for rho/U/e (or h)/k/omega or epsilon |
| `0/U` | Velocity — all patches (MUST_READ by solver) |
| `0/p` | Absolute pressure `[1 -1 -2 0 0 0 0]` in Pa |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` in **Kelvin** — the thermo reads T and initialises h/e from it |
| `constant/thermophysicalProperties` | REQUIRED — defines thermoType, EOS, transport, and energy setting |
| `constant/turbulenceProperties` | Required if turbulent |
| `0/k`, `0/omega`, `0/nut`, `0/alphat` | If kOmegaSST + energy active |
| `0/k`, `0/epsilon`, `0/nut`, `0/alphat` | If kEpsilon + energy active |
| `constant/fvOptions` | **Conditionally required** — MUST include for cryogenic/liquid cases; use `limitTemperature` to suppress `Negative Temperature` divergence during startup (see template below) |

> **`0/rho`**: do NOT generate — rho is `NO_READ`; derived from `thermo.rho()`. However, `rho { }` MUST be in `fvSolution/solvers`.  
> **`0/h` / `0/e`**: **NEVER generate these files.** The thermo package initialises h/e from `0/T` internally. Providing 0/h causes `Negative initial temperature T0` crashes.  
> **`0/alphat`**: MUST generate when turbulence + energy are both active; use `compressible::alphatWallFunction` on walls.

## constant/thermophysicalProperties template

```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;      // or const; sutherland preferred for compressible
    thermo          janaf;           // or hConst
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;  // -> energy field name is "e"
                                             // use sensibleEnthalpy  -> field name is "h"
}

mixture
{
    specie
    {
        nMoles          1;
        molWeight       28.97;   // Air: 28.97 | LN2: 28.014
    }
    thermodynamics
    {
        Cp              1005;    // J/kg/K — from fluid config (used if hConst)
        Hf              0;
    }
    transport
    {
        mu              1.8e-5;  // Dynamic viscosity — from fluid config (used if const)
        Pr              0.72;    // Prandtl number: mu*Cp/lambda
    }
}
```

> **Energy field name rule**: if `energy sensibleInternalEnergy` → field is `e`; if `energy sensibleEnthalpy` → field is `h`. This determines which `div(phi,e)` / `div(phi,h)` and solver entries to generate.

---

## Thermophysical Model Selection

If the user is simulating a fluid, apply the following logic to select the correct thermophysical model:

### Single-Phase Gas
**Condition**: Operating temperature is significantly above the fluid's boiling point (e.g., air or N₂ at room temperature).  
**Model**:
```
type            hePsiThermo;
equationOfState perfectGas;
thermo          hConst;    // or janaf for variable Cp
transport       sutherland; // or const
```

### Single-Phase Liquid — Stable (water, oil, sub-cooled fluids far from boiling point)
**Condition**: Fluid is a **stable liquid that does NOT boil** at the operating temperature (e.g., water at 20 °C, oil at 50 °C, water cooling circuit well below 100 °C).  
**Model**:
```
type            heRhoThermo;    // NOT hePsiThermo — rhoConst requires heRhoThermo
equationOfState rhoConst;       // constant density — ONLY valid when T change is small
thermo          hConst;
transport       const;
```
> ⚠️ **Set `0/p` outlet to the actual operating pressure (e.g., `101325` Pa). NEVER use `0`.** With `rhoConst`, pressure is decoupled from density — an outlet of 0 Pa creates unphysical negative pressures that propagate into temperature divergence.

### Single-Phase Liquid — Cryogenic (LN₂, LH₂, LOX, or any fluid below ~200 K)
**Condition**: Fluid is a **cryogenic liquid** (LN₂, LH₂, LOX, etc.) or the operating temperature is below ~200 K.

> 🚫 **NEVER use `equationOfState rhoConst` for cryogenic fluids.** LN₂ boils at 77 K, LH₂ at 20 K — a hot wall (e.g., 300–400 K) will cause the fluid temperature to rise dramatically above the boiling point, making `rhoConst` (constant density) physically wrong. The solver will produce wildly incorrect results or crash with `Negative Temperature`.

**Model** (temperature-varying density):
```
type            heRhoThermo;
equationOfState icoPolynomial;  // T-dependent ρ — provide rhoCoeffs polynomial
thermo          hConst;
transport       const;
energy          sensibleEnthalpy;
```
Alternatively, if the user intends to model phase-change (LN₂ boiling), escalate to a two-phase solver (`interFoam` or `compressibleInterFoam`) and flag the user.

### Cryogenic Safety Rule
**Applies to**: any cryogenic fluid (LN₂, LOX, LH₂, etc.) or any case where cold initial conditions coexist with warm wall/inlet boundaries (ΔT > 50 K with cold side below 200 K).  
**Constraint**: Always generate `constant/fvOptions` containing a `limitTemperature` source. This prevents the `Negative initial temperature T0` / `Negative Temperature` crash that occurs during the first ~100 iterations when numerical overshoot drives `T → 0` or negative.

---

## constant/fvOptions template (Cryogenic / Liquid cases)

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      fvOptions;
}

temperatureLimiter
{
    type            limitTemperature;
    active          yes;

    selectionMode   all;

    min             1;       // absolute floor [K] — blocks T going to 0 or negative
    max             100000;  // effectively unlimited ceiling
}
```

> `limitTemperature` is a standard OpenFOAM `fvOption`. It clips the temperature field each iteration. This is a **safety net only** — if T is still hitting the floor after iteration ~200 the root cause (wrong BCs, wrong EOS, wrong pressure outlet) must be fixed.

---

## fvSolution template

> **Solver selection rule:** `GAMG` for symmetric elliptic equations (pressure `p`).
> `smoothSolver` / `PBiCGStab` for asymmetric transport equations (`U`, `h`/`e`, turbulence).

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal      { $p; relTol 0; }

    // ✅ REQUIRED — rhoPimpleFoam solves rhoEqn; OpenFOAM looks up this entry at runtime.
    // Missing it causes: "Entry 'rho' not found in dictionary system/fvSolution/solvers"
    rho
    {
        solver      diagonal;
        tolerance   1e-12;
        relTol      0;
    }
    rhoFinal    { $rho; relTol 0; }

    // Energy field: use "h" if energy=sensibleEnthalpy, "e" if energy=sensibleInternalEnergy
    // Include alphat only if turbulence + energy both active
    "(U|h|e|k|omega|epsilon|alphat)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|h|e|k|omega|epsilon|alphat)Final"
    {
        $U;
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
    fields      { p 0.3; rho 0.05; }
    equations   { U 0.9; h 0.7; e 0.7; k 0.9; omega 0.9; epsilon 0.9; }
}
```

## fvSchemes template

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default             none;
    div(phi,U)          Gauss linearUpwind grad(U);
    div(phi,e)          Gauss linearUpwind grad(e);   // if sensibleInternalEnergy
    div(phi,h)          Gauss linearUpwind grad(h);   // if sensibleEnthalpy
    div(phi,K)          Gauss linearUpwind grad(K);
    div(phi,k)          Gauss linearUpwind grad(k);
    div(phi,omega)      Gauss linearUpwind grad(omega);
    div(phi,epsilon)    Gauss linearUpwind grad(epsilon);
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
    div(phid,p)         Gauss limitedLinear 1;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
```

## Critical rules

1. `0/p` is ABSOLUTE pressure in Pa (not kinematic). Inlet/outlet values from config × density if given in bar: 1 bar = 100 000 Pa.
2. `0/T` is **ALWAYS required** and contains temperature values **in Kelvin**. This is the ONLY energy-related field file you generate.
3. **NEVER generate `0/h` or `0/e`** — the thermo package initialises the energy field from `0/T` at startup. Providing them causes `Negative initial temperature T0` (OpenFOAM back-converts h→T and gets a nonsensical value). Standard tutorials (e.g., `$FOAM_TUTORIALS/compressible/rhoPimpleFoam/`) all omit `0/h`.
4. In `fvSolution` and `fvSchemes`, the energy variable is **`e`** (sensibleInternalEnergy) or **`h`** (sensibleEnthalpy) — never `T`. Match the regex group and `div` scheme to the `energy` setting in `thermophysicalProperties`.
5. When turbulence + energy are both active, generate `0/alphat` with `compressible::alphatWallFunction` on walls. Missing `alphat` will cause runtime errors.
6. Fix aliasing: `Final` solver groups must reference their own group (`$U;`), not a different variable. Do NOT write `omegaFinal { $U; }` if omega is in a regex group — use the `"...Final"` regex alias instead.
7. **CRITICAL — `rho` solver entry is REQUIRED in `fvSolution/solvers`**: `rhoPimpleFoam` solves `rhoEqn` internally. Omitting it causes `"Entry 'rho' not found in dictionary system/fvSolution/solvers"`. Always include:
   ```
   rho      { solver diagonal; tolerance 1e-12; relTol 0; }
   rhoFinal { $rho; relTol 0; }
   ```
8. Do NOT generate `constant/g` for rhoPimpleFoam.
9. Fill `thermophysicalProperties` with actual Cp, mu, molWeight from the fluid config.
10. `Pr = mu * Cp / lambda` where `lambda` is thermal conductivity.
11. For cryogenic fluids (LN2): use `heRhoThermo` + `incompressiblePerfectGas` if density varies but Mach << 1, otherwise `hePsiThermo` + `perfectGas`.
12. `startFrom startTime; startTime 0;` — NEVER latestTime.
13. In `thermoType{}`: the key is `thermo  hConst;` (NOT `thermodynamics`). In `mixture{}`: the sub-dict is `thermodynamics { Cp ...; }` (NOT `thermo`).
14. **Fluid phase model selection**: Use `hePsiThermo` + `perfectGas` for gases. Use `heRhoThermo` + `rhoConst` for **stable liquids only** (water, oil — well below their boiling point). Mixing `hePsiThermo` + `rhoConst` causes runtime errors.
15. **🚫 FORBIDDEN: `rhoConst` for cryogenic fluids (LN₂, LH₂, LOX, T < 200 K)**: These fluids have large density variation with temperature and can undergo phase change. Using `rhoConst` produces physically wrong results and causes Negative Temperature crashes. Use `icoPolynomial` instead (T-dependent density polynomial).
16. **Liquid outlet pressure**: When `equationOfState rhoConst` or `icoPolynomial`, always set `0/p` outlet to the actual operating pressure (e.g., 101325 Pa). Never set it to 0 — this causes unphysical pressure fields that generate Negative Temperature divergence.
17. **`constant/fvOptions` with `limitTemperature`**: **MUST generate** for ALL cryogenic or low-temperature simulations (fluid T < 200 K, or cold fluid against hot wall ΔT > 50 K). Add `constant/fvOptions` to the list of generated files in your response. The system will also auto-inject it if missing, but generating it explicitly is preferred.