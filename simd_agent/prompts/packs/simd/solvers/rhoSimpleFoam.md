# Solver: rhoSimpleFoam  ·  OpenFOAM v2406

**Type**: Steady-state · Compressible · SIMPLE (pressure-based)  
**Pressure field**: `p` (absolute static pressure, **Pa**, dimensions `[1 -1 -2 0 0 0 0]`)  
**Energy equation**: ✅ YES — the solver transports `he = thermo.he()` internally (`h` or `e`).  
  ⚠️ **Do NOT generate `0/h` or `0/e`.** The thermo package initialises the energy field from `0/T` at startup — providing `0/h` causes `Negative initial temperature T0` crashes.  
  **Always provide `0/T` with temperature boundary conditions (in Kelvin).**  
**Gravity**: ❌ No `constant/g`

## Key source facts (must follow)

- rhoSimpleFoam reads/uses `p = thermo.p()` and solves an energy equation for `he = thermo.he()` (not T).
- Energy equation contains `fvm::div(phi, he)` and includes extra explicit convection terms:
  - if `he` is `e`: uses a temporary field named `Ekp` and a scheme-name `div(phiv,p)`
  - else uses a temporary field named `K`
- Keep fvSchemes consistent with these names.

## Required files

### system/
| File | Notes |
|------|-------|
| `system/controlDict` | `application rhoSimpleFoam;` · `deltaT 1;` · `endTime` = max_iterations |
| `system/fvSchemes` | Upwind schemes; wallDist if turbulent |
| `system/fvSolution` | `SIMPLE { }` block; solvers for p, U, h/e, k, omega/epsilon |

### 0/ (initial fields)
| File | Notes |
|------|-------|
| `0/U` | Velocity — all patches |
| `0/p` | Absolute pressure `[1 -1 -2 0 0 0 0]` — use actual Pa values (e.g. 101325) |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` in **Kelvin** — thermo reads T and initialises h/e from it |
| `0/k`, `0/omega`, `0/nut` | If kOmegaSST (+ `0/alphat` if turbulent heat transfer wall functions used) |
| `0/k`, `0/epsilon`, `0/nut` | If kEpsilon (+ `0/alphat` similarly) |
| `constant/fvOptions` | **Conditionally required** — MUST include for cryogenic/liquid cases; use `limitTemperature` to suppress `Negative Temperature` divergence during startup (see template below) |

### constant/
| File | Notes |
|------|-------|
| `constant/thermophysicalProperties` | REQUIRED — see template below |
| `constant/turbulenceProperties` | REQUIRED even for laminar (simulationType = laminar/RAS/LES) |

## constant/thermophysicalProperties template

```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;   // OR sensibleInternalEnergy
}

mixture
{
    specie
    {
        nMoles          1;
        molWeight       28.97;   // Air: 28.97  LN2: 28.014
    }
    thermodynamics
    {
        Cp              1005;    // J/kg/K
        Hf              0;
    }
    transport
    {
        mu              1.8e-5;  // Dynamic viscosity [kg/m/s]
        Pr              0.713;   // Prandtl number
    }
}
```
> Adjust `molWeight`, `Cp`, `mu`, `Pr` from the fluid config.

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
transport       const;     // or sutherland
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
        tolerance       1e-06;
        relTol          0.1;
    }
    pFinal { $p; relTol 0; }

    // ✅ REQUIRED — rhoSimpleFoam solves rhoEqn; OpenFOAM looks up this entry at runtime.
    // Missing it causes: "Entry 'rho' not found in dictionary system/fvSolution/solvers"
    rho
    {
        solver      diagonal;
        tolerance   1e-12;
        relTol      0;
    }
    rhoFinal { $rho; relTol 0; }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    UFinal { $U; relTol 0; }

    // Energy variable — choose ONE name based on thermoType.energy:
    //   sensibleEnthalpy        → use h / hFinal
    //   sensibleInternalEnergy  → use e / eFinal
    h
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    hFinal { $h; relTol 0; }

    // If using internal energy instead:
    // e { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-06; relTol 0.1; }
    // eFinal { $e; relTol 0; }

    k       { $U; }
    kFinal  { $k; relTol 0; }
    omega   { $U; }
    omegaFinal { $omega; relTol 0; }
    epsilon { $U; }
    epsilonFinal { $epsilon; relTol 0; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p       1e-4;
        U       1e-4;
        h       1e-6;   // or e if using sensibleInternalEnergy
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

## fvSchemes template

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default             none;

    div(phi,U)          Gauss linearUpwind grad(U);

    // Energy convection — choose based on energy variable
    div(phi,h)          Gauss linearUpwind grad(h);
    div(phi,e)          Gauss linearUpwind grad(e);

    // Extra explicit terms created in EEqn (safe to provide if default is none)
    div(phi,K)          Gauss linearUpwind grad(K);
    div(phi,Ekp)        Gauss linearUpwind grad(Ekp);
    div(phiv,p)         Gauss linearUpwind grad(p);

    // Pressure equation — used when simple.transonic() is true
    div(phid,p)         Gauss limitedLinear 1;

    // Turbulence
    div(phi,k)          Gauss linearUpwind grad(k);
    div(phi,omega)      Gauss linearUpwind grad(omega);
    div(phi,epsilon)    Gauss linearUpwind grad(epsilon);

    // Viscous stress term
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }

// Only include wallDist if running turbulence wall functions
wallDist        { method meshWave; }
```

## Critical rules

1. Pressure `0/p` is **absolute Pa**. Convert bar → Pa (1 bar = 100,000 Pa).
2. **NEVER generate `0/h` or `0/e`**. The thermo package initialises the energy field from `0/T` at startup. Generating `0/h` causes `Negative initial temperature T0` crashes because OpenFOAM back-converts h→T using a thermo reference that doesn't match the raw value provided. Standard tutorials omit `0/h`.
3. `0/T` contains temperatures in **Kelvin**. Use the values from the boundary conditions config directly (e.g., `fixedValue uniform 300;`).
4. In `fvSolution` and `fvSchemes`, the energy variable is `h` or `e` (from thermoType.energy) — never `T`.
5. Always include `constant/turbulenceProperties` (simulationType = laminar/RAS/LES).
6. If `divSchemes { default none; }`, you MUST provide schemes for every name used in EEqn: `div(phi,h)` or `div(phi,e)`, `div(phi,K)` or `div(phi,Ekp)`, and `div(phiv,p)` when `e` is used.
7. `endTime` for steady rhoSimpleFoam acts as an iteration counter; use `<max_iterations>` with `deltaT 1`.
8. `startFrom startTime; startTime 0;` — **NEVER** `startFrom latestTime`.
9. Wall U: `noSlip`; wall p: `zeroGradient`.
9. **`rho` solver entry in `fvSolution/solvers`**: Unlike rhoPimpleFoam, rhoSimpleFoam does **not** call `rhoEqn.solve()` — it assigns `rho = thermo.rho()` and calls `rho.relax()` inside the pressure loop. A `rho {}` solver block is therefore **not strictly required**, but including it is **safe and recommended** for robustness (it is harmless if unused). Keep `relaxationFactors.fields { rho 0.05; }` — that **is** used. Do NOT generate `0/rho` (field file).
10. In `thermoType{}`: key is `thermo  hConst;` (NOT `thermodynamics`). In `mixture{}`: sub-dict is `thermodynamics { Cp ...; }` (NOT `thermo`).
11. **Fluid phase model selection**: Use `hePsiThermo` + `perfectGas` for gases. Use `heRhoThermo` + `rhoConst` for **stable liquids only** (water, oil — well below their boiling point). Mixing `hePsiThermo` + `rhoConst` causes runtime errors.
12. **🚫 FORBIDDEN: `rhoConst` for cryogenic fluids (LN₂, LH₂, LOX, T < 200 K)**: These fluids have large density variation with temperature and can undergo phase change. Using `rhoConst` produces physically wrong results and causes Negative Temperature crashes. Use `icoPolynomial` instead (T-dependent density polynomial).
13. **Liquid outlet pressure**: When `equationOfState rhoConst` or `icoPolynomial`, always set `0/p` outlet to the actual operating pressure (e.g., 101325 Pa). Never set it to 0 — this causes unphysical pressure fields that generate Negative Temperature divergence.
14. **`constant/fvOptions` with `limitTemperature`**: **MUST generate** for ALL cryogenic or low-temperature simulations (fluid T < 200 K, or cold fluid against hot wall ΔT > 50 K). Add `constant/fvOptions` to the list of generated files in your response. The system will also auto-inject it if missing, but generating it explicitly is preferred.