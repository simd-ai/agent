# Solver: compressibleInterFoam — Identity & Global Rules

**Algorithm**: PIMPLE + MULES (transient, two-phase VOF, compressible, non-isothermal)
**Pressure fields**: `p_rgh` (modified pressure) AND `p` (absolute pressure) — BOTH required
**Energy**: YES — generate `0/T` (Kelvin)
**Gravity**: REQUIRED — always generate `constant/g`
**Turbulence**: uses `nut` and `alphat` (same as other modern OF solvers)

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `g`, `thermophysicalProperties` (base), `thermophysicalProperties.<phase1Name>`, `thermophysicalProperties.<phase2Name>`, `turbulenceProperties` |
| `0/` | `U`, `p_rgh`, `p`, `T`, `alpha.<phase1Name>`, `k`/`omega`/`epsilon` (if turbulent), `nut`, `alphat` |

**CRITICAL**: Generate BOTH `0/p` and `0/p_rgh`:
- `0/p_rgh` — modified pressure = p − ρ·g·h (dimensions `[1 -1 -2 0 0 0 0]`)
- `0/p` — absolute pressure (same dimensions `[1 -1 -2 0 0 0 0]`, same BCs as p_rgh)
  OpenFOAM reads `0/p` at startup with `MUST_READ`. Missing it causes "cannot find file 0/p" fatal error.

**Never generate**: `0/h`, `0/e`, `0/mut`, `constant/transportProperties`, `system/fvOptions`

---

## Phase naming (CRITICAL)

Use the phase names from CaseSpec (`alpha_fields`). Examples:
- LN2 boiloff: `liquidNitrogen` and `nitrogenVapour`
- Water/air: `water` and `air`
- Generic: `liquid` and `gas`

Phase name drives:
- `alpha.<phase1Name>` field file name
- `constant/thermophysicalProperties.<phaseName>` per-phase thermo files
- `phases ( <phase1> <phase2> );` in the base thermophysicalProperties

---

## ThermophysicalProperties file structure (CRITICAL)

compressibleInterFoam uses a **three-file** thermo approach:

### 1. `constant/thermophysicalProperties` (base — MINIMAL)
```
FoamFile { ... object thermophysicalProperties; }

phases ( <phase1Name> <phase2Name> );
pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  <sigma>;
```
**DO NOT put thermoType or mixture here** — only phases, pMin, sigma.

### 2. `constant/thermophysicalProperties.<phase1Name>` (liquid phase)
```
FoamFile { ... object thermophysicalProperties.<phase1Name>; }

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       polynomial;   // or const
    thermo          hPolynomial;  // or hConst
    equationOfState icoPolynomial; // for cryogenic liquids
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie        { nMoles 1; molWeight <MW>; }
    thermodynamics { CpCoeffs<8> ( <Cp> 0 0 0 0 0 0 0 ); Hf 0; Sf 0; }
    transport     { muCoeffs<8> ( <mu> 0 0 0 0 0 0 0 ); kappaCoeffs<8> ( <kappa> 0 0 0 0 0 0 0 ); }
    equationOfState { rhoCoeffs<8> ( <a0> <a1> 0 0 0 0 0 0 ); }
}
```

### 3. `constant/thermophysicalProperties.<phase2Name>` (vapour/gas phase)
```
FoamFile { ... object thermophysicalProperties.<phase2Name>; }

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie        { nMoles 1; molWeight <MW>; }
    thermodynamics { Cp <Cp_gas>; Hf 0; }
    transport     { mu <mu_gas>; Pr <Pr_gas>; }
}
```

---

## Global critical rules

1. Generate BOTH `0/p_rgh` AND `0/p` — compressibleInterFoam reads both at startup.
2. Always generate `0/T` (Kelvin) and `constant/g`.
3. `pcorr` solver block is REQUIRED in `fvSolution`.
4. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
4b. **PIMPLE `nOuterCorrectors` MUST be ≥ 2** — icoPolynomial liquid has zero acoustic compressibility (ρ = f(T) only). With `nOuterCorrectors 1` (PISO mode), the T→ρ→p→U coupling is unresolved per timestep: Co spikes from 0.4 → 18 → 404 → 4.6e6 within 5 steps and T crashes to -552 K. Use `nNonOrthogonalCorrectors 1` and `momentumPredictor yes`.
4c. **`maxCo 0.5` in controlDict** — NOT 1.0. At Co=1 the timestep limiter reacts too slowly to prevent the divergence cascade.
5. `divSchemes default` MUST be `Gauss linear` — NEVER `none`, `Gauss upwind`, or `bounded Gauss upwind`. In OF 2406, `none` is treated as a scheme name and causes "attempt to read beyond EOF" (base constructor tries to read interpolation scheme from empty stream). `Gauss upwind` causes same EOF. `Gauss linear` is safe for all field types. Also list BOTH `div(rhoPhi,he)` AND `div(rhoPhi,h)` — OF uses `he` as the internal field name in some code paths.
5. Use `nut` (kinematic) not `mut` — modern OF 2406 ESI uses `nut` for compressible multiphase.
6. Base `thermophysicalProperties` contains ONLY `phases`, `pMin`, `sigma`. All thermo goes in per-phase files.
7. `startFrom startTime; startTime 0;` — never `latestTime`.
8. Every mesh patch must appear in ALL `0/*` files; `empty` patches → `type empty`.
9. **NEVER generate `system/fvOptions`** — `limitTemperature` calls `he()` on `twoPhaseMixtureThermo` which does not implement it → `FOAM FATAL ERROR: Not implemented` at startup.

---

## Initial fill semantics

When the user specifies that no liquid is present initially and the liquid enters through the inlet at the start of the simulation:

- initialise the liquid phase fraction with zero throughout the domain (`internalField uniform 0`)
- impose liquid entry at the inlet through the liquid alpha field (`fixedValue uniform 1`)
- initialise velocity from rest unless the user specifies another initial condition (`internalField uniform (0 0 0)`)
- initialise temperature with an explicit positive field (use inlet temperature when no separate domain temperature is given)
- initialise pressure with an explicit domain value consistent with the case setup

When heating and phase change are expected, the initial fields should describe the starting state, while the thermal boundary conditions drive the later evolution.

When the domain is pre-filled with liquid (e.g. a tank at rest), initialise `alpha.<liquidPhase>` with `uniform 1` and choose an appropriate initial velocity and temperature.
