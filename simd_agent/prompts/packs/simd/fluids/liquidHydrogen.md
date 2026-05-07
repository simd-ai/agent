# Fluid Pack — Liquid Hydrogen (LH2)

**CAS**: 1333-74-0 | **Aliases**: LH2, LHy, para-H₂, liquid H₂
**⚠ FLAMMABILITY**: Highly flammable gas when vaporised. Relevant for safety notes only — CFD model treats as inert fluid.

---

## Phase configuration

| Key | Value |
|---|---|
| `phase1Name` (liquid) | `liquidHydrogen` |
| `phase2Name` (vapour) | `hydrogenVapour` |
| `alpha` field | `alpha.liquidHydrogen` |
| Molar mass | 2.016 g/mol |

---

## Saturation temperatures (phase-change reference)

| Pressure | T_sat |
|---|---|
| 1 atm (101 325 Pa) | **20.3 K** |
| 2 bar (200 000 Pa) | ~23.5 K |
| 5 bar | ~28.1 K |

If any wall or heat source is above T_sat, boiling (hydrogen flash) will occur.

---

## Liquid phase — `thermophysicalProperties.liquidHydrogen`

### For compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam

Use OpenFOAM's **native liquid-properties path**. Do NOT write hand-coded coefficient blocks.

⚠ **CRITICAL**: The native liquidProperties path requires **`energy sensibleInternalEnergy`**.
Using `sensibleEnthalpy` causes a SIGFPE crash inside `heRhoThermo` constructor (`powf64` in `libthermophysicalProperties.so`).

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.liquidHydrogen;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    properties      liquid;
    energy          sensibleInternalEnergy;
}

mixture
{
    H2;
}

// ************************************************************************* //
```

**`properties liquid;`** tells OpenFOAM to use its built-in liquid-properties framework.
**`mixture { H2; }`** selects liquid hydrogen from OpenFOAM's native liquid database.
Do NOT generate `CpCoeffs`, `rhoCoeffs`, `muCoeffs`, `kappaCoeffs`, or a custom `icoPolynomial` block.
**NEVER write `energy sensibleEnthalpy`** for native liquidProperties — it crashes.

### For rhoSimpleFoam / rhoPimpleFoam (single-phase only, fallback)

Use `icoPolynomial` EOS with hand-coded coefficients:

```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       polynomial;
    thermo          hPolynomial;
    equationOfState icoPolynomial;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight   2.016;
    }
    thermodynamics
    {
        CpCoeffs<8>   ( 9685.0 0 0 0 0 0 0 0 );   // J/kg·K — valid 18–30 K
        Hf            0;
        Sf            0;
    }
    transport
    {
        muCoeffs<8>    ( 1.3e-5 0 0 0 0 0 0 0 );   // Pa·s at 20 K
        kappaCoeffs<8> ( 0.098 0 0 0 0 0 0 0 );    // W/m·K at 20 K
    }
    equationOfState
    {
        rhoCoeffs<8>   ( 84.9 -0.7 0 0 0 0 0 0 );  // kg/m³: valid ~14–100 K
    }
}
```

#### icoPolynomial EOS ceiling (rho* solvers only)

`rhoCoeffs<8> (84.9 -0.7 ...)` → ρ(T) = 84.9 − 0.7·T → ρ = 0 at **T = 121.3 K**

- **EOS ceiling 121.3 K** — ρ → 0 above this → SIGFPE.
- For `rhoSimpleFoam`/`rhoPimpleFoam`: use `fvOptions max 109` (= 0.9 × 121.3 K).

---

## Vapour phase — `thermophysicalProperties.hydrogenVapour`

### For compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam

When paired with the native liquid (above), the vapour file **must also use `sensibleInternalEnergy`**.
Use `eConst` (constant-Cv internal energy model) and provide `Cv`, not `Cp`.

Cv = Cp / γ = 14310 / 1.4 = **10221 J/kg·K**

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.hydrogenVapour;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          eConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie
    {
        molWeight   2.016;
    }
    thermodynamics
    {
        Cv          10221;    // J/kg·K  (Cp/γ = 14310/1.4)
        Hf          0;
    }
    transport
    {
        mu          8.0e-6;   // Pa·s at ~300 K
        Pr          0.68;
    }
}

// ************************************************************************* //
```

### For rhoSimpleFoam / rhoPimpleFoam (single-phase vapour only, fallback)

```
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
    specie { molWeight 2.016; }
    thermodynamics { Cp 14310; Hf 0; }
    transport { mu 8.0e-6; Pr 0.68; }
}
```

---

## Base thermophysicalProperties

```
phases ( liquidHydrogen hydrogenVapour );
pMin   [1 -1 -2 0 0 0 0]  5000;
sigma  [1  0 -2 0 0 0 0]  0.0025;   // N/m — very low surface tension at 20 K
```

---

## fvOptions temperature limits (rhoSimpleFoam / rhoPimpleFoam ONLY)

⚠ **compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam**: NEVER generate `system/fvOptions` — crashes with `FOAM FATAL ERROR: Not implemented` (`twoPhaseMixtureThermo::he()`).

For `rhoSimpleFoam` / `rhoPimpleFoam` only:

| Parameter | Value |
|---|---|
| `min` | 50% × coldest BC temp (from CaseSpec.fv_options_t_min) |
| `max` | **109 K** (= 0.9 × 121.3 K EOS ceiling) |

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the trace:
- The failure is in `thermophysicalProperties.liquidHydrogen`, NOT in `thermophysicalProperties.hydrogenVapour`.
- Do NOT blame or regenerate the vapour file based on this trace.
- If the trace also shows `sensibleEnthalpy`: the liquid file has the wrong energy form — fix to `sensibleInternalEnergy`.

## Key warnings

1. LH2 boils at 20.3 K. Even 30 K walls will cause violent boiling.
2. LH2 has the highest specific heat of all liquids — thermal coupling is very strong. Use small deltaT (1e-5 s) for cryogenic startup.
3. compressibleInterFoam treats liquid+vapour as VOF phases — latent heat of vaporisation is NOT modelled.
4. For inter solvers, the native `H2` liquid properties model is valid up to the critical point (~33.2 K). Above this, H2 is supercritical.
5. `energy sensibleInternalEnergy` is MANDATORY for the native liquidProperties path. `sensibleEnthalpy` causes SIGFPE in `powf64` during thermo construction.
