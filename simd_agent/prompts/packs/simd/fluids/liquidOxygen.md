# Fluid Pack — Liquid Oxygen (LOX)

**CAS**: 7782-44-7 | **Aliases**: LOX, LO2, liquid O₂
**⚠ OXIDISER**: Strong oxidiser — relevant for safety context only, not modelled in CFD.

---

## Phase configuration

| Key | Value |
|---|---|
| `phase1Name` (liquid) | `liquidOxygen` |
| `phase2Name` (vapour) | `oxygenVapour` |
| `alpha` field | `alpha.liquidOxygen` |
| Molar mass | 31.999 g/mol |

---

## Saturation temperatures (phase-change reference)

| Pressure | T_sat |
|---|---|
| 1 atm (101 325 Pa) | **90.2 K** |
| 4 bar (400 000 Pa) | ~107 K |
| 10 bar | ~119 K |

If any wall or heat source is above T_sat, boiling will occur.

---

## Liquid phase — `thermophysicalProperties.liquidOxygen`

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
    object      thermophysicalProperties.liquidOxygen;
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
    O2;
}

// ************************************************************************* //
```

**`properties liquid;`** tells OpenFOAM to use its built-in liquid-properties framework.
**`mixture { O2; }`** selects liquid oxygen from OpenFOAM's native liquid database.
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
        molWeight   31.999;
    }
    thermodynamics
    {
        CpCoeffs<8>   ( 1700.0 0 0 0 0 0 0 0 );   // J/kg·K — valid 70–140 K
        Hf            0;
        Sf            0;
    }
    transport
    {
        muCoeffs<8>    ( 1.94e-4 0 0 0 0 0 0 0 );  // Pa·s at 90 K
        kappaCoeffs<8> ( 0.152 0 0 0 0 0 0 0 );    // W/m·K at 90 K
    }
    equationOfState
    {
        rhoCoeffs<8>   ( 1600.0 -5.1 0 0 0 0 0 0 ); // kg/m³: valid 70–280 K
    }
}
```

#### icoPolynomial EOS ceiling (rho* solvers only)

`rhoCoeffs<8> (1600.0 -5.1 ...)` → ρ(T) = 1600.0 − 5.1·T → ρ = 0 at **T = 313.7 K**

- **EOS ceiling 313.7 K** — ρ → 0 above this → SIGFPE.
- For `rhoSimpleFoam`/`rhoPimpleFoam`: use `fvOptions max 282` (= 0.9 × 313.7 K).

---

## Vapour phase — `thermophysicalProperties.oxygenVapour`

### For compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam

When paired with the native liquid (above), the vapour file **must also use `sensibleInternalEnergy`**.
Use `eConst` (constant-Cv internal energy model) and provide `Cv`, not `Cp`.

Cv = Cp / γ = 920 / 1.4 = **657 J/kg·K**

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.oxygenVapour;
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
        molWeight   31.999;
    }
    thermodynamics
    {
        Cv          657;      // J/kg·K  (Cp/γ = 920/1.4)
        Hf          0;
    }
    transport
    {
        mu          2.0e-5;   // Pa·s at ~300 K
        Pr          0.71;
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
    specie { molWeight 31.999; }
    thermodynamics { Cp 920; Hf 0; }
    transport { mu 2.0e-5; Pr 0.71; }
}
```

---

## Base thermophysicalProperties

```
phases ( liquidOxygen oxygenVapour );
pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  0.013;    // N/m surface tension at 90 K
```

---

## fvOptions temperature limits (rhoSimpleFoam / rhoPimpleFoam ONLY)

⚠ **compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam**: NEVER generate `system/fvOptions` — crashes with `FOAM FATAL ERROR: Not implemented` (`twoPhaseMixtureThermo::he()`).

For `rhoSimpleFoam` / `rhoPimpleFoam` only:

| Parameter | Value |
|---|---|
| `min` | 50% × coldest BC temp (from CaseSpec.fv_options_t_min) |
| `max` | **282 K** (= 0.9 × 313.7 K EOS ceiling) |

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the trace:
- The failure is in `thermophysicalProperties.liquidOxygen`, NOT in `thermophysicalProperties.oxygenVapour`.
- Do NOT blame or regenerate the vapour file based on this trace.
- If the trace also shows `sensibleEnthalpy`: the liquid file has the wrong energy form — fix to `sensibleInternalEnergy`.

## Key warnings

1. LOX boils at 90.2 K. Walls above ~100 K will cause film boiling.
2. LOX has the highest density of the common cryogenic propellants — momentum is significant.
3. compressibleInterFoam models liquid+vapour as VOF phases — latent heat of vaporisation is NOT modelled.
4. For inter solvers, the native `O2` liquid properties model is valid up to the critical point (~154.6 K). Above this, O2 is supercritical.
5. `energy sensibleInternalEnergy` is MANDATORY for the native liquidProperties path. `sensibleEnthalpy` causes SIGFPE in `powf64` during thermo construction.
