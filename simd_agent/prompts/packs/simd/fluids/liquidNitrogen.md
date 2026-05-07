# Fluid Pack — Liquid Nitrogen (LN2)

**CAS**: 7727-37-9 | **Aliases**: LN2, LIN, liquid N₂

---

## Phase configuration

| Key | Value |
|---|---|
| `phase1Name` (liquid) | `liquidNitrogen` |
| `phase2Name` (vapour) | `nitrogenVapour` |
| `alpha` field | `alpha.liquidNitrogen` |
| Molar mass | 28.014 g/mol |

---

## Saturation temperatures (phase-change reference)

| Pressure | T_sat |
|---|---|
| 1 atm (101 325 Pa) | **77.4 K** |
| 4 bar (400 000 Pa) | ~91 K |
| 10 bar | ~103 K |

If any wall or heat source is above T_sat, boiling will occur.

---

## Liquid phase — `thermophysicalProperties.liquidNitrogen`

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
    object      thermophysicalProperties.liquidNitrogen;
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
    N2;
}

// ************************************************************************* //
```

**`properties liquid;`** tells OpenFOAM to use its built-in liquid-properties framework.
**`mixture { N2; }`** selects liquid nitrogen from OpenFOAM's native liquid database.
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
        molWeight   28.014;
    }
    thermodynamics
    {
        CpCoeffs<8>   ( 2042.0 0 0 0 0 0 0 0 );   // J/kg·K — valid 65–120 K
        Hf            0;
        Sf            0;
    }
    transport
    {
        muCoeffs<8>    ( 1.58e-4 0 0 0 0 0 0 0 );  // Pa·s at 77 K
        kappaCoeffs<8> ( 0.1396 0 0 0 0 0 0 0 );   // W/m·K at 77 K
    }
    equationOfState
    {
        rhoCoeffs<8>   ( 1169.9 -4.7 0 0 0 0 0 0 ); // kg/m³: valid 65–220 K
    }
}
```

#### icoPolynomial EOS ceiling (rho* solvers only)

`rhoCoeffs<8> (1169.9 -4.7 ...)` → ρ(T) = 1169.9 − 4.7·T → ρ = 0 at **T = 248.9 K**

- **EOS ceiling 248.9 K** — ρ → 0 above this → SIGFPE.
- For `rhoSimpleFoam`/`rhoPimpleFoam`: use `fvOptions max 224` (= 0.9 × 248.9 K).

---

## Vapour phase — `thermophysicalProperties.nitrogenVapour`

### For compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam

When paired with the native liquid (above), the vapour file **must also use `sensibleInternalEnergy`**.
Use `eConst` (constant-Cv internal energy model) and provide `Cv`, not `Cp`.

Cv = Cp / γ = 1040 / 1.4 = **743 J/kg·K**

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.nitrogenVapour;
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
        molWeight   28.0134;
    }
    thermodynamics
    {
        Cv          743;      // J/kg·K  (Cp/γ = 1040/1.4)
        Hf          0;
    }
    transport
    {
        mu          5.4e-06;  // Pa·s at ~77 K (cryogenic; much lower than room-temp value)
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
    specie { molWeight 28.0134; }
    thermodynamics { Cp 1040; Hf 0; }
    transport { mu 1.67e-5; Pr 0.71; }
}
```

---

## Base thermophysicalProperties

```
phases ( liquidNitrogen nitrogenVapour );
pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  0.009;    // N/m surface tension at 77 K
```

---

## fvOptions temperature limits (rhoSimpleFoam / rhoPimpleFoam ONLY)

⚠ **compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam**: NEVER generate `system/fvOptions` — crashes with `FOAM FATAL ERROR: Not implemented` (`twoPhaseMixtureThermo::he()`).

For `rhoSimpleFoam` / `rhoPimpleFoam` only:

| Parameter | Value |
|---|---|
| `min` | 50% × coldest BC temp (from CaseSpec.fv_options_t_min) |
| `max` | **224 K** (= 0.9 × 248.9 K EOS ceiling) |

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the trace:
- The failure is in `thermophysicalProperties.liquidNitrogen`, NOT in `thermophysicalProperties.nitrogenVapour`.
- Do NOT blame or regenerate the vapour file based on this trace.
- If the trace also shows `sensibleEnthalpy`: the liquid file has the wrong energy form — fix to `sensibleInternalEnergy`.

## Key warnings

1. LN2 boils at 77.4 K. Any wall above 87 K will cause local boiling.
2. compressibleInterFoam models two coexisting phases (liquid + vapour VOF) — it does NOT model nucleate boiling physics. Results are an approximation.
3. For inter solvers, the native `N2` liquid properties model is valid up to the critical point (~126 K). Above this, N2 is supercritical — the liquid model extrapolates beyond its design range.
4. `energy sensibleInternalEnergy` is MANDATORY for the native liquidProperties path. `sensibleEnthalpy` causes SIGFPE in `powf64` during thermo construction.
