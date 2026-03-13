# Fluid Pack — Liquid Helium (LHe)

**CAS**: 7440-59-7 | **Aliases**: LHe, LHe-4, liquid He-4
**Note**: This pack covers He-4 (normal liquid) above 2.17 K. Superfluid He-II (below 2.17 K, λ-point) requires a two-fluid model not supported by standard OpenFOAM solvers.

---

## Phase configuration

| Key | Value |
|---|---|
| `phase1Name` (liquid) | `liquidHelium` |
| `phase2Name` (vapour) | `heliumVapour` |
| `alpha` field | `alpha.liquidHelium` |
| Molar mass | 4.003 g/mol |

---

## Saturation temperatures (phase-change reference)

| Pressure | T_sat |
|---|---|
| 1 atm (101 325 Pa) | **4.22 K** |
| 2 bar (200 000 Pa) | ~5.0 K |
| 5 bar | ~6.2 K |

Any wall above T_sat will cause immediate boiling. Even 4.5 K walls cause boiling at 1 atm.

---

## Liquid phase — `thermophysicalProperties.liquidHelium`

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
    object      thermophysicalProperties.liquidHelium;
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
    He;
}

// ************************************************************************* //
```

**`properties liquid;`** tells OpenFOAM to use its built-in liquid-properties framework.
**`mixture { He; }`** selects liquid helium from OpenFOAM's native liquid database.
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
        molWeight   4.003;
    }
    thermodynamics
    {
        CpCoeffs<8>   ( 5240.0 0 0 0 0 0 0 0 );   // J/kg·K at 4.2 K
        Hf            0;
        Sf            0;
    }
    transport
    {
        muCoeffs<8>    ( 3.6e-6 0 0 0 0 0 0 0 );   // Pa·s at 4.2 K
        kappaCoeffs<8> ( 0.020 0 0 0 0 0 0 0 );    // W/m·K at 4.2 K
    }
    equationOfState
    {
        rhoCoeffs<8>   ( 146.0 -5.0 0 0 0 0 0 0 ); // kg/m³: valid 2.5–25 K
    }
}
```

#### icoPolynomial EOS ceiling (rho* solvers only)

`rhoCoeffs<8> (146.0 -5.0 ...)` → ρ(T) = 146.0 − 5.0·T → ρ = 0 at **T = 29.2 K**

- **EOS ceiling 29.2 K** — ρ → 0 above this → SIGFPE.
- For `rhoSimpleFoam`/`rhoPimpleFoam`: use `fvOptions max 26` (= 0.9 × 29.2 K).
- ⚠ **VERY TIGHT EOS RANGE**: The EOS ceiling is only 29.2 K — just ~25 K above boiling. Any wall temperature above 26 K will push cells above the EOS limit.

---

## Vapour phase — `thermophysicalProperties.heliumVapour`

### For compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam

When paired with the native liquid (above), the vapour file **must also use `sensibleInternalEnergy`**.
Use `eConst` (constant-Cv internal energy model) and provide `Cv`, not `Cp`.

Cv = Cp × (3/5) = 5193 × 0.6 = **3116 J/kg·K**  (monatomic ideal gas: γ = 5/3)

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.heliumVapour;
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
        molWeight   4.003;
    }
    thermodynamics
    {
        Cv          3116;     // J/kg·K  (Cp × 3/5 = 5193 × 0.6; monatomic γ = 5/3)
        Hf          0;
    }
    transport
    {
        mu          2.0e-6;   // Pa·s at ~4 K (very low)
        Pr          0.67;
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
    specie { molWeight 4.003; }
    thermodynamics { Cp 5193; Hf 0; }
    transport { mu 2.0e-6; Pr 0.67; }
}
```

---

## Base thermophysicalProperties

```
phases ( liquidHelium heliumVapour );
pMin   [1 -1 -2 0 0 0 0]  1000;
sigma  [1  0 -2 0 0 0 0]  0.0003;   // N/m — extremely low surface tension at 4.2 K
```

---

## fvOptions temperature limits (rhoSimpleFoam / rhoPimpleFoam ONLY)

⚠ **compressibleInterFoam / compressibleInterIsoFoam / compressibleMultiphaseInterFoam**: NEVER generate `system/fvOptions` — crashes with `FOAM FATAL ERROR: Not implemented` (`twoPhaseMixtureThermo::he()`).

For `rhoSimpleFoam` / `rhoPimpleFoam` only:

| Parameter | Value |
|---|---|
| `min` | 50% × coldest BC temp (from CaseSpec.fv_options_t_min) |
| `max` | **26 K** (= 0.9 × 29.2 K EOS ceiling) |

---

## controlDict — cryogenic startup

LHe is extremely sensitive to deltaT. Use:
```
deltaT          1e-6;    // 1 microsecond — helium is very light (rho ~125 kg/m³)
maxCo           0.1;     // very conservative Courant limit for startup
maxAlphaCo      0.1;
```

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the trace:
- The failure is in `thermophysicalProperties.liquidHelium`, NOT in `thermophysicalProperties.heliumVapour`.
- Do NOT blame or regenerate the vapour file based on this trace.
- If the trace also shows `sensibleEnthalpy`: the liquid file has the wrong energy form — fix to `sensibleInternalEnergy`.

## Key warnings

1. LHe boils at 4.22 K. Any significant heat input causes immediate boiling.
2. Liquid helium has very low density (~125 kg/m³) — flow patterns differ significantly from heavier cryogens.
3. Surface tension (0.0003 N/m) is nearly zero — interface dynamics are different from water/nitrogen.
4. This model does NOT capture superfluid He-II (below 2.17 K) effects.
5. compressibleInterFoam treats liquid+vapour as VOF phases — latent heat (20.7 kJ/kg) is NOT modelled.
6. For inter solvers, the native `He` liquid properties model is valid up to the critical point (~5.19 K). Above this, He is supercritical.
7. `energy sensibleInternalEnergy` is MANDATORY for the native liquidProperties path. `sensibleEnthalpy` causes SIGFPE in `powf64` during thermo construction.
