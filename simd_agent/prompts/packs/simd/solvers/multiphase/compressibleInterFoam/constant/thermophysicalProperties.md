# compressibleInterFoam — constant/thermophysicalProperties (all thermo files)

## Three-file structure (MANDATORY)

compressibleInterFoam uses **three separate files**:

1. `constant/thermophysicalProperties` — base file (phases, pMin, sigma ONLY)
2. `constant/thermophysicalProperties.<phase1Name>` — liquid/primary phase thermo
3. `constant/thermophysicalProperties.<phase2Name>` — vapour/gas phase thermo

**DO NOT embed thermoType or mixture in the base thermophysicalProperties file.**

---

## 1. Base file: `constant/thermophysicalProperties`

Contains only phase list, pressure floor, and surface tension. Nothing else.

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

phases ( <phase1Name> <phase2Name> );

pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  <sigma>;   // N/m — surface tension between phases

// ************************************************************************* //
```

Surface tension guidance:
- LN2/vapour: `sigma 0.009`
- LH2/vapour: `sigma 0.0025`
- LOX/vapour: `sigma 0.013`
- LHe/vapour: `sigma 0.0003`
- Water/air: `sigma 0.07`
- Generic liquid/gas: `sigma 0.03`

---

## 2. Per-phase file: liquid phase (PREFERRED: native liquidProperties)

### Option A — Native liquidProperties (preferred for known fluids)

When the fluid is a known cryogenic or common liquid with a native OpenFOAM model, use `properties liquid;`. This is **simpler, more accurate, and avoids hand-coded coefficient errors**.

Mapping of fluid → native class name:
- Liquid nitrogen (LN2) → `N2`
- Liquid oxygen (LOX) → `O2`
- Liquid hydrogen (LH2) → `H2`
- Liquid helium (LHe) → `He`
- Water → `H2O`

⚠ **CRITICAL**: The native liquidProperties path **MUST** use `energy sensibleInternalEnergy`.
`sensibleEnthalpy` causes SIGFPE in `heRhoThermo` constructor (`powf64` in `libthermophysicalProperties.so`) — confirmed crash on OF 2406.

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.<phase1Name>;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    properties      liquid;
    energy          sensibleInternalEnergy;   // MUST be sensibleInternalEnergy — NOT sensibleEnthalpy
}

mixture
{
    <nativeLiquidClass>;    // e.g. N2, O2, H2, He, H2O
}

// ************************************************************************* //
```

**Do NOT generate** `CpCoeffs`, `rhoCoeffs`, `muCoeffs`, `kappaCoeffs`, `specie`, `thermodynamics`, `transport`, or `equationOfState` blocks when using the native path.

### Option B — icoPolynomial (fallback for unknown fluids)

When the fluid does not have a native OpenFOAM liquid class, use `icoPolynomial`:

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.<phase1Name>;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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
        molWeight   <molWeight>;    // nMoles deprecated — omit it
    }
    thermodynamics
    {
        CpCoeffs<8>   ( <Cp> 0 0 0 0 0 0 0 );
        Hf            0;
        Sf            0;
    }
    transport
    {
        muCoeffs<8>    ( <mu> 0 0 0 0 0 0 0 );
        kappaCoeffs<8> ( <kappa> 0 0 0 0 0 0 0 );
    }
    equationOfState
    {
        rhoCoeffs<8>   ( <a0> <a1> 0 0 0 0 0 0 );
    }
}

// ************************************************************************* //
```

### Keywords: `thermo` vs `thermodynamics` (Option B only)
- Inside `thermoType {}`: use keyword `thermo hPolynomial;`
- Inside `mixture {}`: the sub-dict is `thermodynamics { CpCoeffs<8> ...; }` — NOT `thermo`

---

## 3. Per-phase file: `constant/thermophysicalProperties.<phase2Name>` (vapour/gas)

Vapour phase uses `perfectGas`. When the liquid phase uses the native liquidProperties path
(Option A), the vapour **must also use `sensibleInternalEnergy`** for consistency —
`twoPhaseMixtureThermo` requires both phases to use the same energy form.

Use `eConst` (constant Cv) + `sensibleInternalEnergy`:

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.<phase2Name>;
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
        molWeight   <molWeight>;    // nMoles deprecated — omit it
    }
    thermodynamics
    {
        Cv          <Cv_gas>;       // J/kg·K — Cv = Cp/γ  (diatomic γ=1.4, monatomic γ=5/3)
        Hf          0;
    }
    transport
    {
        mu          <mu_gas>;
        Pr          <Pr_gas>;
    }
}

// ************************************************************************* //
```

**Cv reference values** (use these, not Cp):
- N2 vapour:  Cv = 743   J/kg·K  (Cp=1040, γ=1.4)
- O2 vapour:  Cv = 657   J/kg·K  (Cp=920,  γ=1.4)
- H2 vapour:  Cv = 10221 J/kg·K  (Cp=14310,γ=1.4)
- He vapour:  Cv = 3116  J/kg·K  (Cp=5193, γ=5/3)

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the stack trace:
- The failure is in `thermophysicalProperties.<liquidPhase>` (the native liquid file), NOT in the vapour file.
- Do NOT blame or regenerate the vapour file based on this trace — `perfectGas` never appears as `liquidProperties`.
- Inspect the liquid phase file first when this trace appears.

`nMoles` is deprecated — do NOT include it in any `specie {}` block (liquid or vapour).
