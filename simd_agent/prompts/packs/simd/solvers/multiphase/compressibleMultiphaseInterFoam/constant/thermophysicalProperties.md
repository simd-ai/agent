# compressibleMultiphaseInterFoam — constant/thermophysicalProperties (all thermo files)

## Multi-file structure (MANDATORY for N≥3 phases)

compressibleMultiphaseInterFoam uses **N+1 files**:

1. `constant/thermophysicalProperties` — base file (phases list, pMin, sigma pairs ONLY)
2. `constant/thermophysicalProperties.<phase1Name>` — per-phase thermo (one per phase)
3. `constant/thermophysicalProperties.<phase2Name>` — per-phase thermo
4. ... one file per phase

**DO NOT embed thermoType or mixture in the base thermophysicalProperties file.**

---

## 1. Base file: `constant/thermophysicalProperties`

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

phases ( <phase1Name> <phase2Name> <phase3Name> ... );

pMin    [1 -1 -2 0 0 0 0]  10000;

// Surface tension — one entry per phase PAIR (sigma<i><j> where i<j)
sigma12  [1  0 -2 0 0 0 0]  <sigma_12>;   // between phase1 and phase2
sigma13  [1  0 -2 0 0 0 0]  <sigma_13>;   // between phase1 and phase3

// ************************************************************************* //
```

---

## 2. Per-phase files (one per phase)

### Liquid phases — Native liquidProperties (preferred for known fluids)

When the phase is a known liquid with a native OpenFOAM model, use `properties liquid;`.

Mapping of fluid → native class name:
- Liquid nitrogen (LN2) → `N2`
- Liquid oxygen (LOX) → `O2`
- Liquid hydrogen (LH2) → `H2`
- Liquid helium (LHe) → `He`
- Water → `H2O`

⚠ **CRITICAL**: The native liquidProperties path **MUST** use `energy sensibleInternalEnergy`.
`sensibleEnthalpy` causes SIGFPE in `heRhoThermo` constructor — confirmed crash on OF 2406.

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties.<liquidPhaseName>;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    properties      liquid;
    energy          sensibleInternalEnergy;   // MUST — NOT sensibleEnthalpy
}

mixture
{
    <nativeLiquidClass>;    // e.g. N2, O2, H2, He, H2O
}

// ************************************************************************* //
```

**Do NOT generate** coefficient blocks (`CpCoeffs`, `rhoCoeffs`, etc.) when using native path.

### Liquid phases — icoPolynomial (fallback for unknown fluids)

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
    specie        { molWeight <MW>; }   // nMoles deprecated — omit it
    thermodynamics { CpCoeffs<8> ( <Cp> 0 0 0 0 0 0 0 ); Hf 0; Sf 0; }
    transport     { muCoeffs<8> ( <mu> 0 0 0 0 0 0 0 ); kappaCoeffs<8> ( <kappa> 0 0 0 0 0 0 0 ); }
    equationOfState { rhoCoeffs<8> ( <a0> <a1> 0 0 0 0 0 0 ); }
}
```

### Gas/vapour phases — perfectGas (when paired with native liquid phases)

When any liquid phase uses the native liquidProperties path (Option A above), all vapour phases
**must also use `sensibleInternalEnergy`**. Use `eConst` with `Cv` (not `Cp`):

```
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
    specie        { molWeight <MW>; }   // nMoles deprecated — omit it
    thermodynamics { Cv <Cv>; Hf 0; }  // Cv = Cp/γ (diatomic γ=1.4, monatomic γ=5/3)
    transport     { mu <mu>; Pr <Pr>; }
}
```

**Cv reference values**:
- N2 vapour:  Cv = 743   J/kg·K  (Cp=1040, γ=1.4)
- O2 vapour:  Cv = 657   J/kg·K  (Cp=920,  γ=1.4)
- H2 vapour:  Cv = 10221 J/kg·K  (Cp=14310,γ=1.4)
- He vapour:  Cv = 3116  J/kg·K  (Cp=5193, γ=5/3)

---

## Stack-trace diagnostics

When a crash shows `thermophysicalPropertiesSelector<liquidProperties>` in the stack trace:
- The failure is in a liquid phase thermo file (native liquidProperties), NOT in any vapour file.
- Do NOT blame or regenerate vapour files based on this trace — `perfectGas` never appears as `liquidProperties`.
- Inspect the failing liquid phase file first.

`nMoles` is deprecated — do NOT include it in any `specie {}` block.
