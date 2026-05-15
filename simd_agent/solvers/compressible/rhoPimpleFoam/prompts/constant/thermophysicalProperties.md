# rhoPimpleFoam — constant/thermophysicalProperties

Build from `CaseSpec` only — do not invent values.

## Selection rules

| Condition | `type` | EOS | transport |
|-----------|--------|-----|-----------|
| compressible gas | `hePsiThermo` | `perfectGas` | `sutherland` or `const` |
| liquid with heat transfer OR cryogenic (T < 200 K) | `heRhoThermo` | `icoPolynomial` | `polynomial` |
| isothermal liquid (no heat transfer, constant T) | `heRhoThermo` | `rhoConst` | `const` |

**NEVER use `rhoConst` when `enable_heat_transfer=true` or inlet T ≠ wall T.** With `rhoConst` the density is a global constant and never changes with temperature — the energy equation is solved but has zero effect on ρ, breaking physical accuracy in compressible transient simulations.

**NEVER use `rhoConst` for cryogenic liquids (LN2, LH2, LOX) or any liquid with significant temperature variation.** Density of real liquids is strongly T-dependent; `rhoConst` introduces mass-conservation errors that cause divergence.

**`icoPolynomial`**: `rhoCoeffs<8>` = ρ(T) = a0 + a1·T. Compute from inlet conditions:
- LN2/LOX (~77–120 K): dρ/dT ≈ −4.7 kg/m³/K
- LH2 (20–33 K): dρ/dT ≈ −0.7 kg/m³/K
- Water/oil (>250 K): dρ/dT ≈ −0.5 kg/m³/K
- `a0 = ρ_inlet − a1 × T_inlet`

Default energy: `sensibleEnthalpy` → field name `h`. Use `sensibleInternalEnergy` → `e` only if config specifies it.

## CRITICAL key naming

- Inside `thermoType {}`: for `perfectGas`/`rhoConst`: keyword is `thermo  hConst;`
- Inside `thermoType {}`: for `icoPolynomial`: keywords are `transport  polynomial;` and `thermo  hPolynomial;`
- Inside `mixture {}`: for `hConst`: sub-dict is `thermodynamics { Cp …; Hf 0; }`
- Inside `mixture {}`: for `hPolynomial`: sub-dict is `thermodynamics { Hf 0; Sf 0; CpCoeffs<8> (…); }`

## Template

### Gas (perfectGas)
```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { molWeight <molWeight>; }
    thermodynamics { Cp <Cp>; Hf 0; }
    transport   { mu <mu>; Pr <Pr>; }
}
```

### Liquid with heat transfer or cryogenic (icoPolynomial)

**CRITICAL**: `icoPolynomial` ONLY works with `transport=polynomial` + `thermo=hPolynomial`.
`const`+`hConst`+`icoPolynomial` → "Unknown fluidThermo type" FOAM fatal error.
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
    specie      { molWeight <molWeight>; }
    thermodynamics                   // hPolynomial: CpCoeffs<8> + Hf + Sf (NOT plain Cp)
    {
        Hf              0;
        Sf              0;
        CpCoeffs<8>     (<Cp> 0 0 0 0 0 0 0);
    }
    transport                        // polynomial: muCoeffs + kappaCoeffs (NOT mu/Pr)
    {
        muCoeffs<8>     (<mu> 0 0 0 0 0 0 0);
        kappaCoeffs<8>  (<kappa> 0 0 0 0 0 0 0);  // kappa = mu*Cp/Pr
    }
    equationOfState
    {
        // ρ(T) = a0 + a1*T; a0 = ρ_inlet − a1*T_inlet
        rhoCoeffs<8>    (<a0> <a1> 0 0 0 0 0 0);
    }
}
```
