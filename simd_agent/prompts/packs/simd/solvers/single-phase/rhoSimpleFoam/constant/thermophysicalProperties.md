# rhoSimpleFoam — constant/thermophysicalProperties

## Thermo type selection

| Fluid intent | `type` | `equationOfState` |
|---|---|---|
| Compressible ideal gas | `hePsiThermo` | `perfectGas` |
| Liquid with heat transfer OR cryogenic (T < 200 K) | `heRhoThermo` | `icoPolynomial` |
| Isothermal liquid (no heat transfer, constant T) | `heRhoThermo` | `rhoConst` |

**NEVER use `rhoConst` when `enable_heat_transfer=true` or inlet T ≠ wall T.** With `rhoConst` the density is a global constant and never responds to temperature — the energy equation is solved but has zero effect on ρ, making the compressible solver physically meaningless.

**NEVER use `rhoConst` when temperature varies significantly** — density of real liquids (especially cryogenic: LN2, LH2, LOX) changes strongly with T. `rhoConst` introduces large mass-conservation errors and can cause divergence.

**`icoPolynomial`**: `rhoCoeffs<8>` implements ρ(T) = a0 + a1·T + ... Use a linear fit:
- Compute from known ρ at T_inlet and a typical slope for the fluid
- LN2 / LOX range (77–120 K): dρ/dT ≈ −4.7 kg/m³/K
- LH2 range (20–33 K): dρ/dT ≈ −0.7 kg/m³/K
- Water / oil (250–400 K): dρ/dT ≈ −0.5 kg/m³/K
- Formula: `a0 = ρ_inlet − a1 × T_inlet`, `rhoCoeffs<8> (a0 a1 0 0 0 0 0 0)`

Default when not specified in config: use `hePsiThermo` + `perfectGas`.
If `CaseSpec.rho` is set (user provided density) AND heat transfer is active or T < 200 K: use `heRhoThermo` + `icoPolynomial`.
If `CaseSpec.rho` is set and no significant temperature variation: use `heRhoThermo` + `rhoConst`.

## CRITICAL key naming — wrong names cause FOAM FATAL IO ERROR

| Location | Correct key | WRONG key |
|---|---|---|
| Inside `thermoType {}` | `thermo  hConst;` | ~~`thermodynamics hConst;`~~ |
| Inside `mixture {}` | `thermodynamics { Cp …; }` | ~~`thermo { Cp …; }`~~ |

## Energy field name

`thermoType.energy` controls the transported variable name:
- `sensibleEnthalpy` → field name `h` (default)
- `sensibleInternalEnergy` → field name `e`

This MUST match:
- `div(phi,h)` or `div(phi,e)` in `fvSchemes`
- `"(U|h|…)"` regex in `fvSolution`
- `residualControl { h …; }` in `fvSolution`

## Templates

### Gas (perfectGas)
```
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;           // ← MUST be 'thermo' here
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.97; }
    thermodynamics { Cp 1005; Hf 0; }  // ← MUST be 'thermodynamics' here
    transport   { mu 1.8e-5; Pr 0.713; }
}
```

### Liquid with heat transfer or cryogenic (icoPolynomial — REQUIRED for LN2/LH2/LOX)

**CRITICAL**: `icoPolynomial` is ONLY valid with `transport=polynomial` and `thermo=hPolynomial`.
Using `const`+`hConst`+`icoPolynomial` → "Unknown fluidThermo type" FOAM fatal error.
Valid chain: `heRhoThermo<pureMixture<polynomial<hPolynomial<icoPolynomial<specie>>,sensibleEnthalpy>>>`

```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       polynomial;      // MUST be polynomial (not const) with icoPolynomial
    thermo          hPolynomial;     // MUST be hPolynomial (not hConst) with polynomial transport
    equationOfState icoPolynomial;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.014; }
    thermodynamics                   // hPolynomial: CpCoeffs<8> + Hf + Sf (NOT plain Cp)
    {
        Hf              0;
        Sf              0;
        CpCoeffs<8>     (2042 0 0 0 0 0 0 0);   // constant Cp; add higher terms if known
    }
    transport                        // polynomial: muCoeffs + kappaCoeffs (NOT mu/Pr)
    {
        muCoeffs<8>     (1.58e-4 0 0 0 0 0 0 0);
        kappaCoeffs<8>  (0.323 0 0 0 0 0 0 0);  // kappa = mu*Cp/Pr
    }
    equationOfState
    {
        // ρ(T) = a0 + a1*T  (linear fit)
        // LN2 at 77 K, ρ=808: a0 = 808 - (-4.7)*77 = 1169.9, a1 = -4.7
        rhoCoeffs<8>    (1169.9 -4.7 0 0 0 0 0 0);
    }
}
```

### Isothermal liquid (rhoConst — only when temperature is constant)
```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie      { nMoles 1; molWeight 28.97; }
    thermodynamics { Cp 4182; Hf 0; }
    transport   { mu 1e-3; Pr 7.0; }
    equationOfState { rho 1000; }
}
```

## Rules

1. For `icoPolynomial`: compute `a0 = ρ_inlet − a1 × T_inlet` using actual CaseSpec values.
2. For `perfectGas`: omit `equationOfState` sub-dict inside `mixture {}`.
3. Use physical values from CaseSpec: `cp`, `mu`, `Pr`, `rho`, `inlet_temperature`.
4. Do not add a `0/h` or `0/e` file — the thermo package reads `0/T` at startup.
