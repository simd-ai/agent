# buoyantSimpleFoam — constant/thermophysicalProperties

Uses `heRhoThermo` (density-based thermo) with variable density — appropriate for
buoyancy-driven flows where ρ = f(T, p).

## Path A: Gas (air, combustion products, smoke) — default

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
    specie
    {
        nMoles      1;
        molWeight   28.96;   // air [g/mol]; adjust for other gases
    }
    thermodynamics
    {
        Cp          1004;    // J/kg·K
        Hf          0;
    }
    transport
    {
        mu          1.831e-5;   // dynamic viscosity [Pa·s] at ~293 K
        Pr          0.705;
    }
}
```

## Path B: Liquid with significant temperature variation (e.g. water, oil)

Use `icoPolynomial` when ΔT > ~20 K and the fluid is a liquid with density varying with T.

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
        nMoles      1;
        molWeight   18.015;   // water [g/mol]
    }
    equationOfState
    {
        rhoCoeffs<8> ( 1000 -0.5 0 0 0 0 0 0 );  // ρ = 1000 - 0.5*(T-Tref)
    }
    thermodynamics
    {
        Hf          0;
        Sf          0;
        CpCoeffs<8> ( 4180 0 0 0 0 0 0 0 );
    }
    transport
    {
        muCoeffs<8>  ( 1e-3 0 0 0 0 0 0 0 );
        kappaCoeffs<8> ( 0.6 0 0 0 0 0 0 0 );
    }
}
```

## Notes

- `type heRhoThermo` (density-based) — do NOT use `hePsiThermo` (psi-based) for buoyant solvers.
- `energy sensibleEnthalpy` — consistent with `div(phi,h)` in fvSchemes.
- For typical air HVAC cases, Path A (perfectGas) is correct.
- For natural convection in liquids (water, glycol), use Path B with appropriate coefficients.
- File is always named `constant/thermophysicalProperties` in OF2406 ESI.
