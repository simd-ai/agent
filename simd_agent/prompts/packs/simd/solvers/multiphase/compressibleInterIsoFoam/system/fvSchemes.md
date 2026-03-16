# compressibleInterIsoFoam — system/fvSchemes

## Algorithm context

Same as compressibleInterFoam. isoAdvection replaces MULES for geometric interface reconstruction but the equation set is identical.

Main loop: **alphaEqnSubCycle (isoAdvector) → turbulence.correctPhasePhi() → UEqn → TEqn (or EEqn) → pEqn → turbulence.correct()**

ALL of these div terms will be looked up at runtime:
- `div(rhoPhi,U)` — momentum (UEqn)
- `div(rhoPhi,T)` — temperature (TEqn — OF Foundation v9)
- `div(rhoPhi,he)` and `div(rhoPhi,h)` — enthalpy (EEqn — OF ESI 2406 with heRhoThermo)
- `div(rhoPhi,K)` — kinetic energy (explicit in energy equation)
- `div(phi,alpha)`, `div(phirb,alpha)` — generic alpha (alphaEqn subcycle)
- `div(phi,alpha.<phaseName>)`, `div(phirb,alpha.<phaseName>)` — phase-specific alpha
- `div(rhoPhi,k)`, `div(rhoPhi,omega)` — turbulence scalars
- `div((rho*nuEff)*dev2(T(grad(U))))` — Reynolds stress tensor

## divSchemes default — CRITICAL rules

- MUST be `Gauss linear` — NEVER `none`, `Gauss upwind`, or `bounded Gauss upwind`
- `none`: OF 2406 treats it as a scheme name → base constructor reads interpolation scheme from empty stream → **"attempt to read beyond EOF"**
- `Gauss upwind`: same EOF (upwind reads flux field name from stream, empty after 'upwind')
- `bounded Gauss upwind`: **"unknown div scheme bounded"** (Tensor fields use default and don't support bounded)
- `Gauss linear` is safe for ALL field types (scalar/vector/tensor/symmTensor)

## Template

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    // CRITICAL: 'Gauss linear' — see rules above
    default                                 Gauss linear;

    div(rhoPhi,U)                           bounded Gauss linearUpwind grad(U);

    // Alpha advection — generic AND phase-specific entries both required
    div(phi,alpha)                          Gauss vanLeer;
    div(phirb,alpha)                        Gauss linear;
    div(phi,alpha.<phase1Name>)             Gauss vanLeer;
    div(phirb,alpha.<phase1Name>)           Gauss linear;

    // Energy — list BOTH: TEqn (OF Foundation/v9) and heRhoThermo (OF ESI 2406)
    div(rhoPhi,he)                          bounded Gauss linearUpwind grad(he);
    div(rhoPhi,h)                           bounded Gauss linearUpwind grad(h);
    div(rhoPhi,T)                           bounded Gauss linearUpwind grad(T);
    div(rhoPhi,K)                           bounded Gauss upwind;

    // Turbulence
    div(rhoPhi,k)                           bounded Gauss upwind;
    div(rhoPhi,omega)                       bounded Gauss upwind;
    div(rhoPhi,epsilon)                     bounded Gauss upwind;

    // Stress tensor (required for turbulence viscous term)
    div((rho*nuEff)*dev2(T(grad(U))))       Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// fluxRequired: p_rgh + pcorr (for correctPhi) + all alpha fields
fluxRequired
{
    default              no;
    p_rgh;
    pcorr;
    alpha.<phase1Name>;
}

wallDist { method meshWave; }
```

## Additional rules
- Do NOT add `interface interfaceCompression` under `interpolationSchemes`
- `pcorr` in `fluxRequired` is required for `CorrectPhi.H` (mesh flux correction)
- Replace `<phase1Name>` with the actual phase name from CaseSpec (e.g. `liquidNitrogen`)
