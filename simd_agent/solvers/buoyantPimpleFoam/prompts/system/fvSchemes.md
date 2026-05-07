# buoyantPimpleFoam — system/fvSchemes

**Transient buoyancy solver**. Uses `Euler` (or `CrankNicolson 0.9`) for `ddtSchemes`.

## Key differences from rhoPimpleFoam

- **No `div(phid,p)`** — buoyantPimpleFoam does not have a pressure-dilatation term.
- **`fluxRequired { p_rgh; }`** — flux correction uses p_rgh, not p.
- `div(phi,h)` for energy (sensibleEnthalpy).
- `div(phi,K)` for kinetic energy correction.
- For large ΔT (> 100 K), use `bounded Gauss upwind` for `div(phi,h)` to prevent overshoot.

## Template

```
ddtSchemes      { default Euler; }   // CrankNicolson 0.9 for higher accuracy
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                                    bounded Gauss upwind;

    div(phi,U)                                 Gauss limitedLinear 0.2;
    div(phi,h)                                 Gauss limitedLinear 0.2;  // upwind if ΔT > 100K
    div(phi,K)                                 Gauss linear;

    // Turbulence — include only fields in CaseSpec.turbulence_fields
    div(phi,k)                                 Gauss limitedLinear 1;
    div(phi,omega)                             Gauss limitedLinear 1;
    div(phi,epsilon)                           Gauss limitedLinear 1;

    // Compressible viscous stress
    div(((rho*nuEff)*dev2(T(grad(U)))))        Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// REQUIRED: fluxRequired uses p_rgh (NOT p)
fluxRequired
{
    default  no;
    p_rgh;
}

// Include only when sim_type = RAS or LES:
wallDist { method meshWave; }
```

## Checklist

- [ ] `ddtSchemes default Euler` (or CrankNicolson 0.9)
- [ ] `div(phi,h)` present — use `upwind` if ΔT > 100 K
- [ ] `div(phi,K)` present
- [ ] NO `div(phid,p)`
- [ ] `fluxRequired { p_rgh; }` not `p`
- [ ] Viscous term uses `rho*nuEff` and `dev2`
- [ ] `wallDist` present for RAS/LES
