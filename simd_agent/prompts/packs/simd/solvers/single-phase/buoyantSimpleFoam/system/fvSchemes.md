# buoyantSimpleFoam — system/fvSchemes

**Steady-state buoyancy solver**. Uses `steadyState` for `ddtSchemes`.

## Key differences from rhoSimpleFoam

- **No `div(phid,p)`** — buoyantSimpleFoam does NOT have a pressure-dilatation term.
  Including it causes a fatal "cannot find scheme div(phid,p)" crash.
- **`fluxRequired { p_rgh; }`** — the flux-correction loop uses `p_rgh`, not `p`.
- Viscous stress: same compressible form `div(((rho*nuEff)*dev2(T(grad(U)))))`.
- `div(phi,K)` is required (kinetic energy correction in enthalpy equation).

## Template

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    // Safe default — covers any auxiliary term without crashing
    default                                    bounded Gauss upwind;

    div(phi,U)                                 bounded Gauss limitedLinear 0.2;
    div(phi,h)                                 bounded Gauss limitedLinear 0.2;
    div(phi,K)                                 bounded Gauss limitedLinear 0.2;

    // Turbulence — include only fields in CaseSpec.turbulence_fields
    div(phi,k)                                 bounded Gauss limitedLinear 1;
    div(phi,omega)                             bounded Gauss limitedLinear 1;
    div(phi,epsilon)                           bounded Gauss limitedLinear 1;

    // Compressible viscous stress — MUST use dev2 and rho*nuEff
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

- [ ] `ddtSchemes default steadyState`
- [ ] `div(phi,h)` present (sensibleEnthalpy)
- [ ] `div(phi,K)` present
- [ ] NO `div(phid,p)` — this is NOT a rho* solver
- [ ] Viscous term uses `rho*nuEff` and `dev2`
- [ ] `fluxRequired { p_rgh; }` — not `p`
- [ ] `wallDist` present when `sim_type` is RAS/LES
- [ ] Only turbulence fields from CaseSpec.turbulence_fields are listed
