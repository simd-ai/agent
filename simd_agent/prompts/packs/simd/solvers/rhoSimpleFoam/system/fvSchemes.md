# rhoSimpleFoam — system/fvSchemes

**Steady-state compressible** solver. Use `steadyState` for `ddtSchemes`.

## Why `bounded Gauss upwind` as default

With `default none`, OpenFOAM will crash with a fatal "cannot find scheme" error for
any div term not explicitly listed — including internal solver fields like `Ekp`,
turbulence model auxiliaries, or MRF momentum source terms. Using
`default bounded Gauss upwind` prevents all such crashes and is appropriate for
steady compressible flows. Critical terms are then overridden with higher-order schemes.

## Required div terms for rhoSimpleFoam

`div(phi,K)` and `div(phid,p)` are **mandatory** — omitting them with `default none`
causes a fatal crash:
```
Entry div(phi,K) not found in dictionary system/fvSchemes/divSchemes
```

Emit ONLY `div(phi,h)` OR `div(phi,e)` — whichever matches `thermoType.energy`:
- `sensibleEnthalpy` → `div(phi,h)`
- `sensibleInternalEnergy` → `div(phi,e)`

## Viscous stress term

Compressible: `div(((rho*nuEff)*dev2(T(grad(U)))))` — must use `dev2` and `rho*nuEff`.
Incompressible would use `div((nuEff*dev2(T(grad(U)))))` — do NOT use that here.

## wallDist

Include `wallDist { method meshWave; }` when `sim_type` is `RAS` or `LES`.
Omit entirely for `laminar`.

## Template

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    // Safe default — covers any internal solver term without crashing
    default                                   bounded Gauss upwind;

    // IMPORTANT: use first-order upwind for U — linearUpwind causes SIGFPE in
    // symGaussSeidelSmoother after a few iterations due to gradient-correction
    // overshoots making the matrix non-diagonally dominant
    div(phi,U)                                bounded Gauss upwind;

    // Energy convection — emit ONLY the one matching thermoType.energy
    div(phi,h)                                bounded Gauss upwind;   // sensibleEnthalpy
    // div(phi,e)                             bounded Gauss upwind;   // sensibleInternalEnergy

    // Kinetic energy + pressure-work — REQUIRED for compressible energy equation
    div(phi,K)                                bounded Gauss upwind;
    div(phi,Ekp)                              bounded Gauss upwind;
    div(phid,p)                               Gauss limitedLinear 1;

    // Turbulence fields — include only those that exist per CaseSpec.turbulence_fields
    div(phi,k)                                bounded Gauss linearUpwind grad(k);
    div(phi,omega)                            bounded Gauss linearUpwind grad(omega);
    div(phi,epsilon)                          bounded Gauss linearUpwind grad(epsilon);

    // Viscous stress — compressible form: MUST use dev2 and rho*nuEff
    div(((rho*nuEff)*dev2(T(grad(U)))))       Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// Include only when sim_type = RAS or LES:
wallDist { method meshWave; }
```

## Checklist

- [ ] `ddtSchemes default steadyState`
- [ ] `div(phi,h)` OR `div(phi,e)` — exactly one, matching `thermophysicalProperties`
- [ ] `div(phi,K)` present
- [ ] `div(phid,p)` present
- [ ] `div(phi,Ekp)` present
- [ ] Viscous term uses `rho*nuEff` and `dev2`
- [ ] Only turbulence fields in `CaseSpec.turbulence_fields` are listed
- [ ] `wallDist` present when `sim_type` is RAS/LES
