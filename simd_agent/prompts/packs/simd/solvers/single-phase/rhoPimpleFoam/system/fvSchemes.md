# rhoPimpleFoam — system/fvSchemes

**Why `bounded Gauss upwind` default**: `default none` crashes fatally on any div term the solver requests internally (Ekp, turbulence aux). Safe fallback prevents these crashes; explicit entries override with higher-order schemes.

## Mandatory div terms

- `div(phi,K)` and `div(phid,p)` — REQUIRED for the compressible energy equation. Omit with `default none` → fatal crash.
- `div(phi,Ekp)` — kinetic+pressure energy flux in transient runs.
- `div(((rho*nuEff)*dev2(T(grad(U)))))` — compressible viscous stress. Use `dev2`, not `dev`.
- Emit ONLY `div(phi,h)` OR `div(phi,e)` — whichever matches `thermoType.energy`.

## Scheme selection for div(phi,h) — stability vs accuracy

`linearUpwind` is second-order and accurate for smooth temperature fields.
For cases with large temperature spans (inlet T vs wall T differ by > 100 K) or cryogenic fluids
(LN2, LH2, LOX), `linearUpwind` can overshoot enthalpy, produce h < 0, and crash with
"Negative temperature" or FOAM FATAL. Use first-order `upwind` instead:

| Condition | div(phi,h) scheme |
|---|---|
| Moderate T gradient (ΔT < 100 K) | `bounded Gauss linearUpwind grad(h)` |
| Large T gradient (ΔT > 100 K) or cryogenic | `bounded Gauss upwind` |

## Template

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                                 bounded Gauss upwind;

    div(phi,U)                              bounded Gauss linearUpwind grad(U);

    // emit ONLY one of these — match thermoType.energy
    div(phi,h)                              bounded Gauss linearUpwind grad(h);
    // div(phi,e)                           bounded Gauss linearUpwind grad(e);

    div(phi,K)                              bounded Gauss linearUpwind grad(K);
    div(phi,Ekp)                            bounded Gauss upwind;
    div(phid,p)                             Gauss limitedLinear 1;

    div(phi,k)                              bounded Gauss linearUpwind grad(k);
    div(phi,omega)                          bounded Gauss linearUpwind grad(omega);
    div(phi,epsilon)                        bounded Gauss linearUpwind grad(epsilon);

    div(((rho*nuEff)*dev2(T(grad(U)))))     Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// Include wallDist ONLY when turbulence is enabled (RAS/LES)
wallDist { method meshWave; }
```
