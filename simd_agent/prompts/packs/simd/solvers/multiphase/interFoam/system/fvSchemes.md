# interFoam — system/fvSchemes

Use `bounded Gauss upwind` as default (not `none`) to avoid crashes on internal terms.

Alpha advection MUST use `Gauss vanLeer` (bounded) + `Gauss linear` (compression).
Viscous term: `nuEff*dev2` for incompressible.

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                             bounded Gauss upwind;

    div(phi,U)                          bounded Gauss linearUpwind grad(U);

    div(phi,alpha)                      Gauss vanLeer;
    div(phirb,alpha)                    Gauss linear;

    div((nuEff*dev2(T(grad(U)))))       Gauss linear;

    // turbulence — only if active
    div(phi,k)                          bounded Gauss linearUpwind grad(k);
    div(phi,omega)                      bounded Gauss linearUpwind grad(omega);
    div(phi,epsilon)                    bounded Gauss linearUpwind grad(epsilon);
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// Include wallDist only when turbulence is enabled
wallDist { method meshWave; }
```
