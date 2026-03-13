# interIsoFoam — system/fvSchemes

Same as interFoam. isoAdvection changes the reconstruction algorithm, not the fvSchemes entries.

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

wallDist { method meshWave; }
```
