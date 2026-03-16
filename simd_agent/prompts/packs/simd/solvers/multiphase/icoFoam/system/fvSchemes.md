# icoFoam — system/fvSchemes

icoFoam is laminar — no turbulence div terms, no `wallDist`.

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default     none;
    div(phi,U)  Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
```
