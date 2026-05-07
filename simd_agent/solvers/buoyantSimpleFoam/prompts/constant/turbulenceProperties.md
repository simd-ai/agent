# buoyantSimpleFoam — constant/turbulenceProperties

Same as other compressible single-phase solvers.

```
simulationType      RAS;
RAS
{
    RASModel        kOmegaSST;   // or kEpsilon — both work for natural convection
    turbulence      on;
    printCoeffs     on;
}
```

For laminar natural convection (low Ra number, Ra < ~10⁹):
```
simulationType  laminar;
```
