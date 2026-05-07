# compressibleInterFoam — constant/turbulenceProperties

Always generate.

```
simulationType RAS;   // or laminar
RAS
{
    RASModel    <modelName>;
    turbulence  on;
    printCoeffs on;
}
```

If laminar: do NOT generate `k`, `omega`, `epsilon`, `mut`.
