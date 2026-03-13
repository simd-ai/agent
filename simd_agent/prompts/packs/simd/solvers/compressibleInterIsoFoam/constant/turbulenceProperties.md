# compressibleInterIsoFoam — constant/turbulenceProperties

Always generate. If laminar: no turbulence fields.

```
simulationType RAS;
RAS
{
    RASModel    <modelName>;
    turbulence  on;
    printCoeffs on;
}
```
