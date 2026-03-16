# interFoam — constant/turbulenceProperties

Always generate. For laminar, no turbulence fields are needed.

```
simulationType RAS;   // or laminar
RAS
{
    RASModel    <modelName>;   // kOmegaSST or kEpsilon from CaseSpec
    turbulence  on;
    printCoeffs on;
}
```
