# rhoPimpleFoam — constant/turbulenceProperties

```
simulationType RAS;   // or laminar
RAS
{
    RASModel    <modelName>;   // kOmegaSST or kEpsilon from CaseSpec
    turbulence  on;
    printCoeffs on;
}
```

For laminar: `simulationType laminar;` only — no RAS block.
