# simpleFoam — constant/turbulenceProperties

Always generate. Use `CaseSpec.sim_type` and `CaseSpec.turbulence_model`.

```
simulationType  <sim_type>;   // laminar | RAS | LES

RAS
{
    RASModel        <turbulence_model>;   // kOmegaSST | kEpsilon | SpalartAllmaras | ...
    turbulence      on;
    printCoeffs     on;
}
```

For laminar:
```
simulationType  laminar;
```
