# rhoSimpleFoam — constant/turbulenceProperties

Always generate this file. Use `CaseSpec.sim_type` and `CaseSpec.turbulence_model`.

```
simulationType  <sim_type>;   // laminar | RAS | LES

// Include the sub-dict matching simulationType:
RAS
{
    RASModel        <turbulence_model>;   // kOmegaSST | kEpsilon | SpalartAllmaras | ...
    turbulence      on;
    printCoeffs     on;
}
```

For laminar flow:
```
simulationType  laminar;
```
