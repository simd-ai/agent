# pimpleFoam — constant/turbulenceProperties

Same as simpleFoam. Use `CaseSpec.sim_type` and `CaseSpec.turbulence_model`.

```
simulationType  <sim_type>;   // laminar | RAS | LES

RAS
{
    RASModel        <turbulence_model>;
    turbulence      on;
    printCoeffs     on;
}
```
