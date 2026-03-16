# compressibleInterFoam — system/fvOptions

**DO NOT GENERATE THIS FILE.**

`limitTemperature` (and all fvOptions that access temperature) calls `he()` on `twoPhaseMixtureThermo`.
`twoPhaseMixtureThermo` does **not** implement `he()` — including `system/fvOptions` causes:

```
FOAM FATAL ERROR: Not implemented
    From virtual const volScalarField& Foam::twoPhaseMixtureThermo::he() const
```

This is a crash at startup before any time step runs.

**Do not include `system/fvOptions` in any compressibleInterFoam case.**
