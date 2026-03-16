# icoFoam — constant/transportProperties

Contains ONLY kinematic viscosity `nu`. Do NOT add `rho`, `mu`, or any other property.

```
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] <nu_value>;
```

Use `CaseSpec.nu` for the value.
