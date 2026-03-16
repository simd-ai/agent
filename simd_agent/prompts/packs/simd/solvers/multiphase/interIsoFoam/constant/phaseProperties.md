# interIsoFoam — constant/phaseProperties

Same structure as interFoam. Use phase names from CaseSpec.

```
phases ( <phase1Name> <phase2Name> );

<phase1Name>
{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] <nu1>;
    rho             [1 -3 0 0 0 0 0] <rho1>;
}

<phase2Name>
{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] <nu2>;
    rho             [1 -3 0 0 0 0 0] <rho2>;
}

sigma           [1 0 -2 0 0 0 0] <sigma>;
```
