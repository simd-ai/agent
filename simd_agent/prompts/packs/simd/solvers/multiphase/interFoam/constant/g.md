# interFoam — constant/g

**Always generate** — required even if gravity is disabled.

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);   // use (0 0 0) if CaseSpec.gravity = false
```
