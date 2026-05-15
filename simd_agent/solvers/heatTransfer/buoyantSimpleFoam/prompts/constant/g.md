# buoyantSimpleFoam — constant/g

**REQUIRED** — omitting this file causes a fatal IO error at startup:
`"Cannot find file g in constant/"`.

```
dimensions  [0 1 -2 0 0 0 0];
value       (0 -9.81 0);
```

## Gravity direction conventions

The gravity vector must point in the downward direction relative to your mesh geometry:

| Geometry orientation | Gravity vector |
|---------------------|----------------|
| Y is vertical (up)  | `(0 -9.81 0)` |
| Z is vertical (up)  | `(0 0 -9.81)` |
| X is vertical (up)  | `(-9.81 0 0)` |

Use standard gravitational acceleration: 9.81 m/s².

## Notes

- Unlike rhoSimpleFoam/rhoPimpleFoam, buoyantSimpleFoam ALWAYS needs `constant/g`.
- The buoyancy source in the momentum equation is `−(rho − rho_ref) * g`.
- The p_rgh formulation is: `p_rgh = p − rho * (g · x)` where `x` is position from origin.
