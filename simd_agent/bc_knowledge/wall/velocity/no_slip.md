# Wall no-slip (`noSlip`)

## When to use
Use this when the user says:
- no slip
- stationary wall
- wall velocity is zero
- fixed wall

This is the default velocity BC for ordinary stationary solid walls in the MVP.

## Purpose
Enforces zero wall velocity.

## UI fields to expose
Minimal MVP UI:
- `wallMotion = stationary`
- `slipMode = noSlip`

No numeric input required unless you support moving walls.

## OpenFOAM mapping
Primary BC family: `type: noSlip`

`noSlip` is a wrapper around the fixed condition and sets wall velocity to zero.
No additional dictionary entries are needed for stationary no-slip walls.

## Planner rule
If the prompt contains:
- "no slip"
- "stationary wall"
- equivalent wording
- or if no wall motion is specified (default assumption)

Then:
- select this BC with very high confidence
- do not let the LLM choose `slip`, `partialSlip`, or moving-wall variants

## Wall turbulence (MVP)
For turbulence fields at walls use standard wall functions:
- `k`: `kqRWallFunction`
- `omega`: `omegaWallFunction`
- `epsilon`: `epsilonWallFunction`
- `nut`: `nutkWallFunction`

These are the correct wall-function types for high-Re turbulence models like `kOmegaSST`.

## Extension path (post-MVP)
Later, add separate files for:
- `slip`
- `partialSlip`
- `movingWallVelocity`
- `rotatingWallVelocity`
- `translatingWallVelocity`

For MVP, `noSlip` is enough for most internal-flow setups.
