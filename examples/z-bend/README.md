z-bend
======

Transient turbulent water flow through a Z-shaped pipe. Standard
benchmark shape for separation and reattachment around bends.

  - solver:     pimpleFoam
  - turbulence: kOmegaSST
  - regime:     transient, incompressible
  - BCs:        inlet U = 2.3 m/s,
                outlet p = atmospheric,
                walls noSlip
  - duration:   4 seconds (transient end time)

A short, terse prompt — the agent's enrichment pipeline fills in
everything else from the mesh and the standard defaults (atmospheric
outlet, water at room T, deltaT picked from the mesh).

See `Documentation/examples/z-bend.md` for the walkthrough.


reproduce with OpenFOAM directly
--------------------------------

    cd case && pimpleFoam


reproduce via the agent
-----------------------

    simd run examples/z-bend/prompt.txt \
             examples/z-bend/mesh/z-bend.msh

Or upload the mesh + paste the prompt in the frontend at
http://localhost:3000 — same backend.
