examples
========

Four end-to-end CFD cases. Each is self-contained — you can run
the agent on the prompt + mesh, or you can run the case files
under `case/` directly with OpenFOAM and bypass the agent
entirely.

  u-shape-pipe/        compressible inverted-U duct,
                       rhoSimpleFoam + kOmegaSST
  z-bend/              transient turbulent water pipe,
                       pimpleFoam + kOmegaSST
  inner-outer-pipe/    2D LN2/water counter-flow regasifier,
                       chtMultiRegionSimpleFoam
  cylindrical-cht/     natural convection around a heated cylinder,
                       buoyantBoussinesqSimpleFoam (kOmegaSST)

Walk-throughs (the why, the prompt, what to look for) live under
Documentation/examples/.


layout per example
------------------

    examples/<name>/
      ├── README.md         ← quick reference
      ├── prompt.txt        ← the natural-language prompt
      ├── mesh/
      │   └── <name>.msh    ← gmsh mesh
      └── case/             ← complete OpenFOAM case
          ├── 0/
          ├── constant/
          └── system/


running them
------------

Two ways:

  - **Via the agent — CLI** (after `pip install -e .`):

        simd run examples/<name>/prompt.txt \
                 examples/<name>/mesh/<name>.msh

    Walks you through mesh upload, precheck, interactive patch
    review, then streams the run. See Documentation/cli.md.

  - **Via the agent — frontend**: open http://localhost:3000,
    upload the mesh, paste the prompt, click Run. Same backend.

  - **Directly with OpenFOAM (no AI)**:

        cd examples/<name>/case && <solver>

    Bypasses the agent. The solver name is in each example's
    walkthrough.
