examples
========

Four end-to-end cases. Each ships with:

  - `prompt.txt` — the natural-language prompt used to generate it
  - `mesh/<name>.msh` — the gmsh mesh
  - `case/` — the complete OpenFOAM case the agent produced


ordered simple → complex
------------------------

  | example                            | solver                          | doc                    |
  |------------------------------------|---------------------------------|-------------------------|
  | `examples/u-shape-pipe/`           | `rhoSimpleFoam` (kOmegaSST)     | u-shape-pipe.md        |
  | `examples/z-bend/`                 | `pimpleFoam` (kOmegaSST)        | z-bend.md              |
  | `examples/inner-outer-pipe/`       | `chtMultiRegionSimpleFoam`      | inner-outer-pipe.md    |
  | `examples/cylindrical-cht/`        | `buoyantBoussinesqSimpleFoam`   | cylindrical-cht.md     |
  |                                    | (kOmegaSST)                     |                        |

Note: the `cylindrical-cht/` directory is a misnomer — the case is
natural convection around a heated cylinder, not conjugate heat
transfer. See the walkthrough for details and a rename suggestion.


running them
------------

Two paths:

  1. **Via the agent** — open http://localhost:3000, upload the
     mesh, paste the prompt, hit Run. The agent regenerates the
     case from scratch each time.

  2. **Directly with OpenFOAM** — `cd examples/<name>/case &&
     <solver>`. This bypasses the agent entirely. Useful for
     OpenFOAM users who want to study the case files without
     setting up the agent.

The walkthrough for each example explains the prompt, what the
agent does with it, and what the result should look like.
