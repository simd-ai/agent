architecture
============

A five-minute read. By the end you should know which file to open
when you want to understand a behaviour.


the picture
-----------

              ┌──────────────┐
   user  ──→  │   frontend   │   (Next.js, simd-ai/simd-agent-ui)
              └──────┬───────┘
                     │ WebSocket  /ws/run, /ws/precheck, /ws/watch
                     ▼
              ┌──────────────┐        ┌────────────────┐
              │    agent     │ ←───→  │ LLM provider   │
              │  (FastAPI,   │        │ Gemini, Vertex,│
              │  this repo)  │        │   or Ollama    │
              └──────┬───────┘        └────────────────┘
                     │ HTTP + SSE
                     ▼
              ┌──────────────┐
              │  sim-server  │   (OpenFOAM v2406,
              │              │    simd-ai/simd-agent-simulation)
              └──────┬───────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  ┌──────────┐              ┌──────────┐
  │ Postgres │              │ Storage  │
  │  (Neon)  │              │ local/GCS│
  └──────────┘              └──────────┘

Three processes, three repos. Everything below is the middle one
(the agent).


what happens on a run
---------------------

The frontend opens `/ws/run` and sends a `StartRequest` describing
the mesh, the prompt, and the operation (`CFD_LINT` or
`CFD_CODEGEN_RUN`). The agent runs an enrichment pipeline on the
config, picks an OpenFOAM solver, asks the LLM to generate every
case file in parallel, validates the output with the solver's own
plugin (deterministic Python rules), packages everything into a
ZIP, and submits it to the sim-server. The sim-server runs OpenFOAM
and streams events back over SSE; the agent relays them to the
frontend as typed `AgentEvent` objects.

If the solver fails, the agent feeds the stderr to a smaller LLM
("the diagnoser"), gets back a list of files to regenerate, and
loops — up to seven times by default. See `self-healing.md`.


the modules that matter
-----------------------

(everything below lives under `simd_agent/`)

  main.py                FastAPI app, WebSocket handlers, REST
                         endpoints, VTK cache endpoints.

  models.py              All Pydantic models. `SimulationConfigV1`,
                         `StartRequest`, `AgentEvent`, `EventTypes`.
                         Single source of truth for wire formats.

  settings.py            `Settings` loaded from `.env` via
                         pydantic-settings.

  run/orchestration.py   `Orchestrator` — drives the full
                         `CFD_CODEGEN_RUN` workflow. Calls the
                         solver plugin's `validate_full()` after
                         codegen, ships the case, watches the
                         SSE stream, handles retries.

  run/enrichment/        Composable pipeline that turns the
                         freshly-validated config into a fully
                         populated config ready for downstream
                         consumers. Each step is a small async
                         function (see pipeline.py for the order).

  run/genai_codegen.py   `GenAICodeGenerator` — parallel per-file
                         LLM codegen. Delegates per-file prompts to
                         the solver plugin.

  run/value_filler/      Post-codegen pass that rewrites
                         `0/<field>` files (single-region) or
                         `0/<region>/<field>` files (CHT) so the
                         numeric values match the per-patch BCs in
                         the validated config. Closes the loop
                         between user inputs and what the LLM
                         actually wrote.

  run/error_summarizer.py     LLM-based diagnosis of solver
                              failures.

  run/code_verifier.py        Optional super-model quality gate
                              for generated OpenFOAM code.

  run/case_spec/         `CaseSpec` and `RegionSpec` — typed views
                         of the resolved config that the renderer
                         and prompt pack consume.

  run/multi_region/      Multi-region (CHT) auto-detection,
                         per-region detail extraction.

  precheck/              Two-pass LLM precheck pipeline (boundary
                         planner → per-patch agents → merge →
                         review). Runs before any simulation.

  llm/                   LLM provider plugins (gemini, vertex,
                         ollama). Auto-discovered. See
                         `llm-providers/adding-a-provider.md`.

  solvers/               OpenFOAM solver plugins. Each one is a
                         self-contained directory:
                         `solvers/<name>/solver.py` defines the
                         plugin class; `solvers/<name>/prompts/`
                         carries the per-file prompt docs. Drop a
                         new directory in, the registry finds it.

  store.py               `EventStore` — Postgres persistence for
                         runs and events.

  storage/               Object-storage backends (`local`, `gcs`).
                         Holds the case ZIPs, the VTPs from each
                         time step, the merged surface VTP.


design principles
-----------------

  - **Plugin-first.** Solvers and LLM providers are dropped in as
    directories. No global registry edits. Adding rhoCentralFoam or
    a Claude provider is a 30-line `solver_plugin = …` export.

  - **Deterministic where possible.** The renderer for multi-region
    CHT is pure Python (no LLM in the loop) — the LLM only writes
    `system/controlDict`. Anything that has a single right answer
    lives in code, not in a prompt.

  - **One config, many consumers.** The enrichment pipeline
    populates `config["case_defaults"]`, `config["regions"]`, and
    per-patch BCs in one place. Every downstream reader (`CaseSpec`,
    `RegionSpec`, prompt pack, renderer, value-filler) reads from
    there. No re-derivation, no drift.

  - **Self-healing.** Solver failures aren't terminal. The agent
    diagnoses, regenerates the affected files, and re-runs. The
    user sees one final success or one final structured failure,
    not seven progress bars.


where to look first
-------------------

  - I want to add a new solver
    → `solvers/adding-a-solver.md`, then copy `solvers/simpleFoam/`
       as a starting point.

  - I want to add a new LLM provider
    → `llm-providers/adding-a-provider.md`, then copy
       `llm/gemini/` as a starting point.

  - I want to understand the WebSocket protocol
    → `api/websocket.md` + `models.py` (the `EventTypes` enum).

  - I want to debug a failed run
    → `self-healing.md` for the loop; then read the logs in
       `[ENRICH:*]`, `[CODEGEN]`, `[VALIDATE]`, `[SIM_SERVER]`,
       `[VTK]` order.

  - I want to understand the multi-region (CHT) rendering
    → `solvers/multi-region-cht.md` +
       `solvers/families/_multi_region.py` +
       `solvers/families/_multi_region_bcs.py`.
