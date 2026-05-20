Documentation
=============

Everything beyond what's in the top-level README. Pick the section
you need.


getting it running
------------------

  - installation.md         bare-metal: Python venv, system OpenFOAM,
                            local or external Postgres
  - deployment.md           production: GCS storage, Neon Auth,
                            Vertex AI


design
------

  - architecture.md         how the orchestrator, plugins, and
                            deterministic renderer fit together
  - self-healing.md         the diagnose → retry → fix loop, with a
                            real-world walkthrough


reference
---------

  - api/rest.md             REST endpoints
  - api/websocket.md        WebSocket frames (/ws/run, /ws/precheck,
                            /ws/watch)
  - api/events.md           AgentEvent payloads
  - solvers/README.md       supported OpenFOAM solvers
  - llm-providers/          per-provider setup notes
                            (gemini.md, vertex.md, ollama.md)


extending it
------------

  - solvers/adding-a-solver.md          plugin contract — drop a
                                        directory in, no registry edit
  - llm-providers/adding-a-provider.md  LLMProvider contract


examples
--------

  - examples/u-shape-pipe.md
  - examples/z-bend.md
  - examples/inner-outer-pipe.md
  - examples/cylindrical-cht.md

Each walkthrough shows the prompt, the generated case, and the
expected result. The runnable cases live under `examples/` at the
repo root.
