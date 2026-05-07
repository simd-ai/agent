# simd_agent/api — REST API routers
#
# Each router handles a specific resource:
#   users.py         — /api/users/*
#   simulations.py   — /api/simulations/*
#   meshes.py        — /api/meshes/*
#   runs.py          — /api/runs/*
#   chat.py          — /api/chat/*
#   precheck_lint.py — /api/precheck-history/* & /api/lint-reports/*
#   snapshot.py      — /api/simulations/{id}/snapshot
#   solvers.py       — /api/solvers/* (read-only registry info)
