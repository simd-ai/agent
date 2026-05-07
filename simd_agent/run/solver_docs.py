"""solver_docs.py — Prompt pack loader and required-file resolver.

Thin façade over the plugin-centric architecture.  All solver prompts and
required file manifests now live in self-contained plugin packages under
``simd_agent/solvers/<name>/``.  This module is kept for backward compatibility
with the API surface used by the frontend (``PromptPack``, ``load_prompt_pack``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Universal codegen rules still live in the shared pack (not per-solver).
_PACK_DIR = Path(__file__).resolve().parent.parent / "prompts" / "packs" / "simd"


@dataclass(frozen=True)
class PromptPack:
    """Everything the UI / API needs to know about the chosen solver's prompt."""

    solver: str
    # Relative paths of the prompt files that will be concatenated for codegen.
    prompt_files: list[str]
    # Content of each prompt file keyed by its relative path.
    prompts: dict[str, str]
    # Deterministic list of OpenFOAM case files that will be generated.
    required_case_files: list[str]


def list_available_solvers() -> list[str]:
    """Return names of registered solver plugins."""
    from simd_agent.solvers import get_registry
    return get_registry().names()


def load_solver_base(solver: str) -> str:
    """Load the solver-identity doc (``_solver.md``) via the registry."""
    from simd_agent.solvers import get_registry
    plugin = get_registry().get(solver)
    return plugin.system_prompt() if plugin else ""


def load_solver_file_doc(solver: str, file_path: str) -> str:
    """Load a per-file solver doc via the registry.

    ``file_path`` is an OpenFOAM case path (``system/fvSchemes``, ``0/U``),
    not a ``.md`` relative path — the plugin handles the mapping.
    """
    from simd_agent.solvers import get_registry
    plugin = get_registry().get(solver)
    return plugin.prompt_for_file(file_path) if plugin else ""


def load_prompt_pack(
    solver: str,
    validated_config: dict[str, Any] | None = None,
) -> PromptPack:
    """Load the prompt pack for *solver* and compute its required case-file list.

    The pack is assembled from:
      1. Universal ``codegen.md`` rules (shared across solvers).
      2. The plugin's ``_solver.md`` identity doc.
      3. Every per-file doc referenced by ``plugin.required_files(config)``.

    Args:
        solver:           OpenFOAM solver name (e.g. ``"rhoPimpleFoam"``).
        validated_config: Linted/validated simulation config dict.

    Returns:
        PromptPack with prompt file names, their contents, and required_case_files.
    """
    from simd_agent.solvers import get_registry
    from simd_agent.solvers.base import SolverPlugin

    validated_config = validated_config or {}

    prompts: dict[str, str] = {}
    prompt_files: list[str] = []

    # 1. Universal codegen rules — always first.
    codegen_md = _PACK_DIR / "codegen.md"
    if codegen_md.exists():
        prompt_files.append("codegen.md")
        prompts["codegen.md"] = codegen_md.read_text(encoding="utf-8")

    plugin: SolverPlugin | None = get_registry().get(solver)

    if plugin is None:
        return PromptPack(
            solver=solver,
            prompt_files=prompt_files,
            prompts=prompts,
            required_case_files=[],
        )

    # 2. Plugin identity doc.
    solver_md = plugin.prompts_dir / "_solver.md"
    if solver_md.exists():
        rel = f"solvers/{solver}/_solver.md"
        prompt_files.append(rel)
        prompts[rel] = solver_md.read_text(encoding="utf-8")

    # 3. Required case files + per-file prompt docs.
    required = plugin.required_files(validated_config)
    for case_file in required:
        doc = plugin.prompt_for_file(case_file)
        if not doc:
            continue
        doc_relpath = SolverPlugin._file_doc_relpath(case_file)
        rel = f"solvers/{solver}/{doc_relpath}"
        if rel in prompts:
            continue  # dedupe (e.g. multiple alpha.* fields share one doc)
        prompt_files.append(rel)
        prompts[rel] = doc

    return PromptPack(
        solver=solver,
        prompt_files=prompt_files,
        prompts=prompts,
        required_case_files=required,
    )
