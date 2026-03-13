"""solver_docs.py — Prompt pack loader and required-file resolver.

Provides a thin, import-safe layer between orchestration and the prompt files on disk.
All heavy logic (required file computation) delegates to the single source of truth
in genai_codegen.build_required_files_list so there is no duplication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Prompt directory layout mirrors genai_codegen.py constants.
_PACK_DIR   = Path(__file__).resolve().parent / "prompts" / "packs" / "simd"
_SOLVERS_DIR = _PACK_DIR / "solvers"


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
    """Return names of solvers that have a dedicated .md prompt file."""
    if not _SOLVERS_DIR.exists():
        return []
    return sorted(p.stem for p in _SOLVERS_DIR.glob("*.md"))


def load_solver_base(solver: str) -> str:
    """Load solver-identity doc (_solver.md), falling back to legacy monolithic file."""
    base = _SOLVERS_DIR / solver / "_solver.md"
    if base.exists():
        return base.read_text(encoding="utf-8")
    legacy = _SOLVERS_DIR / f"{solver}.md"
    return legacy.read_text(encoding="utf-8") if legacy.exists() else ""


def load_solver_file_doc(solver: str, file_doc_relpath: str) -> str:
    """Load a per-file solver doc (e.g. system/fvSchemes.md, fields/U.md).
    Returns empty string if the file doesn't exist — caller should fall back to hints.
    """
    p = _SOLVERS_DIR / solver / file_doc_relpath
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def load_prompt_pack(
    solver: str,
    validated_config: dict[str, Any] | None = None,
) -> PromptPack:
    """Load the prompt pack for *solver* and compute its required case-file list.

    Delegates required-file computation to ``genai_codegen.build_required_files_list``
    which is the single source of truth used by the codegen prompt itself.

    Args:
        solver:           OpenFOAM solver name (e.g. "rhoPimpleFoam").
        validated_config: Linted/validated simulation config dict.  Used to
                          resolve turbulence model, heat transfer flag, etc.

    Returns:
        PromptPack with prompt file names, their contents, and required_case_files.
    """
    validated_config = validated_config or {}

    prompts: dict[str, str] = {}
    prompt_files: list[str] = []

    def _try_read(path: Path, rel_base: Path) -> None:
        if path.exists():
            rel = str(path.relative_to(rel_base))
            prompt_files.append(rel)
            prompts[rel] = path.read_text(encoding="utf-8")

    # 1. Base codegen rules (always first)
    _try_read(_PACK_DIR / "codegen.md", _PACK_DIR)

    # 2. Solver-specific instructions (appended after base)
    _try_read(_SOLVERS_DIR / f"{solver}.md", _PACK_DIR)

    # 3. Required case files — delegate to the canonical implementation
    #    Import lazily to avoid circular imports (genai_codegen imports solver_selector,
    #    which is fine, but we want this module to be importable standalone too).
    from simd_agent.run.genai_codegen import build_required_files_list
    required = build_required_files_list(solver, validated_config)

    return PromptPack(
        solver=solver,
        prompt_files=prompt_files,
        prompts=prompts,
        required_case_files=required,
    )
