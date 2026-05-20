# simd_agent/run/value_filler/filler.py
"""Public entry point — routes target files through the LLM rewrite pipeline.

The filler runs ONCE per generation iteration, after ``validate_full()``
in the orchestrator.  For each target file (``0/<T|U|p|p_rgh>`` for
single-region, ``0/<region>/<T|U|p|p_rgh>`` for multi-region) it:

  1. Builds the per-file context (see :mod:`contexts`).
  2. Builds the per-file prompt (see :mod:`prompts`).
  3. Calls the LLM in parallel via :func:`asyncio.gather`.
  4. Validates each response structurally (see :mod:`validation`).
  5. Replaces the file body when the rewrite is accepted; keeps the
     original deterministic / LLM-generated template otherwise.

Failure semantics
-----------------
Per-file failures (network, malformed response, structural drift) are
**silent** — the file keeps its original content and the rest of the
files in the batch still get rewritten.  This is the safety-net role
the filler plays: when in doubt, do nothing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from simd_agent.llm import get_provider
from simd_agent.run.value_filler import contexts, prompts, validation

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


async def fill_field_values(
    files: dict[str, str],
    config: dict[str, Any],
    user_requirements: str,
) -> dict[str, str]:
    """Run the value-fill LLM pass on every supported target file.

    Args:
        files: Generated case files (path → content).  NOT mutated —
            the function returns a new dict.
        config: Validated case config.  Must carry ``mesh.patches``
            and optionally ``regions`` (CHT) + ``case_defaults`` (set
            by the enrichment pipeline).
        user_requirements: The user's natural-language prompt.

    Returns:
        A new files dict where each successfully-rewritten target has
        been replaced.  Files that aren't filler targets, files whose
        rewrite failed validation, and files for which the LLM
        returned byte-identical content are returned unchanged.
    """
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for path, content in files.items():
        ctx = contexts.build_for_path(path, config)
        if ctx is not None:
            candidates.append((path, content, ctx))

    if not candidates:
        return dict(files)

    provider = get_provider()
    model = provider.models.get("super", provider.models["default"])
    client = provider.client

    logger.info(
        "[LLM_FILL] starting on %d files: %s",
        len(candidates), [p for p, _, _ in candidates],
    )
    logger.info(
        "[LLM_FILL][user_prompt] BEGIN ------------------------------------\n%s"
        "\n[LLM_FILL][user_prompt] END --------------------------------------",
        user_requirements.strip() or "(empty)",
    )
    # One-line context summary per file — the prompt itself is verbose,
    # but a "what did the filler know about this file?" log is the
    # single most useful diagnostic when a value goes wrong.
    for path, _, ctx in candidates:
        logger.info("[LLM_FILL][ctx] %s: %s", path, _ctx_summary(ctx))

    rewrites = await asyncio.gather(*[
        _rewrite_one(client, model, path, content, ctx, user_requirements)
        for path, content, ctx in candidates
    ])

    out = dict(files)
    for path, new_content in zip([c[0] for c in candidates], rewrites):
        if new_content is not None:
            out[path] = new_content
    return out


# ────────────────────────────────────────────────────────────────────────────
# Per-file LLM call
# ────────────────────────────────────────────────────────────────────────────


async def _rewrite_one(
    client: Any,
    model: str,
    path: str,
    content: str,
    ctx: dict[str, Any],
    user_requirements: str,
) -> str | None:
    """LLM-rewrite one field file.  Returns ``None`` on any failure.

    Logged at INFO so the prompt + response are visible by default;
    bracketed tags (``[LLM_FILL][prompt]``, ``[LLM_FILL][response]``,
    ``[LLM_FILL][result]``) make the log greppable.
    """
    prompt = prompts.build_prompt(path, content, ctx, user_requirements)
    logger.info(
        "[LLM_FILL][prompt] %s: BEGIN ------------------------------------\n%s"
        "\n[LLM_FILL][prompt] %s: END --------------------------------------",
        path, prompt, path,
    )

    try:
        response = await client.aio.models.generate_content(
            model=model, contents=prompt,
        )
    except Exception as exc:
        logger.warning(
            "[LLM_FILL][result] %s: LLM call failed (%s); keeping template",
            path, exc,
        )
        return None

    new_content = validation.extract_file_body(response)
    logger.info(
        "[LLM_FILL][response] %s: BEGIN ------------------------------------\n%s"
        "\n[LLM_FILL][response] %s: END --------------------------------------",
        path,
        new_content if new_content else "(empty / non-text response)",
        path,
    )

    if not new_content:
        logger.warning("[LLM_FILL][result] %s: empty response; keeping template", path)
        return None

    if not validation.looks_structurally_sound(new_content, content):
        logger.warning(
            "[LLM_FILL][result] %s: structural sanity check FAILED "
            "(missing FoamFile/boundaryField or a patch entry was dropped); "
            "keeping template", path,
        )
        return None

    if new_content == content:
        # The LLM agreed with the template — nothing to substitute.
        # Common case for single-region cases where the initial codegen
        # already had every value right.
        logger.info(
            "[LLM_FILL][result] %s: response is byte-identical to template — "
            "no rewrite needed", path,
        )
        return None

    logger.info(
        "[LLM_FILL][result] %s: ACCEPTED (%d → %d chars)",
        path, len(content), len(new_content),
    )
    return new_content


# ────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────


def _ctx_summary(ctx: dict[str, Any]) -> str:
    """One-line summary of the per-file context, for the log header."""
    field = ctx.get("field")
    n_patches = len(ctx.get("patches") or [])
    if ctx.get("mode") == "multi":
        return (
            f"mode=multi field={field} region={ctx.get('name')!r} "
            f"kind={ctx.get('kind')!r} patches={n_patches} "
            f"T_init={ctx.get('T_init')} U_init={ctx.get('U_init')} "
            f"p_init={ctx.get('p_init')}"
        )
    return (
        f"mode=single field={field} fluid={ctx.get('fluid_name')!r} "
        f"patches={n_patches} "
        f"case_defaults_keys={sorted((ctx.get('case_defaults') or {}).keys())}"
    )
