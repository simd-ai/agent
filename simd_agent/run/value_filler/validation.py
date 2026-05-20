# simd_agent/run/value_filler/validation.py
"""Response parsing + structural sanity check.

Kept in its own module so that:

  * Markdown stripping evolves independently of prompt building.
  * The "did the LLM drop a patch?" check has a single, testable home.

Both functions are pure and have no LLM / IO dependencies.
"""

from __future__ import annotations

import re
from typing import Any


_NOISY_PREFIX = re.compile(r"^\s*(?:```(?:[a-zA-Z]+)?\s*)?", re.MULTILINE)
_NOISY_SUFFIX = re.compile(r"\s*```\s*$")


def extract_file_body(response: Any) -> str | None:
    """Pull the text from a Gemini response and strip markdown noise.

    Returns ``None`` if no usable text is found.  Strips a single
    leading code fence (``` or ```cpp etc.) and a single trailing one
    — leaving any internal fences alone (in case the file legitimately
    contains backticks, though OpenFOAM dicts don't).
    """
    text: str | None = None
    for candidate in (getattr(response, "candidates", None) or []):
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                text = t if text is None else text + t
    if text is None:
        text = getattr(response, "text", None)
    if not isinstance(text, str):
        return None
    cleaned = _NOISY_PREFIX.sub("", text, count=1)
    cleaned = _NOISY_SUFFIX.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def looks_structurally_sound(new_content: str, original: str) -> bool:
    """Cheap sanity check: every patch in ``original`` appears in ``new_content``.

    Catches the most common LLM failure modes:

      * Returned a fragment instead of the full file (no ``FoamFile``
        header or no ``boundaryField`` block).
      * Dropped a patch entry while "summarising" the file.
      * Invented a new patch the mesh doesn't have.

    Patch detection is regex-based against the original — anything
    that's an OpenFOAM-style identifier at the second indent level
    inside ``boundaryField`` is treated as a patch name.
    """
    if "FoamFile" not in new_content or "boundaryField" not in new_content:
        return False
    patch_block_names = re.findall(
        r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*$", original, re.MULTILINE,
    )
    for name in patch_block_names:
        if re.search(
            rf"^\s{{4}}{re.escape(name)}\s*$", new_content, re.MULTILINE,
        ) is None:
            return False
    return True
