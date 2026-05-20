"""Shared display helpers for the CLI.

Built on top of ``rich`` so colour output degrades gracefully in dumb
terminals.  Submodules:

  - ``stages``    — the 5-stage progress display for ``simd run``
  - ``patches``   — the interactive patch-review REPL
  - ``summary``   — the final summary block printed at run completion

Keeping the display layer separate from the wire layer (``client.py``)
means we can swap ``rich`` for ``prompt_toolkit`` later without touching
the API calls.
"""

from __future__ import annotations

import sys
from rich.console import Console


# Single console instance — shared by every subcommand.  ``stderr=False``
# means we go to stdout by default; we'll explicitly use
# ``console.print(file=sys.stderr)`` for errors when we need to.
console = Console(highlight=False, soft_wrap=False)


def is_tty() -> bool:
    """True when stdout is a terminal — disables progress animations
    in CI / pipe contexts."""
    return sys.stdout.isatty()


def err(*args, **kwargs) -> None:
    """Print to stderr in red.  Used for fatal errors."""
    Console(stderr=True, highlight=False).print(*args, style="red", **kwargs)
