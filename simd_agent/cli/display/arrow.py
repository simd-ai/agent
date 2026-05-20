"""Arrow-key menu — thin wrapper around ``questionary``.

Earlier we hand-rolled this with termios + ANSI escapes and ran into
the classic raw-mode bug where ``\\n`` no longer translates to
``\\r\\n``, so redraws drifted to the right on every keystroke.
``questionary`` (built on ``prompt_toolkit``) handles all the terminal
state correctly across macOS / Linux / various $TERM values, so we
delegate to it and keep the same ``arrow_choice(prompt, options) -> int``
signature the call sites already use.

The visual style we ask for:

  ? prompt
    option a
  ❯ option b
    option c

Non-TTY stdin (CI pipes, scripted runs) falls back to a numbered
prompt — same behavior as before.
"""

from __future__ import annotations

import sys
from typing import Sequence

from simd_agent.cli.display import console


__all__ = ["arrow_choice"]


def arrow_choice(prompt: str, options: Sequence[str]) -> int:
    """Show an arrow-key menu, return the 0-based index of the choice.

    Raises ``KeyboardInterrupt`` when the user cancels (Ctrl-C, or
    Esc / answering ``None``).  Falls back to a plain numbered prompt
    on non-TTY stdin so the same call site works in scripts and CI.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _numbered_fallback(prompt, options)

    # Import lazily — keeps non-TTY callers (and unit tests) from
    # paying the prompt_toolkit startup cost.
    import questionary
    from questionary import Style

    style = Style([
        ("qmark",       "fg:cyan bold"),
        ("question",    "bold"),
        ("pointer",     "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected",    "fg:cyan bold"),
        ("instruction", "italic"),
    ])

    choices = [
        questionary.Choice(title=opt, value=i)
        for i, opt in enumerate(options)
    ]
    answer = questionary.select(
        prompt,
        choices=choices,
        use_indicator=True,
        instruction="(↑/↓ to move, Enter to select)",
        qmark="?",
        style=style,
    ).ask()

    # ``ask()`` returns None when the user cancels (Ctrl-C / Esc).
    if answer is None:
        raise KeyboardInterrupt()
    return answer


def _numbered_fallback(prompt: str, options: Sequence[str]) -> int:
    """Plain numbered prompt for non-TTY contexts (CI, pipes)."""
    console.print(f"[bold]{prompt}[/]")
    for i, opt in enumerate(options, start=1):
        console.print(f"  {i}) {opt}")
    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            raise KeyboardInterrupt()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        console.print(f"  unknown choice {raw!r}")
