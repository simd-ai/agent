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

    # ``questionary.ask()`` internally calls ``asyncio.run()`` which
    # fails when we're already inside a running event loop (the case
    # here — ``simd run`` is async at the top level).  Run questionary
    # on a worker thread so it gets its own loop; the calling thread
    # blocks on ``join()`` so this is still synchronous from the
    # caller's POV.
    import threading

    result: list[object] = [None]
    exc: list[BaseException | None] = [None]

    def _run() -> None:
        try:
            result[0] = _ask_questionary(prompt, options)
        except BaseException as e:  # noqa: BLE001 — propagate to main thread
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()

    if exc[0] is not None:
        raise exc[0]
    answer = result[0]
    if answer is None:
        raise KeyboardInterrupt()
    return int(answer)  # type: ignore[arg-type]


def _ask_questionary(prompt: str, options: Sequence[str]) -> int | None:
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
    return questionary.select(
        prompt,
        choices=choices,
        use_indicator=True,
        instruction="(↑/↓ to move, Enter to select)",
        qmark="?",
        style=style,
    ).ask()


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
