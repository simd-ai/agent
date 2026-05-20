"""Allow ``python -m simd_agent.cli`` as an alias for the ``simd`` script."""

from simd_agent.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
