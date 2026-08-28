"""Compatibility entry point for retained engineering commands."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate unchanged research commands to the legacy parser."""
    from radjax_tome.cli.main import _legacy_main

    return _legacy_main(list(argv or ()))
