"""Lazy optional-TUI launcher and honest non-interactive fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from radjax_tome.cli.models import CLIResult, CLIWarning


def tui_fallback_or_run(workflow: str, config: Path | None) -> CLIResult:
    command = (
        f"radjax-tome corpus build --config {config}"
        if workflow == "corpus" and config
        else f"radjax-tome build --config {config}"
        if config
        else f"radjax-tome {workflow} --help"
    )
    if not sys.stdin.isatty() or os.environ.get("TERM") == "dumb":
        return CLIResult(
            "tui",
            "pass",
            0,
            reports={"fallback": True, "command": command},
            warnings=[
                CLIWarning(
                    "NON_INTERACTIVE",
                    f"interactive TUI unavailable; run `{command}` instead",
                )
            ],
        )
    try:
        import textual  # noqa: F401
    except ImportError:
        return CLIResult(
            "tui",
            "pass",
            0,
            reports={"fallback": True, "command": command},
            warnings=[
                CLIWarning(
                    "OPTIONAL_DEPENDENCY_MISSING",
                    f"install radjax-tome[tui], then run `{command}`",
                )
            ],
        )
    from radjax_tome.tui.app import WizardApp

    WizardApp(workflow, config).run()
    return CLIResult("tui", "pass", 0, reports={"fallback": False})


__all__ = ["tui_fallback_or_run"]
