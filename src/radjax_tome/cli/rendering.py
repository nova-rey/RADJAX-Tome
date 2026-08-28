"""Public CLI stream and rendering policy."""

from __future__ import annotations

import json
import sys

from .models import CLIResult


def emit(result: CLIResult, *, machine: bool, quiet: bool = False) -> None:
    if machine:
        print(json.dumps(result.to_dict(), sort_keys=True, default=str))
        return
    if result.error is not None:
        print(f"ERROR {result.error.code}: {result.error.message}", file=sys.stderr)
        if result.error.repair:
            print(f"repair: {result.error.repair}", file=sys.stderr)
    elif not quiet:
        print(f"{result.command}: {result.status}")
        if result.artifact:
            for key, value in result.artifact.items():
                if value is not None:
                    print(f"{key}: {value}")
