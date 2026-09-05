"""Read-only projections of owner-written production and corpus progress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_production_progress(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        return {"status": "missing", "path": str(target)}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(target), "error": str(exc)}
    if not isinstance(value, dict):
        return {"status": "invalid", "path": str(target), "error": "not an object"}
    return {"status": "present", "path": str(target), "progress": value}


def read_corpus_journal(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        return {"status": "missing", "path": str(target)}
    latest: dict[str, Any] | None = None
    count = 0
    try:
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("journal event is not an object")
                latest = event
                count += 1
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(target), "error": str(exc)}
    return {
        "status": "present",
        "path": str(target),
        "event_count": count,
        "latest": latest,
    }


__all__ = ["read_corpus_journal", "read_production_progress"]
