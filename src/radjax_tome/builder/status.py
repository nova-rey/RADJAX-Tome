"""Read-only projections of owner-written production and corpus progress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import canonical_bytes, sha256


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
    previous_hash: str | None = None
    try:
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("journal event is not an object")
                if event.get("sequence") != count:
                    raise ValueError("journal sequence is invalid")
                supplied = event.pop("event_hash", None)
                if supplied != sha256(canonical_bytes(event)):
                    raise ValueError("journal event hash is invalid")
                if event.get("previous_event_hash") != previous_hash:
                    raise ValueError("journal predecessor is invalid")
                event["event_hash"] = supplied
                latest = event
                previous_hash = str(supplied)
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
