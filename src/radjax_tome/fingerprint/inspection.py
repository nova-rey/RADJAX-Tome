from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_tome.fingerprint.artifacts import summarize_fingerprint_artifact
from radjax_tome.fingerprint.exemplars import summarize_exemplar_reservoir
from radjax_tome.fingerprint.summary import build_fingerprint_summary_payload


def inspect_fingerprint_artifact(path: str | Path) -> dict[str, Any]:
    return build_fingerprint_summary_payload(path)


def inspect_fingerprint_exemplars(path: str | Path) -> dict[str, Any]:
    return summarize_exemplar_reservoir(path).to_dict()


__all__ = [
    "inspect_fingerprint_artifact",
    "inspect_fingerprint_exemplars",
    "summarize_fingerprint_artifact",
]
