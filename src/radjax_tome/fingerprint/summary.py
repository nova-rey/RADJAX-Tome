from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_tome.fingerprint.artifacts import (
    FingerprintArtifactSummary,
    summarize_fingerprint_artifact,
)
from radjax_tome.fingerprint.capture_summary import (
    read_real_teacher_capture_summary,
)
from radjax_tome.fingerprint.exemplars import (
    FingerprintExemplarReservoirSummary,
    summarize_exemplar_reservoir,
)


def build_fingerprint_summary_payload(path: str | Path) -> dict[str, Any]:
    artifact = summarize_fingerprint_artifact(path)
    payload: dict[str, Any] = artifact.to_dict()
    capture_summary_path = Path(path) / "capture_summary.json"
    if capture_summary_path.is_file():
        payload["capture_summary"] = read_real_teacher_capture_summary(
            capture_summary_path
        ).to_dict()
    if artifact.has_exemplars:
        payload["exemplar_reservoir"] = summarize_exemplar_reservoir(path).to_dict()
    return payload


__all__ = [
    "FingerprintArtifactSummary",
    "FingerprintExemplarReservoirSummary",
    "build_fingerprint_summary_payload",
    "summarize_exemplar_reservoir",
    "summarize_fingerprint_artifact",
]
