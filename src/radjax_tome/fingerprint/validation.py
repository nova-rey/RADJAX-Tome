from __future__ import annotations

from pathlib import Path

from radjax_tome.fingerprint.artifacts import (
    FingerprintValidationResult,
    validate_fingerprint_artifact,
    validate_fingerprint_byte_accounting,
)


def validate_fingerprint_artifact_path(path: str | Path) -> FingerprintValidationResult:
    return validate_fingerprint_artifact(path)


__all__ = [
    "FingerprintValidationResult",
    "validate_fingerprint_artifact",
    "validate_fingerprint_artifact_path",
    "validate_fingerprint_byte_accounting",
]
