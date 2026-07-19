"""Recommended public fingerprint API.

Advanced fingerprint types and utilities remain available from their defining
submodules, such as ``radjax_tome.fingerprint.artifacts`` and
``radjax_tome.fingerprint.generation``.
"""

from __future__ import annotations

from typing import Any

from radjax_tome._lazy_exports import (
    LazyExportMap,
    lazy_export_names,
    resolve_lazy_export,
)

_LAZY_EXPORTS: LazyExportMap = {
    "FingerprintValidationResult": (
        "radjax_tome.fingerprint.artifacts",
        "FingerprintValidationResult",
    ),
    "validate_fingerprint_artifact": (
        "radjax_tome.fingerprint.artifacts",
        "validate_fingerprint_artifact",
    ),
    "summarize_exemplar_reservoir": (
        "radjax_tome.fingerprint.exemplars",
        "summarize_exemplar_reservoir",
    ),
    "build_minimal_fingerprint_artifact_from_target_store": (
        "radjax_tome.fingerprint.generation",
        "build_minimal_fingerprint_artifact_from_target_store",
    ),
    "generate_corridor_measurement_report": (
        "radjax_tome.fingerprint.generation",
        "generate_corridor_measurement_report",
    ),
    "generate_corridor_subset_receipt": (
        "radjax_tome.fingerprint.generation",
        "generate_corridor_subset_receipt",
    ),
    "generate_exemplar_reservoir": (
        "radjax_tome.fingerprint.generation",
        "generate_exemplar_reservoir",
    ),
    "inspect_fingerprint_artifact": (
        "radjax_tome.fingerprint.inspection",
        "inspect_fingerprint_artifact",
    ),
}

__all__ = [
    "FingerprintValidationResult",
    "build_minimal_fingerprint_artifact_from_target_store",
    "generate_corridor_measurement_report",
    "generate_corridor_subset_receipt",
    "generate_exemplar_reservoir",
    "inspect_fingerprint_artifact",
    "summarize_exemplar_reservoir",
    "validate_fingerprint_artifact",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return lazy_export_names(globals(), _LAZY_EXPORTS)
