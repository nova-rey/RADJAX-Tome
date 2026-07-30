"""Pure M5D derivation of a v3 cover from an artifact directory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.contracts import (
    PackageInventoryEntry,
    TomeSemanticIdentity,
    TrainingPayloadEntry,
    build_canonical_content_manifest,
    build_canonical_tome_cover,
    build_tome_semantic_identity,
    canonical_json_digest,
    validate_canonical_tome_cover,
)

_COVER = "cover_page.json"
_SEMANTIC_JSON_FILES = (
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "corridors/corridor_summary.json",
    "corridors/corridor_modes.json",
    "corridors/mode_assignments.json",
    "leaderboards/selected_exemplars.json",
)
_RUNTIME_KEYS = frozenset({"created_at", "completed_at", "updated_at"})


def derive_tome_semantic_identity(root: Path) -> TomeSemanticIdentity:
    """Derive profile-independent identity from the source semantic payload.

    This happens before profile materialization.  Package-specific debug
    receipts, student portability rewrites, cover contents, and transport
    wrapping never enter this identity.
    """

    metadata = _read_required(root, "metadata.json")
    payload = tuple(
        TrainingPayloadEntry(relative, _semantic_file_digest(root / relative))
        for relative in _SEMANTIC_JSON_FILES
        if (root / relative).is_file()
    )
    if not payload:
        raise ValueError("canonical Tome identity requires semantic payload files")
    production = _read_optional(root, "production_build_report.json") or {}
    return build_tome_semantic_identity(
        training_payload=payload,
        training_contract={
            "target_type": metadata.get("target_type"),
            "sequence_length": metadata.get("sequence_length"),
            "vocab_size": metadata.get("vocab_size"),
            "tome_version": metadata.get("tome_version"),
        },
        authority={
            "selection_integration_config_hash": production.get(
                "selection_integration_config_hash"
            ),
            "score_pass_authority_hash": production.get("score_pass_authority_hash"),
            "authority_hash_contract": production.get("authority_hash_contract"),
        },
    )


def build_canonical_artifact_cover(
    root: Path,
    *,
    profile: str,
    transport: str,
    semantic_identity: TomeSemanticIdentity | None = None,
) -> dict[str, Any]:
    """Build a v3 cover and complete raw inventory without mutating ``root``."""

    identity = semantic_identity or derive_tome_semantic_identity(root)
    manifest = build_canonical_content_manifest(
        profile=profile,
        semantic_identity=identity,
        inventory=_inventory(root),
    )
    validation = _read_optional(root, "validation_report.json") or {}
    cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=manifest,
        provenance={
            "raw_artifact_digests": _raw_integrity_digests(root),
            "legacy_cover_schema": _legacy_cover_schema(root),
        },
        validation={
            "status": validation.get("status"),
            "validation_report_path": "validation_report.json",
        },
        transport=transport,
    )
    validate_canonical_tome_cover(cover)
    return cover


def validate_canonical_artifact_directory(root: Path, cover: dict[str, Any]) -> None:
    """Prove a v3 cover's profile inventory exactly binds this directory."""

    validate_canonical_tome_cover(cover)
    expected = cover["manifests"]["content"]["inventory"]
    observed = [entry.to_dict() for entry in _inventory(root)]
    if expected != observed:
        raise ValueError("canonical content inventory does not match directory")


def _inventory(root: Path) -> tuple[PackageInventoryEntry, ...]:
    entries: list[PackageInventoryEntry] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == _COVER:
            continue
        entries.append(
            PackageInventoryEntry(
                path=relative,
                sha256=_raw_digest(path),
                size_bytes=path.stat().st_size,
                classification=_classification(relative),
                training_authoritative=relative in _SEMANTIC_JSON_FILES,
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _classification(relative: str) -> str:
    if relative in _SEMANTIC_JSON_FILES or relative.startswith("shards/"):
        return "training_critical"
    if relative.startswith(("reports/", "leaderboards/")):
        return "diagnostic"
    if relative.startswith("manifests/"):
        return "integrity_or_provenance"
    return "operational"


def _raw_integrity_digests(root: Path) -> dict[str, str | None]:
    return {
        relative: _raw_digest(root / relative) if (root / relative).is_file() else None
        for relative in (
            "metadata.json",
            "corridors/mode_assignments.json",
            "corridors/corridor_modes.json",
            "c6/production_global_selector.json",
        )
    }


def _semantic_file_digest(path: Path) -> str:
    return canonical_json_digest(_strip_runtime(_read_json(path)))


def _strip_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime(nested)
            for key, nested in value.items()
            if key not in _RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [_strip_runtime(item) for item in value]
    return value


def _legacy_cover_schema(root: Path) -> str | None:
    cover = _read_optional(root, _COVER)
    if cover is None:
        return None
    value = cover.get("schema_version")
    if isinstance(value, str):
        return value
    version = cover.get("cover_page_version")
    return f"cover_page_v{version}" if isinstance(version, int) else None


def _read_required(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing required canonical source file: {relative}")
    return _read_json(path)


def _read_optional(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    return _read_json(path) if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _raw_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
