"""Explicit validated source artifact handoff for package materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.canonical_artifact import (
    derive_tome_semantic_identity,
    validate_canonical_artifact_directory,
)
from radjax_tome.tome.contracts import (
    CANONICAL_TOME_COVER_SCHEMA,
    CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA,
    CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA,
    CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA,
    CANONICAL_TOME_STUDENT_CONSUMPTION_V5_COVER_SCHEMA,
    CanonicalContentManifest,
    PackageInventoryEntry,
    TomeSemanticIdentity,
    TrainingPayloadEntry,
    validate_canonical_tome_cover,
)
from radjax_tome.tome.producer_validation import validate_full_debug_producer


@dataclass(frozen=True)
class ValidatedProducerArtifact:
    """Validated pre-package source accepted by the producer path adapter.

    The source root and its training-authoritative semantic projection are
    checked before materialization.  A package is deliberately permitted to
    start from an older or subsequently augmented producer directory because
    packaging builds and validates its own profile-specific v3 inventory.
    """

    root: Path
    semantic_identity: TomeSemanticIdentity
    validation_evidence: dict[str, Any]
    authority_references: dict[str, Any]

    @classmethod
    def from_directory(cls, artifact_dir: Path) -> ValidatedProducerArtifact:
        root = artifact_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"artifact directory does not exist: {root}")
        status, blockers = validate_full_debug_producer(root)
        if status == "fail":
            raise ValueError(
                "producer artifact validation failed: " + "; ".join(blockers)
            )
        identity = derive_tome_semantic_identity(root)
        validation = _read_optional_object(root / "validation_report.json")
        production = _read_optional_object(root / "production_build_report.json")
        return cls(
            root=root,
            semantic_identity=identity,
            validation_evidence=validation,
            authority_references={
                key: production.get(key)
                for key in (
                    "selection_integration_config_hash",
                    "score_pass_authority_hash",
                    "authority_hash_contract",
                )
            },
        )


@dataclass(frozen=True)
class ValidatedTomeArtifact:
    """Complete validated canonical-v3 package/inspection descriptor."""

    root: Path
    cover: dict[str, Any]
    semantic_identity: TomeSemanticIdentity
    content_manifest: CanonicalContentManifest
    profile: str
    inventory: tuple[PackageInventoryEntry, ...]
    validation_evidence: dict[str, Any]
    authority_references: dict[str, Any]
    student_consumption: dict[str, Any] | None = None

    @classmethod
    def from_canonical_directory(cls, artifact_dir: Path) -> ValidatedTomeArtifact:
        root = artifact_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"artifact directory does not exist: {root}")
        cover = read_json_object(root / "cover_page.json")
        if cover.get("schema_version") not in {
            CANONICAL_TOME_COVER_SCHEMA,
            CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA,
            CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA,
            CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA,
            CANONICAL_TOME_STUDENT_CONSUMPTION_V5_COVER_SCHEMA,
        }:
            raise ValueError("canonical v3 cover_page.json is required")
        validate_canonical_tome_cover(cover)
        validate_canonical_artifact_directory(root, cover)
        identity_payload = cover["identity"]
        manifest_payload = cover["manifests"]["content"]
        identity = TomeSemanticIdentity(
            training_payload=tuple(
                TrainingPayloadEntry(**entry)
                for entry in identity_payload["training_payload"]
            ),
            training_contract=dict(identity_payload["training_contract"]),
            authority=dict(identity_payload["authority"]),
            semantic_digest=str(identity_payload["semantic_digest"]),
            schema_version=str(identity_payload["schema_version"]),
        )
        inventory = tuple(
            PackageInventoryEntry(**entry) for entry in manifest_payload["inventory"]
        )
        manifest = CanonicalContentManifest(
            profile=str(manifest_payload["profile"]),
            semantic_identity_digest=str(manifest_payload["semantic_identity_digest"]),
            inventory=inventory,
            manifest_digest=str(manifest_payload["manifest_digest"]),
            schema_version=str(manifest_payload["schema_version"]),
        )
        return cls(
            root=root,
            cover=cover,
            semantic_identity=identity,
            content_manifest=manifest,
            profile=manifest.profile,
            inventory=inventory,
            validation_evidence=dict(cover["validation"]),
            authority_references=dict(cover["authority"]),
            student_consumption=(
                dict(cover["student_consumption"])
                if "student_consumption" in cover
                else None
            ),
        )


def _read_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path)
