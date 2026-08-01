"""Tome-local admission for the closed native-v3 Student-consumption v2 cover."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from radjax_tome.tome.artifact_descriptor import ValidatedTomeArtifact
from radjax_tome.tome.contracts import (
    CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA,
    PackageInventoryEntry,
    TrainingPayloadEntry,
    build_canonical_content_manifest,
    build_canonical_tome_cover,
    build_tome_semantic_identity,
    canonical_json_digest,
    validate_canonical_tome_cover,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _base_identity():
    return build_tome_semantic_identity(
        training_payload=(
            TrainingPayloadEntry("legacy-targets", "sha256:" + "1" * 64),
        ),
        training_contract={"sequence_length": 4, "vocab_size": 32},
        authority={"selection_integration_config_hash": "sha256:" + "2" * 64},
    )


def _v2_cover(root: Path) -> dict[str, object]:
    sidecar_path = root / "manifests" / "student_consumption_v2.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    sidecar_bytes = sidecar_path.read_bytes()
    identity = _base_identity()
    manifest = build_canonical_content_manifest(
        profile="student",
        semantic_identity=identity,
        inventory=(
            PackageInventoryEntry(
                "manifests/student_consumption_v2.json",
                _sha256(sidecar_bytes),
                len(sidecar_bytes),
                "integrity_or_provenance",
                False,
            ),
        ),
    )
    cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=manifest,
        provenance={"producer": "test"},
        validation={"status": "pass"},
    )
    cover["schema_version"] = CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA
    cover["student_consumption"] = {
        "profile_id": "native_v3_student_v2",
        "manifest_path": "manifests/student_consumption_v2.json",
        "manifest_sha256": _sha256(sidecar_bytes),
        "semantic_digest": canonical_json_digest({"derived": "test"}),
        "digest_method": "sha256",
        "required_capabilities": [],
    }
    return cover


def test_v2_cover_declaration_is_closed_and_preserves_base_identity(
    tmp_path: Path,
) -> None:
    cover = _v2_cover(tmp_path)
    base_identity = deepcopy(cover["identity"])

    validate_canonical_tome_cover(cover)

    assert cover["identity"] == base_identity
    assert (
        cover["manifests"]["content"]["semantic_identity_digest"]
        == base_identity["semantic_digest"]
    )

    unexpected = deepcopy(cover)
    unexpected["student_consumption"]["unexpected"] = True
    with pytest.raises(ValueError, match="allowed keys: unexpected unexpected"):
        validate_canonical_tome_cover(unexpected)

    nonempty_capability = deepcopy(cover)
    nonempty_capability["student_consumption"]["required_capabilities"] = ["future"]
    with pytest.raises(ValueError, match="required_capabilities must be empty"):
        validate_canonical_tome_cover(nonempty_capability)


def test_validated_artifact_descriptor_admits_v2_cover_without_reinterpreting_v3(
    tmp_path: Path,
) -> None:
    cover = _v2_cover(tmp_path)
    (tmp_path / "cover_page.json").write_text(
        json.dumps(cover, sort_keys=True), encoding="utf-8"
    )

    descriptor = ValidatedTomeArtifact.from_canonical_directory(tmp_path)

    assert (
        descriptor.semantic_identity.semantic_digest
        == cover["identity"]["semantic_digest"]
    )
    assert descriptor.student_consumption == cover["student_consumption"]


def test_plain_v3_cover_remains_accepted_without_student_consumption(
    tmp_path: Path,
) -> None:
    cover = _v2_cover(tmp_path)
    cover["schema_version"] = "radjax_tome_cover_v3"
    del cover["student_consumption"]

    validate_canonical_tome_cover(cover)
