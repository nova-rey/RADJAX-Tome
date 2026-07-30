from __future__ import annotations

from copy import deepcopy

import pytest

from radjax_tome.tome.contracts import (
    CANONICAL_TOME_COVER_SCHEMA,
    PackageInventoryEntry,
    TrainingPayloadEntry,
    build_canonical_content_manifest,
    build_canonical_tome_cover,
    build_tome_semantic_identity,
    compare_canonical_tome_identities,
    validate_canonical_tome_cover,
)


def _identity():
    return build_tome_semantic_identity(
        training_payload=(
            TrainingPayloadEntry("corridor-modes", "sha256:" + "1" * 64),
            TrainingPayloadEntry("selected-payloads", "sha256:" + "2" * 64),
        ),
        training_contract={
            "target_policy": "corridor_exemplar_v1",
            "sequence_length": 128,
            "vocab_size": 262144,
        },
        authority={
            "selection_integration_config_hash": "sha256:" + "3" * 64,
            "contract_version": "selection_authority_v1",
        },
    )


def _inventory(*entries: PackageInventoryEntry) -> tuple[PackageInventoryEntry, ...]:
    return entries


def test_m5b_profiles_have_distinct_manifests_but_one_tome_identity() -> None:
    identity = _identity()
    student = build_canonical_content_manifest(
        profile="student",
        semantic_identity=identity,
        inventory=_inventory(
            PackageInventoryEntry(
                "corridors/corridor_modes.json",
                "sha256:" + "4" * 64,
                10,
                "training_critical",
                True,
            ),
        ),
    )
    debug = build_canonical_content_manifest(
        profile="full_debug_provenance",
        semantic_identity=identity,
        inventory=_inventory(
            PackageInventoryEntry(
                "corridors/corridor_modes.json",
                "sha256:" + "4" * 64,
                10,
                "training_critical",
                True,
            ),
            PackageInventoryEntry(
                "reports/debug-provenance.json",
                "sha256:" + "5" * 64,
                20,
                "diagnostic",
                False,
            ),
        ),
    )

    student_cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=student,
        provenance={"producer": "test"},
        validation={"status": "pass"},
    )
    debug_cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=debug,
        provenance={"producer": "test", "debug_receipts": True},
        validation={"status": "pass"},
        transport="tgz",
    )

    assert student.manifest_digest != debug.manifest_digest
    assert (
        student_cover["identity"]["semantic_digest"]
        == debug_cover["identity"]["semantic_digest"]
    )
    assert compare_canonical_tome_identities(
        student_cover["identity"], debug_cover["identity"]
    )
    assert student_cover["package"]["transport"] != debug_cover["package"]["transport"]


def test_m5b_identity_rejects_runtime_only_metadata_instead_of_normalizing_it() -> None:
    with pytest.raises(ValueError, match="runtime-only key created_at"):
        build_tome_semantic_identity(
            training_payload=(
                TrainingPayloadEntry("corridor-modes", "sha256:" + "1" * 64),
            ),
            training_contract={"created_at": "2026-07-30T00:00:00Z"},
            authority={"selection": "sha256:" + "2" * 64},
        )


def test_m5b_authority_bearing_training_semantics_change_identity() -> None:
    identity = _identity()
    changed = build_tome_semantic_identity(
        training_payload=identity.training_payload,
        training_contract=identity.training_contract,
        authority={
            **identity.authority,
            "selection_integration_config_hash": "sha256:" + "7" * 64,
        },
    )

    assert changed.semantic_digest != identity.semantic_digest


def test_m5b_cover_is_nested_and_rejects_circular_manifest_inventory() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="exclude cover_page.json"):
        build_canonical_content_manifest(
            profile="student",
            semantic_identity=identity,
            inventory=_inventory(
                PackageInventoryEntry(
                    "cover_page.json",
                    "sha256:" + "6" * 64,
                    1,
                    "integrity_or_provenance",
                    False,
                ),
            ),
        )

    manifest = build_canonical_content_manifest(
        profile="student",
        semantic_identity=identity,
        inventory=(),
    )
    cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=manifest,
        provenance={"producer": "test"},
        validation={"status": "pass"},
    )

    assert cover["schema_version"] == CANONICAL_TOME_COVER_SCHEMA
    assert set(cover) == {
        "schema_version",
        "identity",
        "training",
        "package",
        "manifests",
        "authority",
        "provenance",
        "validation",
    }
    validate_canonical_tome_cover(cover)
    malformed = deepcopy(cover)
    malformed["manifests"]["content"]["profile"] = "full_debug_provenance"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_canonical_tome_cover(malformed)


def test_m5b_identity_comparisons_fail_closed_across_contract_versions() -> None:
    identity = _identity().to_dict()
    historical = {**identity, "schema_version": "radjax_tome_semantic_identity_v0"}

    with pytest.raises(ValueError, match="unsupported"):
        compare_canonical_tome_identities(identity, historical)
