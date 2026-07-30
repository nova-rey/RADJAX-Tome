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
    canonical_json_digest,
    compare_canonical_tome_identities,
    validate_canonical_content_manifest,
    validate_canonical_tome_cover,
    validate_canonical_tome_semantic_identity,
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


def _manifest():
    return build_canonical_content_manifest(
        profile="student",
        semantic_identity=_identity(),
        inventory=_inventory(
            PackageInventoryEntry(
                "corridors/corridor_modes.json",
                "sha256:" + "4" * 64,
                10,
                "training_critical",
                True,
            ),
            PackageInventoryEntry(
                "reports/provenance.json",
                "sha256:" + "5" * 64,
                20,
                "integrity_or_provenance",
                False,
            ),
        ),
    )


def _rehash_identity(identity: dict[str, object]) -> None:
    identity["semantic_digest"] = canonical_json_digest(
        {
            key: identity[key]
            for key in (
                "schema_version",
                "training_payload",
                "training_contract",
                "authority",
            )
        }
    )


def _rehash_manifest(manifest: dict[str, object]) -> None:
    manifest["manifest_digest"] = canonical_json_digest(
        {
            key: manifest[key]
            for key in (
                "schema_version",
                "profile",
                "semantic_identity_digest",
                "inventory",
            )
        }
    )


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


def test_m5b_identity_validator_rejects_stale_digest_before_comparison() -> None:
    original = _identity().to_dict()
    tampered = deepcopy(original)
    tampered["training_contract"]["vocab_size"] = 32

    with pytest.raises(ValueError, match="semantic_digest mismatch"):
        compare_canonical_tome_identities(original, tampered)


@pytest.mark.parametrize(
    ("target", "bad_digest"),
    (
        ("semantic_digest", "sha256:short"),
        ("semantic_digest", "sha256:" + "A" * 64),
        ("payload_digest", "sha256:" + "g" * 64),
    ),
)
def test_m5b_identity_validator_requires_exact_lowercase_sha256(
    target: str,
    bad_digest: str,
) -> None:
    identity = _identity().to_dict()
    if target == "payload_digest":
        identity["training_payload"][0]["semantic_digest"] = bad_digest
    else:
        identity[target] = bad_digest

    with pytest.raises(ValueError, match="64 lowercase hex"):
        validate_canonical_tome_semantic_identity(identity)


def test_m5b_identity_validator_rejects_unsorted_duplicate_and_unknown_shape() -> None:
    unsorted = _identity().to_dict()
    unsorted["training_payload"].reverse()
    _rehash_identity(unsorted)
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        validate_canonical_tome_semantic_identity(unsorted)

    duplicate = _identity().to_dict()
    duplicate["training_payload"].append(
        {
            "logical_id": "selected-payloads",
            "semantic_digest": "sha256:" + "6" * 64,
        }
    )
    _rehash_identity(duplicate)
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        validate_canonical_tome_semantic_identity(duplicate)

    missing = _identity().to_dict()
    del missing["authority"]
    with pytest.raises(ValueError, match="exact keys: missing authority"):
        validate_canonical_tome_semantic_identity(missing)

    unexpected = _identity().to_dict()
    unexpected["unexpected"] = True
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_tome_semantic_identity(unexpected)

    unexpected_entry = _identity().to_dict()
    unexpected_entry["training_payload"][0]["unexpected"] = True
    _rehash_identity(unexpected_entry)
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_tome_semantic_identity(unexpected_entry)


def test_m5b_identity_validator_rejects_invalid_nested_field_types() -> None:
    identity = _identity().to_dict()
    identity["training_payload"][0]["logical_id"] = True
    _rehash_identity(identity)

    with pytest.raises(ValueError, match="logical_id must be non-empty"):
        validate_canonical_tome_semantic_identity(identity)

    identity = _identity().to_dict()
    identity["training_contract"] = []
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_canonical_tome_semantic_identity(identity)

    identity = _identity().to_dict()
    identity["authority"]["created_at"] = "2026-07-30T00:00:00Z"
    _rehash_identity(identity)
    with pytest.raises(ValueError, match="runtime-only key created_at"):
        validate_canonical_tome_semantic_identity(identity)


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


def test_m5b_content_manifest_validator_rejects_noncanonical_inventory() -> None:
    unsorted = _manifest().to_dict()
    unsorted["inventory"].reverse()
    _rehash_manifest(unsorted)
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        validate_canonical_content_manifest(unsorted)

    duplicate = _manifest().to_dict()
    duplicate["inventory"].append(deepcopy(duplicate["inventory"][0]))
    _rehash_manifest(duplicate)
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        validate_canonical_content_manifest(duplicate)

    traversal = _manifest().to_dict()
    traversal["inventory"][0]["path"] = "../escape.json"
    _rehash_manifest(traversal)
    with pytest.raises(ValueError, match="normalized relative path"):
        validate_canonical_content_manifest(traversal)

    cover_page = _manifest().to_dict()
    cover_page["inventory"][0]["path"] = "cover_page.json"
    _rehash_manifest(cover_page)
    with pytest.raises(ValueError, match="exclude cover_page.json"):
        validate_canonical_content_manifest(cover_page)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("sha256", "sha256:short", "64 lowercase hex"),
        ("size_bytes", True, "non-negative integer"),
        ("classification", "unknown", "classification is invalid"),
        ("training_authoritative", 1, "must be boolean"),
    ),
)
def test_m5b_content_manifest_validator_rejects_invalid_inventory_types(
    field: str,
    value: object,
    match: str,
) -> None:
    manifest = _manifest().to_dict()
    manifest["inventory"][0][field] = value
    _rehash_manifest(manifest)

    with pytest.raises(ValueError, match=match):
        validate_canonical_content_manifest(manifest)


def test_m5b_content_manifest_validator_rejects_missing_and_extra_keys() -> None:
    missing = _manifest().to_dict()
    del missing["inventory"][0]["classification"]
    _rehash_manifest(missing)
    with pytest.raises(ValueError, match="exact keys: missing classification"):
        validate_canonical_content_manifest(missing)

    unexpected = _manifest().to_dict()
    unexpected["inventory"][0]["unexpected"] = True
    _rehash_manifest(unexpected)
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_content_manifest(unexpected)

    missing_root = _manifest().to_dict()
    del missing_root["profile"]
    with pytest.raises(ValueError, match="exact keys: missing profile"):
        validate_canonical_content_manifest(missing_root)

    malformed_root_digest = _manifest().to_dict()
    malformed_root_digest["semantic_identity_digest"] = "sha256:not-a-digest"
    with pytest.raises(ValueError, match="64 lowercase hex"):
        validate_canonical_content_manifest(malformed_root_digest)


def test_m5b_content_manifest_validator_rejects_noncanonical_dot_path() -> None:
    manifest = _manifest().to_dict()
    manifest["inventory"][0]["path"] = "."
    _rehash_manifest(manifest)

    with pytest.raises(ValueError, match="normalized relative path"):
        validate_canonical_content_manifest(manifest)


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


def test_m5b_cover_validator_rejects_unexpected_top_level_and_nested_sections() -> None:
    identity = _identity()
    manifest = _manifest()
    cover = build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=manifest,
        provenance={"producer": "test"},
        validation={"status": "pass"},
    )

    extra_top_level = deepcopy(cover)
    extra_top_level["unexpected"] = True
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_tome_cover(extra_top_level)

    extra_package_field = deepcopy(cover)
    extra_package_field["package"]["unexpected"] = True
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_tome_cover(extra_package_field)

    extra_manifest_section = deepcopy(cover)
    extra_manifest_section["manifests"]["unexpected"] = True
    with pytest.raises(ValueError, match="exact keys: unexpected unexpected"):
        validate_canonical_tome_cover(extra_manifest_section)

    missing_validation = deepcopy(cover)
    del missing_validation["validation"]
    with pytest.raises(ValueError, match="exact keys: missing validation"):
        validate_canonical_tome_cover(missing_validation)


def test_m5b_identity_comparisons_fail_closed_across_contract_versions() -> None:
    identity = _identity().to_dict()
    historical = {**identity, "schema_version": "radjax_tome_semantic_identity_v0"}

    with pytest.raises(ValueError, match="unsupported"):
        compare_canonical_tome_identities(identity, historical)
