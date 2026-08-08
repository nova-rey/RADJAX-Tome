"""Authority and reproducibility checks for the published P6.U1 producer."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import (
    open_verified_student_jsonl_records_v6,
    open_verified_student_m7_payload_v6,
    validate_and_resolve_student_consumption,
)

from radjax_tome.tome import STUDENT, validate_tome_bundle, validate_tome_package
from radjax_tome.tome.v5_fixture import raw_tree_digest, sha256_file, tree_digest
from radjax_tome.tome.v6_reduced_burn_fixture import (
    CONTRACT_COMMIT,
    CONTRACT_RELEASE,
    EXAMPLE_COUNT,
    FIXTURE_ID,
    PROFILE_ID,
    SELECTED_ID_COUNT,
    VALID_TOKEN_COUNT,
    build_v6_reduced_burn_pair,
    canonical_declared_input_bytes,
    declared_input_digest,
    load_declared_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native_v3_student_v6_reduced_burn"
SPEC = ROOT / "tests/fixtures/p6_u1_reduced_burn_inputs.json"


def test_p6_u1_committed_fixture_is_strictly_admitted_and_qualified() -> None:
    receipt = json.loads((FIXTURE / "FIXTURE_RECEIPT.json").read_text())
    assert receipt["fixture_id"] == FIXTURE_ID
    assert receipt["contract"] == {
        "package": "radjax-contract",
        "release": CONTRACT_RELEASE,
        "commit": CONTRACT_COMMIT,
    }
    assert receipt["qualification"] == {
        "stable_examples": EXAMPLE_COUNT,
        "valid_tokens": VALID_TOKEN_COUNT,
        "selected_example_ids": SELECTED_ID_COUNT,
    }
    directory = FIXTURE / "student"
    archive = FIXTURE / "student.tgz"
    assert validate_tome_package(directory, profile=STUDENT).ok
    assert validate_tome_bundle(archive).ok
    directory_descriptor = _resolve(directory)
    archive_descriptor = _resolve(archive)
    for field in (
        "language_binding_digest",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    ):
        assert getattr(directory_descriptor, field) == getattr(
            archive_descriptor, field
        )
        assert receipt["directory"][field] == getattr(directory_descriptor, field)
        assert receipt["archive"][field] == getattr(archive_descriptor, field)
    assert tree_digest(directory) == receipt["directory"]["tree_digest"]
    assert raw_tree_digest(directory) == receipt["directory"]["raw_tree_digest"]
    assert sha256_file(archive) == receipt["archive"]["raw_sha256"]
    assert (FIXTURE / "producer_artifact.v4.tgz").read_bytes() == (
        directory / "student_consumption/v6/selected_exemplar_payload.m7.tgz"
    ).read_bytes()
    roles = {
        resource.role
        for resource in (
            *directory_descriptor.authority_resources,
            *directory_descriptor.non_authority_resources,
        )
    }
    assert {
        "corridor_assignment",
        "corridor_mode_table",
        "example_registry",
        "selected_exemplar_payload",
        "selected_passport_index",
        "target_shard",
    } <= roles
    assert receipt["contract_validation"] == {
        "archive_strict_v6": "pass",
        "directory_archive_equivalent": True,
        "directory_strict_v6": "pass",
    }


def test_p6_u1_fresh_identical_input_builds_are_byte_identical(tmp_path: Path) -> None:
    spec_path, spec = load_declared_inputs(SPEC)
    first = build_v6_reduced_burn_pair(
        tmp_path / "first", tmp_path / "second", spec_path=spec_path
    )
    first_receipt = json.loads((first.parent / "FIXTURE_RECEIPT.json").read_text())
    second_receipt = json.loads((tmp_path / "second/FIXTURE_RECEIPT.json").read_text())
    first_input_bytes = canonical_declared_input_bytes(spec)
    _second_spec_path, second_spec = load_declared_inputs(SPEC)
    second_input_bytes = canonical_declared_input_bytes(second_spec)
    assert first_input_bytes == second_input_bytes
    assert first_receipt["declared_inputs"][
        "canonical_sha256"
    ] == declared_input_digest(spec)
    assert (
        first_receipt["declared_inputs"]["canonical_sha256"]
        == second_receipt["declared_inputs"]["canonical_sha256"]
    )
    assert first_receipt["reproducibility_pair"]["verified_by_test"] is True
    assert (
        first_receipt["reproducibility_pair"]["verification_receipt"]
        == "REPRODUCIBILITY_PAIR.json"
    )
    assert first_receipt["reproducibility_pair"][
        "verification_receipt_sha256"
    ].startswith("sha256:")
    assert first_receipt["qualification"] == second_receipt["qualification"]
    assert _file_bytes(first) == _file_bytes(tmp_path / "second/student")
    assert (first.parent / "student.tgz").read_bytes() == (
        tmp_path / "second/student.tgz"
    ).read_bytes()
    assert (first.parent / "producer_artifact.v4.tgz").read_bytes() == (
        tmp_path / "second/producer_artifact.v4.tgz"
    ).read_bytes()


def test_p6_u1_source_passport_linkage_and_selection_fields() -> None:
    artifact = FIXTURE / "student"
    with open_verified_student_jsonl_records_v6(
        artifact, "selected_passport_index/default", strict=True
    ) as records:
        passports = list(records)
    with open_verified_student_jsonl_records_v6(
        artifact, "example_registry/default", strict=True
    ) as records:
        stable_ids = {str(row["example_id"]) for row in records}
    with open_verified_student_m7_payload_v6(
        artifact, "selected_exemplar_payload/default", strict=True
    ) as records:
        payloads = list(records)
    passport_keys = {
        (str(row["selected_example_id"]), int(row["selected_position"]))
        for row in passports
    }
    payload_keys = {
        (str(row["selected_example_id"]), int(row["selected_position"]))
        for row in payloads
    }
    assert payload_keys == passport_keys
    assert {key[0] for key in payload_keys} <= stable_ids
    assert len({key[0] for key in payload_keys}) == SELECTED_ID_COUNT
    for row in payloads:
        assert row["payload_ref"]["c5_authoritative_coordinate"] is True
        assert row["payload_ref"]["source_position"] == row["selected_position"]
        assert row["source_score"] == row["selected_score"]
        assert isinstance(row["score_top_token_id"], int)
        assert row["top_token_ids"]


def test_p6_u1_student_profile_has_no_private_or_host_leakage() -> None:
    artifact = FIXTURE / "student"
    relative_files = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file()
    }
    assert not any(
        path.startswith(("c6/", "reports/", ".staging")) for path in relative_files
    )
    for path in artifact.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/var/folders/" not in text
        assert "/tmp/" not in text
        assert ".staging" not in text


def test_p6_u1_behavior_input_mutation_is_identity_bearing() -> None:
    _, spec = load_declared_inputs(SPEC)
    mutated = json.loads(json.dumps(spec))
    mutated["behavior"]["sequence_length"] += 1
    assert declared_input_digest(mutated) != declared_input_digest(spec)


def _resolve(artifact: Path):
    result = validate_and_resolve_student_consumption(
        artifact, profile_id=PROFILE_ID, strict=True
    )
    assert result.ok and result.descriptor is not None
    return result.descriptor


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
