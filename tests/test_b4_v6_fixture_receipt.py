"""B4 ordinary-production evidence for explicit native-v3 Student v6."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import (
    open_verified_student_m7_payload_v6,
    validate_and_resolve_student_consumption,
)

from radjax_tome.tome import STUDENT, validate_tome_bundle, validate_tome_package
from radjax_tome.tome.v6_fixture import (
    CONTRACT_COMMIT,
    PROFILE_ID,
    build_v6_behavioral_fixture,
    raw_tree_digest,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native_v3_student_v6_smoke"


def test_b4_committed_fixture_resolves_all_v6_authority_domains() -> None:
    receipt = json.loads((FIXTURE / "FIXTURE_RECEIPT.json").read_text())
    assert receipt["contract"]["commit"] == CONTRACT_COMMIT
    assert receipt["profile_id"] == PROFILE_ID
    directory = FIXTURE / "student"
    assert validate_tome_package(directory, profile=STUDENT).ok
    assert validate_tome_bundle(FIXTURE / "student.tgz").ok
    directory_result = validate_and_resolve_student_consumption(
        directory, profile_id=PROFILE_ID, strict=True
    )
    archive_result = validate_and_resolve_student_consumption(
        FIXTURE / "student.tgz", profile_id=PROFILE_ID, strict=True
    )
    assert directory_result.ok and archive_result.ok
    assert directory_result.descriptor is not None
    assert archive_result.descriptor is not None
    for name in (
        "language_binding_digest",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    ):
        assert receipt["directory"][name] == getattr(directory_result.descriptor, name)
        assert receipt["archive"][name] == getattr(archive_result.descriptor, name)
    assert _receipt_resources(receipt["directory"]["authority_resources"]) == [
        _descriptor_resource(item.to_dict())
        for item in directory_result.descriptor.authority_resources
    ]
    assert _receipt_resources(receipt["directory"]["non_authority_resources"]) == [
        _descriptor_resource(item.to_dict())
        for item in directory_result.descriptor.non_authority_resources
    ]
    assert _receipt_resources(receipt["archive"]["authority_resources"]) == [
        _descriptor_resource(item.to_dict())
        for item in archive_result.descriptor.authority_resources
    ]
    assert _receipt_resources(receipt["archive"]["non_authority_resources"]) == [
        _descriptor_resource(item.to_dict())
        for item in archive_result.descriptor.non_authority_resources
    ]
    for name in (
        "language_binding_digest",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    ):
        assert getattr(directory_result.descriptor, name) == getattr(
            archive_result.descriptor, name
        )
    assert receipt["directory"]["raw_tree_digest"] == raw_tree_digest(directory)
    assert receipt["archive"]["raw_sha256"] == sha256_file(FIXTURE / "student.tgz")
    assert receipt["native_m7_sibling"]["sha256"] == sha256_file(
        FIXTURE / "producer_artifact.v4.tgz"
    )
    assert (FIXTURE / "producer_artifact.v4.tgz").read_bytes() == (
        directory / "student_consumption/v6/selected_exemplar_payload.m7.tgz"
    ).read_bytes()
    selected = next(
        item
        for item in directory_result.descriptor.authority_resources
        if item.role == "selected_exemplar_payload"
    )
    assert selected.encoding == "m7_tome_archive"
    assert receipt["production_config"]["selected_exemplar_budget"] == 3
    assert receipt["production_config"]["total_selected_exemplar_budget"] == 3
    assert _selected_coordinates(directory) == [
        ("corpus_000000003", 0),
        ("corpus_000000003", 3),
        ("corpus_000000001", 2),
    ]
    assert len({example_id for example_id, _ in _selected_coordinates(directory)}) >= 2


def test_b4_fresh_ordinary_production_is_repeatably_admitted(tmp_path: Path) -> None:
    first = build_v6_behavioral_fixture(tmp_path / "fixture")
    first_receipt = json.loads((first.parent / "FIXTURE_RECEIPT.json").read_text())
    second = build_v6_behavioral_fixture(tmp_path / "fixture", overwrite=True)
    second_receipt = json.loads((second.parent / "FIXTURE_RECEIPT.json").read_text())
    for key in (
        "language_binding_digest",
        "behavioral_source_identity",
        "behavioral_authority_digest",
        "package_semantic_identity",
        "composition_digest",
    ):
        assert first_receipt["directory"][key] == second_receipt["directory"][key]


def _receipt_resources(rows: object) -> list[dict[str, object]]:
    assert isinstance(rows, list)
    return [
        {key: value for key, value in row.items() if key != "authority"} for row in rows
    ]


def _descriptor_resource(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in {**row, "components": list(row["components"])}.items()
        if key != "authority"
    }


def _selected_coordinates(directory: Path) -> list[tuple[str, int]]:
    with open_verified_student_m7_payload_v6(
        directory, "selected_exemplar_payload/default", strict=True
    ) as reader:
        rows = list(reader)
    return [
        (str(row["selected_example_id"]), int(row["selected_position"])) for row in rows
    ]
