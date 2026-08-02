"""B4 ordinary-production evidence for explicit native-v3 Student v6."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

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
    assert (
        directory_result.descriptor.behavioral_authority_digest
        == archive_result.descriptor.behavioral_authority_digest
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
