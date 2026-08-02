"""B4 ordinary-production evidence for explicit native-v3 Student v6."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import STUDENT, validate_tome_package
from radjax_tome.tome.v6_fixture import CONTRACT_COMMIT, PROFILE_ID

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native_v3_student_v6_smoke"


def test_b4_committed_fixture_resolves_all_v6_authority_domains() -> None:
    receipt = json.loads((FIXTURE / "FIXTURE_RECEIPT.json").read_text())
    assert receipt["contract"]["commit"] == CONTRACT_COMMIT
    assert receipt["profile_id"] == PROFILE_ID
    directory = FIXTURE / "student"
    assert validate_tome_package(directory, profile=STUDENT).ok
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
    assert receipt["directory"]["raw_tree_digest"] != receipt["archive"].get(
        "raw_sha256"
    )
    selected = next(
        item
        for item in directory_result.descriptor.authority_resources
        if item.role == "selected_exemplar_payload"
    )
    assert selected.encoding == "m7_tome_archive"
