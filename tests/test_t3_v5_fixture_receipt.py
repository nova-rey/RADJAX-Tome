"""T3 evidence for the ordinary deterministic v5 smoke producer path."""

from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.tome import STUDENT, validate_tome_package
from radjax_tome.tome.v5_fixture import (
    CONTRACT_COMMIT,
    CONTRACT_VERSION,
    FIXTURE_ID,
    FIXTURE_SCHEMA_VERSION,
    FIXTURE_TOME_COMMIT,
    PROFILE_ID,
    build_v5_language_tokenizer_fixture,
    raw_tree_digest,
    sha256_file,
    tree_digest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native_v3_student_v5_smoke"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t3_committed_v5_fixture_receipt_binds_contract_and_physical_evidence() -> None:
    receipt = _json(FIXTURE / "FIXTURE_RECEIPT.json")
    student = FIXTURE / "student"
    binding = _json(student / "manifests/language_tokenizer_binding_v1.json")
    cover = _json(student / "cover_page.json")

    assert receipt["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert receipt["fixture_id"] == FIXTURE_ID
    assert receipt["contract"] == {
        "version": CONTRACT_VERSION,
        "commit": CONTRACT_COMMIT,
    }
    assert receipt["tome_commit_at_production"] == FIXTURE_TOME_COMMIT
    assert receipt["profile_id"] == PROFILE_ID
    assert receipt["generic_binding_digest"] == binding["canonical_binding_digest"]
    assert receipt["fixture_semantic_digest"] == cover["identity"]["semantic_digest"]
    assert receipt["fixture_raw_digest"] == raw_tree_digest(student)
    assert receipt["fixture_tree_digest"] == tree_digest(student)
    assert receipt["binding_sha256"] == sha256_file(
        student / "manifests/language_tokenizer_binding_v1.json"
    )
    assert receipt["vocabulary_sha256"] == sha256_file(
        student / "resources/tokenizer_vocabulary.jsonl"
    )
    assert receipt["validator"] == {
        "entry_point": (
            "radjax_contract.tome.validate_and_resolve_student_consumption"
        ),
        "arguments": {"profile_id": PROFILE_ID, "strict": True},
    }


def test_t3_contract_and_tome_validators_admit_the_committed_v5_fixture() -> None:
    student = FIXTURE / "student"

    tome = validate_tome_package(student, profile=STUDENT)
    contract = validate_and_resolve_student_consumption(
        student,
        profile_id=PROFILE_ID,
        strict=True,
    )

    assert tome.ok, tome.blockers
    assert contract.ok, contract.issues


def test_t3_repeated_ordinary_smoke_production_preserves_v5_semantics(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    left = build_v5_language_tokenizer_fixture(fixture_root)
    left_receipt = _json(fixture_root / "FIXTURE_RECEIPT.json")
    left_binding = (left / "manifests/language_tokenizer_binding_v1.json").read_bytes()
    left_vocabulary = (left / "resources/tokenizer_vocabulary.jsonl").read_bytes()
    right = build_v5_language_tokenizer_fixture(fixture_root, overwrite=True)
    right_receipt = _json(fixture_root / "FIXTURE_RECEIPT.json")

    assert (
        left_binding
        == (right / "manifests/language_tokenizer_binding_v1.json").read_bytes()
    )
    assert (
        left_vocabulary == (right / "resources/tokenizer_vocabulary.jsonl").read_bytes()
    )
    assert left_receipt["binding_sha256"] == right_receipt["binding_sha256"]
    assert left_receipt["vocabulary_sha256"] == right_receipt["vocabulary_sha256"]
    assert (
        left_receipt["generic_binding_digest"]
        == right_receipt["generic_binding_digest"]
    )
    assert (
        left_receipt["fixture_semantic_digest"]
        == right_receipt["fixture_semantic_digest"]
    )
