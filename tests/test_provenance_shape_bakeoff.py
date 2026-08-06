"""Private tests for the corrected provenance-shape experiment only."""

from __future__ import annotations

import json

import pytest

from experiments.provenance_shape_bakeoff import (
    ATTESTATION_SCHEMA,
    build_projection,
    compare_immutable_expected_identity,
    materially_reduced,
    require_external_attestation,
    validate_archive,
    validate_candidate,
    validate_standard_projection,
    validate_transaction,
)
from experiments.provenance_shape_benchmark import deterministic_records

AUTHORITY = {
    "teacher": "declared",
    "tokenizer": "declared",
    "selection": "fixed",
}
CONTRACT_VERSION = "experimental-contract-vnext"
BEHAVIORAL_POLICY = "experimental-behavior-policy-vnext"


def _records() -> list[dict[str, object]]:
    source = (
        "tests/fixtures/native_v3_student_v6_smoke/producer_artifact.v4/"
        "selected_exemplars/shards/shard-00000.jsonl"
    )
    return [json.loads(line) for line in open(source, encoding="utf-8")]


def _projection(tmp_path, *, shape: str, capacity: int = 3):
    return build_projection(
        _records(),
        tmp_path / shape,
        authority=AUTHORITY,
        capacity=capacity,
        shape=shape,
        contract_version=CONTRACT_VERSION,
        behavioral_policy_identity=BEHAVIORAL_POLICY,
    )


def _rejects_standard(result) -> None:
    with pytest.raises(ValueError):
        list(validate_standard_projection(result.root, authority=AUTHORITY))


def _rejects_transaction(result) -> None:
    with pytest.raises(ValueError):
        validate_transaction(
            result.root,
            authority=AUTHORITY,
            configuration=result.configuration,
        )


def _rewrite_jsonl(path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _attestation(result) -> dict[str, str]:
    cover = json.loads((result.root / "cover.json").read_text())
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "semantic_root": cover["semantic_root"],
        "semantic_authority_identity": cover["semantic_authority_identity"],
        "contract_version": cover["contract_version"],
        "behavioral_policy_identity": cover["behavioral_policy_identity"],
        "reference": "https://attester.example.invalid/releases/test-receipt",
    }


def test_candidate_has_only_operational_public_proofs_and_private_journal(
    tmp_path,
) -> None:
    current = _projection(tmp_path / "current", shape="current", capacity=1)
    candidate = _projection(tmp_path / "candidate", shape="candidate", capacity=1)
    assert current.sequence_digest == candidate.sequence_digest
    assert current.semantic_root == candidate.semantic_root
    row = json.loads(
        (candidate.root / "payload-index.jsonl").read_text().splitlines()[0]
    )
    assert set(row) == {
        "logical_id",
        "selection_index",
        "shard_id",
        "row",
        "record_digest",
    }
    assert not (candidate.root / ".native").exists()
    assert (candidate.root / ".journal.json").is_file()
    assert (candidate.root / ".journal-sealed.jsonl").is_file()
    assert (
        list(validate_standard_projection(candidate.root, authority=AUTHORITY))
        == _records()
    )
    validate_transaction(
        candidate.root,
        authority=AUTHORITY,
        configuration=candidate.configuration,
    )


def test_standard_consumer_does_not_depend_on_private_transaction_state(
    tmp_path,
) -> None:
    result = _projection(tmp_path, shape="candidate", capacity=1)
    (result.root / ".journal.json").unlink()
    (result.root / ".journal-sealed.jsonl").unlink()
    assert (
        list(validate_standard_projection(result.root, authority=AUTHORITY))
        == _records()
    )
    with pytest.raises(FileNotFoundError):
        validate_transaction(
            result.root,
            authority=AUTHORITY,
            configuration=result.configuration,
        )


@pytest.mark.parametrize("shape", ["current", "candidate"])
@pytest.mark.parametrize(
    "mutation",
    [
        "probability",
        "token_id",
        "corridor",
        "linkage",
        "delete",
        "duplicate",
        "reorder",
    ],
)
def test_stale_public_proofs_reject_operational_content_drift(
    tmp_path, shape, mutation
) -> None:
    result = _projection(tmp_path, shape=shape)
    shard = result.root / "shards" / "shard-00000.jsonl"

    def mutate(rows) -> None:
        if mutation == "probability":
            rows[0]["top_probs"][0] += 0.01
        elif mutation == "token_id":
            rows[0]["top_token_ids"][0] += 1
        elif mutation == "corridor":
            rows[0]["corridor_mode_id"] += 1
        elif mutation == "linkage":
            rows[0]["payload_ref"]["source_shard_id"] += 1
        elif mutation == "delete":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(dict(rows[0]))
        else:
            rows[0], rows[1] = rows[1], rows[0]

    _rewrite_jsonl(shard, mutate)
    _rejects_standard(result)


@pytest.mark.parametrize("shape", ["current", "candidate"])
@pytest.mark.parametrize(
    "member,mutation",
    [
        ("payload-index.jsonl", "index"),
        ("shard-index.jsonl", "index"),
        ("cover.json", "cover"),
    ],
)
def test_stale_public_proofs_reject_index_cover_and_pointer_incoherence(
    tmp_path, shape, member, mutation
) -> None:
    result = _projection(tmp_path, shape=shape)
    path = result.root / member
    if mutation == "cover":
        value = json.loads(path.read_text())
        value["manifest_header"]["path"] = "other-header.json"
        path.write_text(json.dumps(value))
    elif member == "payload-index.jsonl":
        _rewrite_jsonl(path, lambda rows: rows[0].__setitem__("selection_index", 99))
    else:
        _rewrite_jsonl(path, lambda rows: rows[0].__setitem__("first", 99))
    _rejects_standard(result)


@pytest.mark.parametrize("shape", ["current", "candidate"])
@pytest.mark.parametrize(
    "mutation", ["flip", "truncate", "append", "delete", "replace"]
)
def test_standard_validation_rejects_raw_shard_mutations_before_rows(
    tmp_path, shape, mutation
) -> None:
    result = _projection(tmp_path, shape=shape, capacity=1)
    first = result.root / "shards" / "shard-00000.jsonl"
    raw = first.read_bytes()
    if mutation == "flip":
        first.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif mutation == "truncate":
        first.write_bytes(raw[:-1])
    elif mutation == "append":
        first.write_bytes(raw + b"x")
    elif mutation == "delete":
        first.unlink()
    else:
        first.write_bytes((result.root / "shards" / "shard-00001.jsonl").read_bytes())
    _rejects_standard(result)


@pytest.mark.parametrize("shape", ["current", "candidate"])
def test_archive_raw_identity_rejects_truncation_and_append(tmp_path, shape) -> None:
    result = _projection(tmp_path, shape=shape)
    archive = result.root.with_suffix(".tar")
    validate_archive(archive, expected_digest=result.archive_digest)
    raw = archive.read_bytes()
    archive.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="archive raw-integrity"):
        validate_archive(archive, expected_digest=result.archive_digest)
    archive.write_bytes(raw + b"unrelated")
    with pytest.raises(ValueError, match="archive raw-integrity"):
        validate_archive(archive, expected_digest=result.archive_digest)


def test_transaction_faults_are_rejected_without_affecting_standard_format(
    tmp_path,
) -> None:
    result = _projection(tmp_path, shape="candidate", capacity=1)
    journal_path = result.root / ".journal.json"

    journal = json.loads(journal_path.read_text())
    journal["state"] = "open"
    journal_path.write_text(json.dumps(journal))
    _rejects_transaction(result)

    result = _projection(tmp_path / "partial", shape="candidate", capacity=1)
    (result.root / "shards" / "shard-partial.jsonl").write_text("partial")
    _rejects_standard(result)

    result = _projection(tmp_path / "unreceipted", shape="candidate", capacity=1)
    sealed = result.root / ".journal-sealed.jsonl"
    sealed.write_text("")
    _rejects_transaction(result)

    result = _projection(tmp_path / "incomplete", shape="candidate", capacity=1)
    journal_path = result.root / ".journal.json"
    journal = json.loads(journal_path.read_text())
    del journal["promotion_marker"]
    journal_path.write_text(json.dumps(journal))
    _rejects_transaction(result)

    result = _projection(tmp_path / "stale", shape="candidate", capacity=1)
    with pytest.raises(ValueError, match="stale transaction configuration"):
        validate_transaction(
            result.root,
            authority=AUTHORITY,
            configuration={"different": "transaction"},
        )

    result = _projection(tmp_path / "cross", shape="candidate", capacity=1)
    with pytest.raises(ValueError, match="cross-authority"):
        validate_transaction(
            result.root,
            authority={"teacher": "different"},
            configuration=result.configuration,
        )

    result = _projection(tmp_path / "mixed", shape="candidate", capacity=1)
    altered = _records()
    altered[0]["top_token_ids"][0] += 1
    other = build_projection(
        altered,
        tmp_path / "other",
        authority=AUTHORITY,
        capacity=1,
        shape="candidate",
        contract_version=CONTRACT_VERSION,
        behavioral_policy_identity=BEHAVIORAL_POLICY,
    )
    (result.root / ".journal-sealed.jsonl").write_bytes(
        (other.root / ".journal-sealed.jsonl").read_bytes()
    )
    _rejects_transaction(result)


def test_semantic_identity_ignores_declared_resharding_and_repackaging(
    tmp_path,
) -> None:
    single = _projection(tmp_path / "single", shape="candidate", capacity=1)
    grouped = _projection(tmp_path / "grouped", shape="candidate", capacity=3)
    assert single.sequence_digest == grouped.sequence_digest
    assert single.semantic_root == grouped.semantic_root
    assert single.archive_digest != grouped.archive_digest
    assert (
        list(validate_standard_projection(single.root, authority=AUTHORITY))
        == _records()
    )
    assert (
        list(validate_standard_projection(grouped.root, authority=AUTHORITY))
        == _records()
    )


def test_recomputed_artifact_is_standard_valid_but_rejected_by_expected_identity(
    tmp_path,
) -> None:
    clean = _projection(tmp_path / "clean", shape="candidate")
    altered = _records()
    altered[0]["top_token_ids"][0] += 1
    substituted = build_projection(
        altered,
        tmp_path / "substituted",
        authority=AUTHORITY,
        capacity=3,
        shape="candidate",
        contract_version=CONTRACT_VERSION,
        behavioral_policy_identity=BEHAVIORAL_POLICY,
    )
    assert substituted.semantic_root != clean.semantic_root
    assert (
        list(validate_standard_projection(substituted.root, authority=AUTHORITY))
        == altered
    )
    with pytest.raises(ValueError, match="immutable expected semantic"):
        compare_immutable_expected_identity(
            substituted.root, expected_semantic_root=clean.semantic_root
        )
    require_external_attestation(clean.root, attestation=_attestation(clean))
    with pytest.raises(ValueError, match="external attestation identity"):
        require_external_attestation(substituted.root, attestation=_attestation(clean))


def test_standard_authority_mismatch_is_rejected(tmp_path) -> None:
    result = _projection(tmp_path, shape="candidate")
    with pytest.raises(ValueError, match="semantic authority"):
        list(validate_standard_projection(result.root, authority={"teacher": "wrong"}))


def test_full_private_lifecycle_helper_remains_available_for_experiment(
    tmp_path,
) -> None:
    result = _projection(tmp_path, shape="candidate")
    assert (
        list(
            validate_candidate(
                result.root,
                authority=AUTHORITY,
                configuration=result.configuration,
            )
        )
        == _records()
    )


def test_materiality_is_predeclared_and_noise_aware() -> None:
    assert materially_reduced([100.0, 101.0, 102.0], [70.0, 71.0, 72.0])
    assert not materially_reduced([100.0, 100.0, 100.0], [85.0, 85.0, 85.0])


def test_deterministic_benchmark_expansion_is_unique_and_no_inference() -> None:
    records = list(deterministic_records(_records(), 9))
    assert [record["selected_position"] for record in records] == list(range(9))
    assert len({record["selected_example_id"] for record in records}) == 9
    assert records[0]["top_token_ids"] == _records()[0]["top_token_ids"]
