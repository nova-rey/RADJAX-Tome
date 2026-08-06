"""Private audit-branch tests for the experimental provenance bake-off only."""

from __future__ import annotations

import json

import pytest

from experiments.provenance_shape_bakeoff import (
    build_projection,
    materially_reduced,
    validate_archive,
    validate_candidate,
)


def _records() -> list[dict[str, object]]:
    source = (
        "tests/fixtures/native_v3_student_v6_smoke/producer_artifact.v4/"
        "selected_exemplars/shards/shard-00000.jsonl"
    )
    return [json.loads(line) for line in open(source, encoding="utf-8")]


AUTHORITY = {"teacher": "declared", "selection": "fixed"}


def _projection(tmp_path, *, shape: str, capacity: int = 3):
    return build_projection(
        _records(),
        tmp_path / shape,
        authority=AUTHORITY,
        capacity=capacity,
        shape=shape,
    )


def _rejects(result) -> None:
    with pytest.raises(ValueError):
        list(
            validate_candidate(
                result.root,
                authority=AUTHORITY,
                configuration=result.configuration,
            )
        )


def _rewrite_jsonl(path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_candidate_omits_only_audit_targets_and_validates_before_yield(
    tmp_path,
) -> None:
    current = build_projection(
        _records(),
        tmp_path / "current",
        authority=AUTHORITY,
        capacity=1,
        shape="current",
    )
    candidate = build_projection(
        _records(),
        tmp_path / "candidate",
        authority=AUTHORITY,
        capacity=1,
        shape="candidate",
    )
    assert current.sequence_digest == candidate.sequence_digest
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
    assert (
        list(
            validate_candidate(
                candidate.root,
                authority=AUTHORITY,
                configuration=candidate.configuration,
            )
        )
        == _records()
    )


def test_candidate_rejects_corruption_and_cross_authority_journal(tmp_path) -> None:
    result = build_projection(
        _records(),
        tmp_path / "candidate",
        authority=AUTHORITY,
        capacity=1,
        shape="candidate",
    )
    shard = result.root / "shards" / "shard-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ValueError, match="inventory member mismatch"):
        list(validate_candidate(result.root, authority=AUTHORITY))
    # Restore by rebuilding, then prove a stale/cross-authority private journal fails.
    result = build_projection(
        _records(),
        tmp_path / "candidate2",
        authority=AUTHORITY,
        capacity=1,
        shape="candidate",
    )
    journal = json.loads((result.root / ".journal.json").read_text())
    journal["authority_digest"] = "sha256:" + "0" * 64
    (result.root / ".journal.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError, match="cross-authority"):
        list(validate_candidate(result.root, authority=AUTHORITY))


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
def test_stale_public_proofs_reject_required_content_mutations(
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
    _rejects(result)


@pytest.mark.parametrize("shape", ["current", "candidate"])
@pytest.mark.parametrize(
    "member,mutation",
    [
        ("payload-index.jsonl", "index"),
        ("shard-index.jsonl", "index"),
        ("cover.json", "cover"),
    ],
)
def test_stale_public_proofs_reject_index_and_cover_mutations(
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
    _rejects(result)


@pytest.mark.parametrize("shape", ["current", "candidate"])
@pytest.mark.parametrize(
    "mutation", ["flip", "truncate", "append", "delete", "replace"]
)
def test_stale_public_proofs_reject_required_raw_shard_mutations(
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
    _rejects(result)


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


def test_candidate_transaction_faults_are_rejected_before_public_rows(tmp_path) -> None:
    result = _projection(tmp_path, shape="candidate", capacity=1)
    journal_path = result.root / ".journal.json"

    journal = json.loads(journal_path.read_text())
    journal["state"] = "open"
    journal_path.write_text(json.dumps(journal))
    _rejects(result)

    result = _projection(tmp_path / "unreceipted", shape="candidate", capacity=1)
    journal_path = result.root / ".journal.json"
    journal = json.loads(journal_path.read_text())
    journal["sealed"].pop()
    journal_path.write_text(json.dumps(journal))
    _rejects(result)

    result = _projection(tmp_path / "incomplete", shape="candidate", capacity=1)
    journal_path = result.root / ".journal.json"
    journal = json.loads(journal_path.read_text())
    del journal["promotion_marker"]
    journal_path.write_text(json.dumps(journal))
    _rejects(result)

    result = _projection(tmp_path / "partial", shape="candidate", capacity=1)
    (result.root / "shards" / "shard-partial.jsonl").write_text("partial")
    _rejects(result)

    result = _projection(tmp_path / "stale", shape="candidate", capacity=1)
    with pytest.raises(ValueError, match="stale transaction configuration"):
        list(
            validate_candidate(
                result.root,
                authority=AUTHORITY,
                configuration={"different": "transaction"},
            )
        )


def test_candidate_semantic_identity_ignores_declared_resharding(tmp_path) -> None:
    single = _projection(tmp_path / "single", shape="candidate", capacity=1)
    grouped = _projection(tmp_path / "grouped", shape="candidate", capacity=3)
    assert single.sequence_digest == grouped.sequence_digest
    assert single.archive_digest != grouped.archive_digest
    assert (
        list(
            validate_candidate(
                single.root,
                authority=AUTHORITY,
                configuration=single.configuration,
            )
        )
        == _records()
    )
    assert (
        list(
            validate_candidate(
                grouped.root,
                authority=AUTHORITY,
                configuration=grouped.configuration,
            )
        )
        == _records()
    )


def test_candidate_cannot_replace_v6_behavioral_authority(tmp_path) -> None:
    altered = _records()
    altered[0]["top_token_ids"][0] += 1
    clean = _projection(tmp_path / "clean", shape="candidate")
    altered_result = build_projection(
        altered,
        tmp_path / "altered",
        authority=AUTHORITY,
        capacity=3,
        shape="candidate",
    )
    assert altered_result.sequence_digest != clean.sequence_digest
    assert (
        list(
            validate_candidate(
                altered_result.root,
                authority=AUTHORITY,
                configuration=altered_result.configuration,
            )
        )
        == altered
    )


def test_materiality_is_predeclared_and_noise_aware() -> None:
    assert materially_reduced([100.0, 101.0, 102.0], [70.0, 71.0, 72.0])
    assert not materially_reduced([100.0, 100.0, 100.0], [85.0, 85.0, 85.0])
