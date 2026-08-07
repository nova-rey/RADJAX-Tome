from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from radjax_contract.tome.v3 import (
    AttestationRequirement,
    compare_governed_tome_artifact_v3,
    open_tome_artifact_v3,
    validate_archive_receipt_v3,
    validate_tome_artifact_v3,
    verify_external_tome_attestation_v3,
)
from radjax_contract.tome.v3.issues import TomeV3ValidationError

from radjax_tome.tome.artifact_v3 import FinalizedV3Handoff, publish_v3_from_handoff

FIXTURE = Path("tests/fixtures/tome_artifact_v3_smoke")
DIRECTORY = FIXTURE / "artifact.v3"
ARCHIVE = FIXTURE / "artifact.v3.tgz"
EXPECTED = FIXTURE / "governed_expectation.json"
ATTESTATION = FIXTURE / "external_attestation.json"
RECEIPT = FIXTURE / "archive_receipt.json"
EVALUATION = datetime(2026, 8, 6, tzinfo=UTC)


def _fixture_handoff() -> FinalizedV3Handoff:
    shard_index = [
        json.loads(line)
        for line in (DIRECTORY / "selected_exemplars/payload-shards.jsonl")
        .read_text()
        .splitlines()
    ]
    records = []
    for shard in shard_index:
        path = DIRECTORY / shard["path"]
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    authority = json.loads(
        (DIRECTORY / "provenance/semantic-authority.json").read_text()
    )
    policy = json.loads((DIRECTORY / "provenance/behavioral-policy.json").read_text())
    return FinalizedV3Handoff(
        tuple(records),
        authority,
        policy,
        tuple(range(len(records))),
        2,
    )


def _copy_directory(tmp_path: Path) -> Path:
    target = tmp_path / "artifact.v3"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DIRECTORY, target)
    return target


def _raise_validation(callable_obj):
    with pytest.raises(TomeV3ValidationError):
        callable_obj()


def test_fixture_directory_archive_governed_external_and_receipt() -> None:
    directory = validate_tome_artifact_v3(DIRECTORY)
    archive = validate_tome_artifact_v3(ARCHIVE)
    assert directory.semantic_root == archive.semantic_root
    assert directory.record_count == archive.record_count == 4
    assert directory.shard_count == archive.shard_count == 2
    assert compare_governed_tome_artifact_v3(DIRECTORY, EXPECTED).matches
    assert compare_governed_tome_artifact_v3(ARCHIVE, EXPECTED).matches
    for artifact in (DIRECTORY, ARCHIVE):
        report = verify_external_tome_attestation_v3(
            artifact,
            ATTESTATION,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )
        assert report.status == "verified"
    receipt = validate_archive_receipt_v3(ARCHIVE, RECEIPT)
    assert receipt.matches


def test_fixture_raw_and_graph_corruption_are_rejected(tmp_path: Path) -> None:
    raw_corrupt = _copy_directory(tmp_path)
    shard = raw_corrupt / "selected_exemplars/shards/shard-00000.jsonl"
    data = bytearray(shard.read_bytes())
    data[0] ^= 1
    shard.write_bytes(data)
    _raise_validation(lambda: validate_tome_artifact_v3(raw_corrupt))

    graph_corrupt = _copy_directory(tmp_path / "graph")
    cover_path = graph_corrupt / "cover_page.json"
    cover = json.loads(cover_path.read_text())
    cover["record_count"] += 1
    cover_path.write_text(json.dumps(cover, sort_keys=True, separators=(",", ":")))
    _raise_validation(lambda: validate_tome_artifact_v3(graph_corrupt))


def test_fixture_streaming_rejects_first_and_later_corrupt_shards(
    tmp_path: Path,
) -> None:
    first = _copy_directory(tmp_path / "first")
    first_shard = first / "selected_exemplars/shards/shard-00000.jsonl"
    first_bytes = bytearray(first_shard.read_bytes())
    first_bytes[-2] ^= 1
    first_shard.write_bytes(first_bytes)
    with pytest.raises(TomeV3ValidationError):
        with open_tome_artifact_v3(first) as reader:
            next(iter(reader))

    later = _copy_directory(tmp_path / "later")
    later_shard = later / "selected_exemplars/shards/shard-00001.jsonl"
    later_bytes = bytearray(later_shard.read_bytes())
    later_bytes[-2] ^= 1
    later_shard.write_bytes(later_bytes)
    with open_tome_artifact_v3(later) as reader:
        rows = iter(reader)
        assert next(rows)["selected_example_id"]
        assert next(rows)["selected_example_id"]
        with pytest.raises(TomeV3ValidationError):
            next(rows)


def test_fixture_resharding_preserves_semantic_root(tmp_path: Path) -> None:
    handoff = _fixture_handoff()
    one = publish_v3_from_handoff(
        handoff.__class__(
            handoff.records,
            handoff.authority,
            handoff.policy,
            handoff.selection_indexes,
            1,
        ),
        tmp_path / "one",
    )
    four = publish_v3_from_handoff(
        handoff.__class__(
            handoff.records,
            handoff.authority,
            handoff.policy,
            handoff.selection_indexes,
            4,
        ),
        tmp_path / "four",
    )
    assert one.semantic_root == four.semantic_root
    assert validate_tome_artifact_v3(one.directory).semantic_root == four.semantic_root


def test_coherent_replacement_passes_standard_but_fails_external_comparisons(
    tmp_path: Path,
) -> None:
    handoff = _fixture_handoff()
    changed = copy.deepcopy(handoff.records)
    replacement_token = int(changed[0]["top_token_ids"][0]) + 1
    changed[0]["top_token_ids"][0] = replacement_token
    changed[0]["source_top_token_id"] = replacement_token
    changed[0]["score_top_token_id"] = replacement_token
    replacement = publish_v3_from_handoff(
        FinalizedV3Handoff(
            tuple(changed),
            handoff.authority,
            handoff.policy,
            handoff.selection_indexes,
            handoff.shard_capacity,
        ),
        tmp_path / "replacement",
    )
    assert (
        validate_tome_artifact_v3(replacement.directory).semantic_root
        != json.loads((DIRECTORY / "provenance/semantic-identity.json").read_text())[
            "semantic_root"
        ]
    )
    _raise_validation(
        lambda: compare_governed_tome_artifact_v3(replacement.directory, EXPECTED)
    )
    _raise_validation(
        lambda: verify_external_tome_attestation_v3(
            replacement.directory,
            ATTESTATION,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )
    )


def test_external_evidence_must_be_supplied_outside_artifact() -> None:
    _raise_validation(
        lambda: verify_external_tome_attestation_v3(
            DIRECTORY,
            DIRECTORY / "cover_page.json",
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )
    )
