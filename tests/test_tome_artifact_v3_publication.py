from __future__ import annotations

import json
from pathlib import Path

import pytest
from radjax_contract.tome.v3.issues import TomeV3ValidationError

from radjax_tome.tome.artifact_v3 import (
    FinalizedV3Handoff,
    publish_v3_from_handoff,
)

CONTRACT_ROOT = Path("contracts/radjax_tome/v3")


def _digest(value: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _handoff() -> FinalizedV3Handoff:
    vector = json.loads(
        (CONTRACT_ROOT / "vectors" / "tome_provenance_v3_vectors.json").read_text()
    )["normative_root_vectors"][1]
    records = tuple(
        {key: value for key, value in record.items() if key != "selection_index"}
        for record in vector["ordered_records"]
    )
    return FinalizedV3Handoff(
        records=records,
        authority=vector["semantic_context"]["authority"],
        policy=vector["semantic_context"]["behavioral_policy"],
        selection_indexes=tuple(range(len(records))),
        shard_capacity=1,
    )


def test_v3_publication_validates_directory_and_archive(tmp_path: Path) -> None:
    publication = publish_v3_from_handoff(_handoff(), tmp_path / "output")
    from radjax_contract.tome.v3.validation import validate_tome_artifact_v3

    directory = validate_tome_artifact_v3(publication.directory)
    archive = validate_tome_artifact_v3(publication.archive)
    assert directory.ok and archive.ok
    assert directory.semantic_root == archive.semantic_root == publication.semantic_root
    assert publication.record_count == 2
    assert publication.shard_count == 2


def test_v3_semantic_root_survives_resharding(tmp_path: Path) -> None:
    handoff = _handoff()
    first = publish_v3_from_handoff(handoff, tmp_path / "first")
    second = publish_v3_from_handoff(
        FinalizedV3Handoff(
            handoff.records,
            handoff.authority,
            handoff.policy,
            handoff.selection_indexes,
            shard_capacity=2,
        ),
        tmp_path / "second",
    )
    assert first.semantic_root == second.semantic_root


def test_v3_projection_rejects_missing_closed_record_field() -> None:
    handoff = _handoff()
    record = dict(handoff.records[0])
    record.pop("selected_score")
    with pytest.raises((ValueError, TomeV3ValidationError)):
        publish_v3_from_handoff(
            FinalizedV3Handoff((record,), handoff.authority, handoff.policy, (0,), 1),
            Path("/tmp/unused-v3-test-output"),
        )
