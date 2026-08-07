from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from radjax_contract.tome.v3 import (
    AttestationRequirement,
    compare_governed_tome_artifact_v3,
    open_tome_artifact_v3,
    semantic_root,
    validate_archive_receipt_v3,
    validate_tome_artifact_v3,
    verify_external_tome_attestation_v3,
)
from radjax_contract.tome.v3.issues import TomeV3ValidationError

from radjax_tome.tome.artifact_v3 import (
    FinalizedV3Handoff,
    pack_v3_rtome,
    publish_v3_from_handoff,
    resume_v3_archive_from_directory,
)

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


def test_fixture_rtome_transport_is_accepted(tmp_path: Path) -> None:
    rtome = pack_v3_rtome(DIRECTORY, tmp_path / "artifact.rtome")
    repeat = pack_v3_rtome(DIRECTORY, tmp_path / "artifact-repeat.rtome")
    report = validate_tome_artifact_v3(rtome)
    assert report.transport == "rtome"
    assert rtome.read_bytes() == repeat.read_bytes()
    assert (
        report.semantic_root
        == json.loads((DIRECTORY / "provenance/semantic-identity.json").read_text())[
            "semantic_root"
        ]
    )


def test_fixture_archive_resume_reproduces_promoted_archive_and_refuses_replace(
    tmp_path: Path,
) -> None:
    source = publish_v3_from_handoff(_fixture_handoff(), tmp_path / "source")
    resumed = resume_v3_archive_from_directory(
        source.directory, tmp_path / "resumed.tgz"
    )
    assert resumed.read_bytes() == source.archive.read_bytes()
    assert validate_tome_artifact_v3(resumed).semantic_root == source.semantic_root
    with pytest.raises(ValueError, match="v3_archive_resume_target_invalid"):
        resume_v3_archive_from_directory(source.directory, resumed)


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


@pytest.mark.parametrize(
    ("case_id", "relative", "mutation"),
    (
        (
            "PC_fixture_payload_index_pointer",
            "selected_exemplars/payload-index.jsonl",
            lambda rows: rows[0].__setitem__("row", 99),
        ),
        (
            "PC_fixture_payload_index_declared_order",
            "selected_exemplars/payload-index.jsonl",
            lambda rows: rows[0].__setitem__("selection_index", 1),
        ),
        (
            "PC_fixture_shard_index_count",
            "selected_exemplars/payload-shards.jsonl",
            lambda rows: rows[0].__setitem__("record_count", 3),
        ),
        (
            "PC_fixture_shard_index_range",
            "selected_exemplars/payload-shards.jsonl",
            lambda rows: rows[1].__setitem__("first_selection_index", 0),
        ),
    ),
    ids=("pointer", "declared-order", "shard-count", "shard-range"),
)
def test_fixture_index_pointer_count_and_range_mutations_fail_closed(
    tmp_path: Path,
    case_id: str,
    relative: str,
    mutation,
) -> None:
    """Each index incoherence is detected, even with stale raw receipts."""

    copied = _copy_directory(tmp_path / case_id)
    path = copied / relative
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    _raise_validation(lambda: validate_tome_artifact_v3(copied))


@pytest.mark.parametrize(
    ("case_id", "operation"),
    (
        ("PC_fixture_record_deleted", "delete"),
        ("PC_fixture_record_duplicated", "duplicate"),
        ("PC_fixture_record_reordered", "reorder"),
    ),
)
def test_fixture_record_deletion_duplication_and_reorder_fail_closed(
    tmp_path: Path, case_id: str, operation: str
) -> None:
    copied = _copy_directory(tmp_path / case_id)
    shard = copied / "selected_exemplars/shards/shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    if operation == "delete":
        rows.pop()
    elif operation == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows.reverse()
    shard.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    _raise_validation(lambda: validate_tome_artifact_v3(copied))


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


def test_fixture_root_changes_for_order_authority_and_policy_changes(
    tmp_path: Path,
) -> None:
    handoff = _fixture_handoff()
    baseline = publish_v3_from_handoff(handoff, tmp_path / "baseline")
    reordered = publish_v3_from_handoff(
        FinalizedV3Handoff(
            tuple(reversed(handoff.records)),
            handoff.authority,
            handoff.policy,
            handoff.selection_indexes,
            handoff.shard_capacity,
        ),
        tmp_path / "reordered",
    )
    changed_authority = copy.deepcopy(handoff.authority)
    changed_authority["entries"][0]["identity"] = "sha256:" + "f" * 64
    authority_variant = publish_v3_from_handoff(
        FinalizedV3Handoff(
            handoff.records,
            changed_authority,
            handoff.policy,
            handoff.selection_indexes,
            handoff.shard_capacity,
        ),
        tmp_path / "authority",
    )
    changed_policy = copy.deepcopy(handoff.policy)
    changed_policy["selection_policy"] = "different_governed_policy_v1"
    policy_variant = publish_v3_from_handoff(
        FinalizedV3Handoff(
            handoff.records,
            handoff.authority,
            changed_policy,
            handoff.selection_indexes,
            handoff.shard_capacity,
        ),
        tmp_path / "policy",
    )
    roots = {
        baseline.semantic_root,
        reordered.semantic_root,
        authority_variant.semantic_root,
        policy_variant.semantic_root,
    }
    assert len(roots) == 4
    for publication in (reordered, authority_variant, policy_variant):
        assert validate_tome_artifact_v3(publication.directory).ok


@pytest.mark.parametrize("field", ("contract_version", "semantic_profile_id"))
def test_fixture_root_changes_for_contract_and_profile_changes(field: str) -> None:
    identity = json.loads((DIRECTORY / "provenance/semantic-identity.json").read_text())
    original_root = identity.pop("semantic_root")
    changed = copy.deepcopy(identity)
    changed[field] = str(changed[field]) + ".changed"
    assert semantic_root(identity) == original_root
    assert semantic_root(changed) != original_root


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
    with pytest.raises(TomeV3ValidationError, match="governed_expected_root_mismatch"):
        compare_governed_tome_artifact_v3(replacement.directory, EXPECTED)
    with pytest.raises(TomeV3ValidationError, match="attestation_binding_mismatch"):
        verify_external_tome_attestation_v3(
            replacement.directory,
            ATTESTATION,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
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


def test_external_attestation_optional_required_and_unavailable_outcomes(
    tmp_path: Path,
) -> None:
    optional = verify_external_tome_attestation_v3(
        DIRECTORY,
        None,
        requirement=AttestationRequirement.OPTIONAL,
        evaluation_time_utc=EVALUATION,
    )
    assert optional.status == "not_supplied_optional"
    with pytest.raises(TomeV3ValidationError, match="attestation_unavailable_required"):
        verify_external_tome_attestation_v3(
            DIRECTORY,
            None,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )
    valid_optional = verify_external_tome_attestation_v3(
        DIRECTORY,
        ATTESTATION,
        requirement=AttestationRequirement.OPTIONAL,
        evaluation_time_utc=EVALUATION,
    )
    assert valid_optional.status == "verified"

    malformed = tmp_path / "malformed-attestation.json"
    attestation = json.loads(ATTESTATION.read_text())
    attestation["envelope"] = "%%%"
    malformed.write_text(json.dumps(attestation, sort_keys=True, separators=(",", ":")))
    with pytest.raises(TomeV3ValidationError, match="attestation_envelope_invalid"):
        verify_external_tome_attestation_v3(
            DIRECTORY,
            malformed,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )

    unsupported = tmp_path / "unsupported-attestation.json"
    attestation = json.loads(ATTESTATION.read_text())
    attestation["envelope_algorithm_id"] = "future-signature-v1"
    unsupported.write_text(
        json.dumps(attestation, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(
        TomeV3ValidationError, match="attestation_algorithm_unsupported"
    ):
        verify_external_tome_attestation_v3(
            DIRECTORY,
            unsupported,
            requirement=AttestationRequirement.REQUIRED,
            evaluation_time_utc=EVALUATION,
        )


def test_fixture_rebuild_reproduces_semantic_receipt_projection(tmp_path: Path) -> None:
    """Identical finalized handoffs produce identical public semantic receipt data."""

    handoff = _fixture_handoff()
    first = publish_v3_from_handoff(handoff, tmp_path / "first")
    second = publish_v3_from_handoff(handoff, tmp_path / "second")
    first_report = validate_tome_artifact_v3(first.directory)
    second_report = validate_tome_artifact_v3(second.directory)
    identity = json.loads((DIRECTORY / "provenance/semantic-identity.json").read_text())
    assert first_report.semantic_root == second_report.semantic_root
    assert first_report.semantic_root == identity["semantic_root"]
    assert first_report.record_count == second_report.record_count
    assert first_report.shard_count == second_report.shard_count
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert (
        hashlib.sha256(first.archive.read_bytes()).hexdigest()
        == hashlib.sha256(second.archive.read_bytes()).hexdigest()
    )
    assert json.loads(
        (first.directory / "provenance/semantic-authority.json").read_text()
    ) == json.loads(
        (second.directory / "provenance/semantic-authority.json").read_text()
    )
    assert json.loads(
        (first.directory / "provenance/behavioral-policy.json").read_text()
    ) == json.loads(
        (second.directory / "provenance/behavioral-policy.json").read_text()
    )
