from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from radjax_contract.tome.v3.codec import digest
from radjax_contract.tome.v3.issues import TomeV3ValidationError
from radjax_contract.tome.v3.journal import (
    journal_restart_disposition_v3,
    validate_journal_state_v3,
)
from radjax_contract.tome.v3.models import JournalStateV3

from radjax_tome.tome.artifact_v3 import (
    FinalizedV3Handoff,
    V3ArchivePublicationError,
    V3PublicationCrash,
    publish_v3_from_handoff,
    resume_v3_archive_from_directory,
)

AUTHORITY = digest(b"authority", "authority")


def _state(
    state: str, *, completion: bool = False, promoted: bool = False
) -> JournalStateV3:
    return JournalStateV3(
        transaction_id="tx",
        configuration_identity="3.0.0",
        semantic_authority_identity=AUTHORITY,
        state=state,
        completion_intent=completion,
        promotion_marker=promoted,
    )


@pytest.mark.parametrize(
    ("case", "state", "completion", "promoted", "public_present", "action", "visible"),
    (
        (
            "PC39_before_shard_sealing",
            "OPEN",
            False,
            False,
            False,
            "resume_committed_prefix",
            False,
        ),
        (
            "PC40_after_bytes_before_receipt",
            "SEALING",
            False,
            False,
            False,
            "resume_committed_prefix",
            False,
        ),
        (
            "PC41_after_receipt_before_range_commit",
            "SEALING",
            False,
            False,
            False,
            "resume_committed_prefix",
            False,
        ),
        (
            "PC42_after_range_commit_before_completion",
            "OPEN",
            False,
            False,
            False,
            "resume_committed_prefix",
            False,
        ),
        (
            "PC43_after_completion_intent",
            "COMPLETE_INTENT",
            True,
            False,
            False,
            "derive_public_evidence",
            False,
        ),
        (
            "PC44_after_promotion_intent_without_target",
            "PROMOTING",
            True,
            False,
            False,
            "retry_promotion",
            False,
        ),
        (
            "PC45_after_promotion_intent_with_target",
            "PROMOTING",
            True,
            False,
            True,
            "validate_public_then_mark",
            False,
        ),
        (
            "PC46_after_atomic_rename_before_marker",
            "PROMOTING",
            True,
            False,
            True,
            "validate_public_then_mark",
            False,
        ),
        (
            "PC47_after_durable_completion_marker",
            "PROMOTED",
            True,
            True,
            True,
            "open_completed_public_package",
            True,
        ),
    ),
)
def test_each_journal_crash_boundary_has_an_independent_case(
    case: str,
    state: str,
    completion: bool,
    promoted: bool,
    public_present: bool,
    action: str,
    visible: bool,
) -> None:
    assert case.startswith("PC")
    disposition = journal_restart_disposition_v3(
        _state(state, completion=completion, promoted=promoted),
        public_location_present=public_present,
    )
    assert disposition.action == action
    assert disposition.public_visible is visible


def test_invalid_cross_authority_or_unreceipted_state_fails_closed() -> None:
    state = _state("OPEN")
    with pytest.raises(TomeV3ValidationError):
        validate_journal_state_v3(state, expected_semantic_authority_identity="other")
    with pytest.raises(TomeV3ValidationError):
        validate_journal_state_v3(state, staged_member_paths=("shard-0",))


def _publication_handoff() -> FinalizedV3Handoff:
    vectors = json.loads(
        Path(
            "contracts/radjax_tome/v3/vectors/tome_provenance_v3_vectors.json"
        ).read_text(encoding="utf-8")
    )
    vector = vectors["normative_root_vectors"][1]
    records = tuple(
        {key: value for key, value in record.items() if key != "selection_index"}
        for record in vector["ordered_records"]
    )
    return FinalizedV3Handoff(
        records,
        vector["semantic_context"]["authority"],
        vector["semantic_context"]["behavioral_policy"],
        tuple(range(len(records))),
        1,
    )


@pytest.mark.parametrize(
    "case",
    (
        "PC39_before_shard_sealing",
        "PC40_after_shard_bytes_durable",
        "PC41_after_receipt_durable",
        "PC42_after_range_commit",
        "PC43_after_completion_intent",
        "PC44_after_promotion_intent",
        "PC45_after_target_visible",
        "PC46_after_atomic_rename",
        "PC47_after_completion_marker",
    ),
)
def test_tome_publisher_fault_boundary_preserves_private_state(
    tmp_path: Path, case: str
) -> None:
    """Each PC39-PC47 boundary is independently injectable and inspectable."""

    output_base = tmp_path / case

    def interrupt(event: str) -> None:
        if event == case:
            raise V3PublicationCrash(event)

    with pytest.raises(V3PublicationCrash, match=case):
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=interrupt
        )

    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
    assert not archive.exists()
    if case in {
        "PC45_after_target_visible",
        "PC46_after_atomic_rename",
        "PC47_after_completion_marker",
    }:
        assert directory.is_dir()
    else:
        assert not directory.exists()

    private = sorted(
        {
            *tmp_path.glob(f".{output_base.name}.v3-*"),
            *tmp_path.glob(f".{output_base.name}.v3-journal-*"),
        }
    )
    assert private, case
    # The test emulates a process restart by quarantining the private state;
    # normal production cleanup is still exercised by the non-fault tests.
    for path in private:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def test_new_v3_run_refuses_stale_private_transaction(tmp_path: Path) -> None:
    output_base = tmp_path / "stale"
    stale = tmp_path / ".stale.v3-journal-leftover"
    stale.mkdir()
    with pytest.raises(ValueError, match="stale_private_transaction"):
        publish_v3_from_handoff(_publication_handoff(), output_base)


def test_archive_failure_leaves_directory_partial_and_resume_is_safe(
    tmp_path: Path,
) -> None:
    output_base = tmp_path / "archive-partial"

    def fail_archive(event: str) -> None:
        if event == "ARCHIVE_after_promotion_intent":
            raise RuntimeError("injected_archive_failure")

    with pytest.raises(V3ArchivePublicationError) as failure:
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=fail_archive
        )
    assert failure.value.directory_promoted is True
    assert failure.value.archive_promoted is False
    assert failure.value.directory.is_dir()
    assert not failure.value.archive.exists()

    archive = resume_v3_archive_from_directory(
        failure.value.directory, failure.value.archive
    )
    assert archive.is_file()
    assert hashlib.sha256(archive.read_bytes()).hexdigest()


def test_archive_bytes_and_receipt_are_deterministic(tmp_path: Path) -> None:
    first = publish_v3_from_handoff(_publication_handoff(), tmp_path / "first")
    second = publish_v3_from_handoff(_publication_handoff(), tmp_path / "second")
    first_bytes = first.archive.read_bytes()
    second_bytes = second.archive.read_bytes()
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).digest() == hashlib.sha256(second_bytes).digest()


def test_partial_archive_can_resume_after_directory_promotion(tmp_path: Path) -> None:
    output_base = tmp_path / "partial"

    def interrupt(event: str) -> None:
        if event == "ARCHIVE_after_promotion_intent":
            raise V3PublicationCrash(event)

    with pytest.raises(V3PublicationCrash):
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=interrupt
        )
    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
    assert directory.is_dir()
    assert not archive.exists()
    resumed = resume_v3_archive_from_directory(directory, archive)
    assert resumed == archive
    assert archive.is_file()


def test_archive_validation_failure_happens_before_public_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import radjax_tome.tome.artifact_v3 as artifact_v3

    real_validate = artifact_v3.validate_tome_artifact_v3

    def fail_archive(path: Path, *args: object, **kwargs: object):
        if Path(path).suffix == ".tgz":
            raise RuntimeError("injected archive validation failure")
        return real_validate(path, *args, **kwargs)

    monkeypatch.setattr(artifact_v3, "validate_tome_artifact_v3", fail_archive)
    output_base = tmp_path / "invalid-archive"
    with pytest.raises(V3ArchivePublicationError):
        publish_v3_from_handoff(_publication_handoff(), output_base)
    assert output_base.with_name(output_base.name + ".v3").is_dir()
    assert not output_base.with_name(output_base.name + ".v3.tgz").exists()


def _crashed_archive_publication(
    tmp_path: Path, name: str = "guarded"
) -> tuple[Path, Path]:
    output_base = tmp_path / name

    def interrupt(event: str) -> None:
        if event == "ARCHIVE_after_promotion_intent":
            raise V3PublicationCrash(event)

    with pytest.raises(V3PublicationCrash):
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=interrupt
        )
    return (
        output_base.with_name(output_base.name + ".v3"),
        output_base.with_name(output_base.name + ".v3.tgz"),
    )


def _archive_journal_root(directory: Path) -> Path:
    roots = sorted(directory.parent.glob(f".{directory.name[:-3]}.v3-journal-*"))
    assert len(roots) == 1
    return roots[0]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("semantic_authority_identity", "sha256:" + "0" * 64),
        ("configuration_identity", "sha256:" + "1" * 64),
        ("semantic_root", "sha256:" + "2" * 64),
        ("transaction_id", "wrong-transaction"),
    ),
    ids=("authority", "configuration", "semantic-root", "transaction"),
)
def test_archive_resume_validates_existing_binding_before_writing(
    tmp_path: Path, field: str, value: str
) -> None:
    directory, archive = _crashed_archive_publication(tmp_path, f"mutate-{field}")
    journal = _archive_journal_root(directory) / "archive-journal.json"
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["binding"][field] = value
    journal.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = journal.read_bytes()
    with pytest.raises(ValueError):
        resume_v3_archive_from_directory(directory, archive)
    assert journal.read_bytes() == before
    assert directory.is_dir()
    assert not archive.exists()


@pytest.mark.parametrize("mutation", ("malformed", "missing-receipt"))
def test_archive_resume_rejects_malformed_or_unsealed_private_state(
    tmp_path: Path, mutation: str
) -> None:
    directory, archive = _crashed_archive_publication(tmp_path, mutation)
    root = _archive_journal_root(directory)
    journal = root / "archive-journal.json"
    if mutation == "malformed":
        journal.write_text("{", encoding="utf-8")
    else:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["sealed_shards"] = []
        journal.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError):
        resume_v3_archive_from_directory(directory, archive)
    assert directory.is_dir()
    assert not archive.exists()


def test_archive_resume_rejects_mixed_run_candidate_before_selection(
    tmp_path: Path,
) -> None:
    directory, archive = _crashed_archive_publication(tmp_path, "mixed")
    root = _archive_journal_root(directory)
    duplicate = root.with_name(root.name + "-other")
    shutil.copytree(root, duplicate)
    with pytest.raises(ValueError, match="multiple_journals"):
        resume_v3_archive_from_directory(directory, archive)
    assert not archive.exists()
    assert root.exists() and duplicate.exists()


def test_completion_marker_recovery_cleans_private_state_and_is_idempotent(
    tmp_path: Path,
) -> None:
    output_base = tmp_path / "completion"

    def interrupt(event: str) -> None:
        if event == "ARCHIVE_after_completion_marker":
            raise V3PublicationCrash(event)

    with pytest.raises(V3PublicationCrash):
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=interrupt
        )
    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
    assert directory.is_dir() and archive.is_file()
    assert _archive_journal_root(directory).exists()
    resumed = resume_v3_archive_from_directory(directory, archive)
    assert resumed == archive
    assert not list(tmp_path.glob(f".{output_base.name}.v3-*"))
    assert not list(tmp_path.glob(f".{output_base.name}.v3-journal-*"))
    before = archive.stat().st_mtime_ns
    assert resume_v3_archive_from_directory(directory, archive) == archive
    assert archive.stat().st_mtime_ns == before


def test_interrupted_cleanup_is_resumable(tmp_path: Path) -> None:
    output_base = tmp_path / "cleanup"

    def interrupt(event: str) -> None:
        if event == "CLEANUP_after_staging_removed":
            raise V3PublicationCrash(event)

    with pytest.raises(V3PublicationCrash):
        publish_v3_from_handoff(
            _publication_handoff(), output_base, publication_hook=interrupt
        )
    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
    assert directory.is_dir() and archive.is_file()
    assert list(tmp_path.glob(f".{output_base.name}.v3-journal-*"))
    resume_v3_archive_from_directory(directory, archive)
    assert not list(tmp_path.glob(f".{output_base.name}.v3-journal-*"))
    assert not list(tmp_path.glob(f".{output_base.name}.v3-*"))


def test_conflicting_existing_archive_does_not_damage_directory(tmp_path: Path) -> None:
    directory, archive = _crashed_archive_publication(tmp_path, "conflict")
    other_handoff = _publication_handoff()
    other_authority = json.loads(json.dumps(other_handoff.authority))
    other_authority["entries"][0]["identity"] = "sha256:" + "f" * 64
    other_handoff = FinalizedV3Handoff(
        other_handoff.records,
        other_authority,
        other_handoff.policy,
        other_handoff.selection_indexes,
        other_handoff.shard_capacity,
    )
    other = publish_v3_from_handoff(other_handoff, tmp_path / "other")
    shutil.copyfile(other.archive, archive)
    with pytest.raises(ValueError, match="conflicting_existing_archive"):
        resume_v3_archive_from_directory(directory, archive)
    assert directory.is_dir()
    assert archive.read_bytes() == other.archive.read_bytes()


def test_existing_target_is_never_replaced(tmp_path: Path) -> None:
    output_base = tmp_path / "existing"
    directory = output_base.with_name(output_base.name + ".v3")
    directory.mkdir()
    marker = directory / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="target_exists"):
        publish_v3_from_handoff(_publication_handoff(), output_base)
    assert marker.read_text(encoding="utf-8") == "keep"
