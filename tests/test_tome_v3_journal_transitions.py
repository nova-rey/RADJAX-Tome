from __future__ import annotations

import pytest
from radjax_contract.tome.v3.codec import digest
from radjax_contract.tome.v3.issues import TomeV3ValidationError
from radjax_contract.tome.v3.journal import (
    journal_restart_disposition_v3,
    validate_journal_state_v3,
)
from radjax_contract.tome.v3.models import JournalStateV3

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
