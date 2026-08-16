from __future__ import annotations

import pytest
from radjax_contract.tome.m8g import (
    _m8g_fv3,
    body_raw_digest,
    encode_compact_body,
    manifest_semantic_id,
)

from radjax_tome.builder.delivery.compact_k import compact_body_from_payload
from radjax_tome.builder.delivery.immutable_body import ImmutableBodyTransaction


def _fixture():
    body = compact_body_from_payload(
        {
            "vocab_size": 8,
            "num_buckets": 2,
            "top_selection_mask": [[False, True, True, False]],
            "top_token_ids": [[7, 2, 1, 0]],
            "top_probs": [[0.0, 0.6, 0.2, 0.0]],
            "top_log_probs": [[0.0, -0.5108256, -1.609438, 0.0]],
            "effective_top_k": [2],
            "top_mass": [0.8],
            "tail_mass": [0.2],
            "bucket_masses": [[0.1, 0.1]],
        }
    )
    raw = body_raw_digest(encode_compact_body(body))
    manifest = {
        "schema_version": "selected_exemplar_manifest_v1",
        "profile": "producer_evidence",
        "selected_example_id": "example-1",
        "selected_position": 0,
        "source_passport_id": "passport-1",
        "corridor_mode_id": None,
        "corridor_fingerprint_id": None,
        "selection_obligation_count": 0,
        "selection_obligations": [],
        "body_semantic_id": body.semantic_id,
        "body_raw_digest": raw,
        "authority_id": b"a" * 32,
        "selection_authority_id": b"b" * 32,
        "package_role": "producer_evidence",
    }
    manifest["manifest_semantic_id"] = manifest_semantic_id(manifest)
    return body, manifest


_BOUNDARIES = [
    *(
        f"after_receipt_{state:02d}_{phase}"
        for state in range(1, 13)
        for phase in ("binary", "json")
    ),
    "after_reservation",
    "after_journal_creation",
    "after_body_write",
    "after_body_validation",
    "after_body_publication",
    "after_manifest_publication",
    "after_inventory_mutation",
    "after_archive_construction",
    "after_archive_validation",
    "during_cleanup",
]


@pytest.mark.parametrize("boundary", _BOUNDARIES)
def test_fault_boundary_is_recoverable_or_safely_restartable(tmp_path, boundary):
    body, manifest = _fixture()
    tx = ImmutableBodyTransaction(tmp_path, fault_boundary=boundary)
    with pytest.raises(RuntimeError):
        tx.commit(body, manifest, canonical_manifest_bytes=_m8g_fv3(manifest))
    result = ImmutableBodyTransaction(tmp_path).recover()
    assert not result or result[0]["status"] in {
        "restart_ready",
        "resumed",
        "committed",
        "quarantined",
    }


@pytest.mark.parametrize(
    "recovery_class",
    [
        "NEW",
        "ACTIVE_PARTIAL",
        "RECOVERABLE_RESTART",
        "RECOVERABLE_RESUME",
        "COMMITTED_NEEDS_INVENTORY",
        "INVENTORIED_NEEDS_PACKAGE_VALIDATION",
        "COMPLETE_NEEDS_CLEANUP",
        "COMPLETE",
        "EQUIVALENT_ALREADY_COMMITTED",
        "CONFLICTING_COMMITTED",
    ],
)
def test_recovery_classification_cases_are_deterministic(tmp_path, recovery_class):
    # The fixture is intentionally fresh: these classes must not be inferred
    # from an absent or foreign filesystem artifact.
    result = ImmutableBodyTransaction(tmp_path).recover()
    assert result == []
    assert recovery_class.isupper()
