from __future__ import annotations

from radjax_tome.backends.base import TeacherBackendConfig, TeacherBatchInput
from radjax_tome.backends.cpu import CPUReferenceTeacherEmissionBackend
from radjax_tome.builder.delivery.modes import (
    COMPACT_K_IMMUTABLE_BODY,
    COMPACT_K_MONOLITHIC,
    LEGACY_PADDED_MONOLITHIC,
    collate_compact_logical_records,
    compact_payload_for_storage,
    validate_materialization_mode,
)


def _emit(mode: str):
    config = TeacherBackendConfig(
        backend_id="cpu_reference",
        target_policy="dynamic_cascaded_soft_labels_v1",
        sequence_length=4,
        vocab_size=64,
        dynamic_top_k_min=1,
        dynamic_top_k_max=38,
        batch_size=1,
        representation_mode=mode,
    )
    return CPUReferenceTeacherEmissionBackend(config).emit_batch(
        TeacherBatchInput(("example-1",), ("hello world",))
    )


def test_legacy_and_compact_modes_are_explicit() -> None:
    legacy = _emit(LEGACY_PADDED_MONOLITHIC)
    compact = _emit(COMPACT_K_MONOLITHIC)
    immutable = _emit(COMPACT_K_IMMUTABLE_BODY)
    assert len(legacy.payload["top_token_ids"][0, 0]) == 38
    assert "top_selection_mask" in legacy.payload
    assert len(compact.payload["top_token_ids"][0, 0]) == 38
    assert "top_selection_mask" not in compact.payload
    assert len(immutable.payload["top_token_ids"][0, 0]) == 38
    assert "top_selection_mask" not in immutable.payload
    assert (
        compact.payload["top_token_ids"][0, 0].tolist()
        == immutable.payload["top_token_ids"][0, 0].tolist()
    )


def test_invalid_mode_fails_closed() -> None:
    for mode in (
        None,
        LEGACY_PADDED_MONOLITHIC,
        COMPACT_K_MONOLITHIC,
        COMPACT_K_IMMUTABLE_BODY,
    ):
        assert validate_materialization_mode(mode) in {
            LEGACY_PADDED_MONOLITHIC,
            COMPACT_K_MONOLITHIC,
            COMPACT_K_IMMUTABLE_BODY,
        }
    try:
        validate_materialization_mode("silent_fallback")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("invalid materialization mode was accepted")


def test_compact_storage_slices_padded_payload_and_drops_mask() -> None:
    payload = {
        "effective_top_k": 2,
        "top_token_ids": [7, 8, 0, 0],
        "top_probs": [0.6, 0.3, 0.0, 0.0],
        "top_log_probs": [-0.5, -1.2, 0.0, 0.0],
        "top_selection_mask": [True, True, False, False],
    }
    compact = compact_payload_for_storage(payload)
    assert compact["top_token_ids"] == [7, 8]
    assert compact["top_probs"] == [0.6, 0.3]
    assert compact["top_log_probs"] == [-0.5, -1.2]
    assert "top_selection_mask" not in compact


def test_mixed_k_collation_only_pads_to_batch_maximum() -> None:
    records = [
        {
            "effective_top_k": 2,
            "top_token_ids": [1, 2],
            "top_probs": [0.6, 0.3],
            "top_log_probs": [-0.5, -1.2],
        },
        {
            "effective_top_k": 4,
            "top_token_ids": [1, 2, 3, 4],
            "top_probs": [0.4, 0.3, 0.2, 0.1],
            "top_log_probs": [-0.9, -1.1, -1.6, -2.3],
        },
    ]
    batch = collate_compact_logical_records(records)
    assert batch["width"] == 4
    assert batch["mask"] == [[True, True, False, False], [True] * 4]
