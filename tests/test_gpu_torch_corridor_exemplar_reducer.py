from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.backends.cpu import _corridor_exemplar_payload


def _config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "gpu_torch",
        "runtime_mode": "cpu_gpu",
        "target_policy": "corridor_exemplar_v1",
        "sequence_length": 3,
        "batch_size": 2,
        "vocab_size": 6,
        "top_k": 3,
        "num_buckets": 2,
        "exemplar_top_n": 2,
        "dynamic_top_k_min": 2,
        "dynamic_top_k_max": 4,
        "dynamic_mass_threshold": 0.75,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def test_tensor_to_numpy_casts_bfloat16_floats_before_numpy_transfer() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    tensor = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)

    converted = gpu_torch._tensor_to_numpy(tensor, np.float32)

    assert converted.dtype == np.dtype("float32")
    np.testing.assert_allclose(converted, np.asarray([1.25, -2.5], dtype=np.float32))


def test_tensor_to_numpy_casts_floating_dtype_before_numpy_without_torch() -> None:
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    tensor = _NumpyRejectingFloatTensor()

    converted = gpu_torch._tensor_to_numpy(tensor, np.float32)

    assert tensor.float_called is True
    assert converted.dtype == np.dtype("float32")
    np.testing.assert_allclose(converted, np.asarray([1.25], dtype=np.float32))


@pytest.mark.parametrize(
    ("source_policy", "policy_id", "kind", "bucketed", "dynamic"),
    (
        ("dense_logits", 1, "full_resolution", False, False),
        ("cascaded_soft_labels_v1", 2, "fixed_cascaded", True, False),
        ("dynamic_cascaded_soft_labels_v1", 3, "dynamic_cascaded", True, True),
    ),
)
def test_gpu_corridor_exemplar_helper_matches_cpu_production_contract(
    source_policy: str,
    policy_id: int,
    kind: str,
    bucketed: bool,
    dynamic: bool,
) -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    config = _config(exemplar_source_policy=source_policy)
    logits_np = np.asarray(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                [0.5, 2.0, -0.5, -1.5, -2.0, -3.0],
                [1.0, 0.0, 3.0, -2.0, -3.0, -4.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 4.0, 1.0, -1.0, -2.0, -3.0],
                [1.0, 2.0, 4.0, 0.0, -1.0, -2.0],
            ],
        ],
        dtype=np.float32,
    )
    logits = torch.tensor(logits_np, dtype=torch.float32)

    gpu_payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_corridor_exemplar_reduce(torch, logits, config=config)
    )
    gpu_payload.update(
        gpu_torch._gpu_corridor_payload_records(
            config,
            gpu_payload,
            effective_vocab_size=logits_np.shape[-1],
        )
    )
    cpu_payload = _corridor_exemplar_payload(logits_np, config=config)

    assert set(gpu_payload) == {
        "corridor_records",
        "corridor_summary",
        "exemplar_records",
        "exemplar_summary",
        "mode_records",
        "source_policy_summary",
        "schema_metadata",
        "corridor_top_token_ids",
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "exemplar_positions",
        "exemplar_scores",
        "exemplar_selection_mask",
        "exemplar_source_policy_ids",
        "exemplar_source_top_token_ids",
        "exemplar_source_top_log_probs",
        "exemplar_source_top_probs",
        "exemplar_source_top_selection_mask",
        "exemplar_source_effective_top_k",
        "exemplar_source_top_mass",
        "exemplar_source_tail_mass",
        "exemplar_source_bucket_masses",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    }
    assert gpu_payload["corridor_top_token_ids"].shape == (2, 3)
    assert gpu_payload["corridor_top_probs"].shape == (2, 3)
    assert gpu_payload["corridor_teacher_entropy"].shape == (2, 3)
    assert gpu_payload["corridor_confidence"].shape == (2, 3)
    assert gpu_payload["corridor_lengths"].shape == (2,)
    assert gpu_payload["exemplar_positions"].shape == (2, 2)
    assert gpu_payload["exemplar_scores"].shape == (2, 2)
    assert gpu_payload["exemplar_selection_mask"].shape == (2, 3)
    assert gpu_payload["exemplar_source_policy_ids"].shape == (2, 3)
    assert gpu_payload["exemplar_source_top_token_ids"].shape[:2] == (2, 3)
    assert gpu_payload["exemplar_source_top_log_probs"].shape[:2] == (2, 3)
    assert gpu_payload["exemplar_source_top_probs"].shape[:2] == (2, 3)
    assert gpu_payload["exemplar_source_top_selection_mask"].shape[:2] == (2, 3)
    assert gpu_payload["exemplar_source_effective_top_k"].shape == (2, 3)
    assert gpu_payload["exemplar_source_top_mass"].shape == (2, 3)
    assert gpu_payload["exemplar_source_tail_mass"].shape == (2, 3)
    assert gpu_payload["exemplar_source_bucket_masses"].shape == (
        2,
        3,
        config.num_buckets,
    )
    assert gpu_payload["score_max_entropy"].shape == (2,)
    assert gpu_payload["score_selected_position"].shape == (2,)
    assert (gpu_payload["exemplar_source_policy_ids"] == policy_id).all()
    assert (gpu_payload["score_source_policy_ids"] == policy_id).all()
    assert np.isfinite(gpu_payload["corridor_top_probs"]).all()
    assert np.isfinite(gpu_payload["corridor_teacher_entropy"]).all()
    assert np.isfinite(gpu_payload["corridor_confidence"]).all()
    assert np.isfinite(gpu_payload["exemplar_scores"]).all()
    assert np.isfinite(gpu_payload["exemplar_source_top_mass"]).all()
    assert np.isfinite(gpu_payload["exemplar_source_tail_mass"]).all()

    np.testing.assert_array_equal(
        gpu_payload["corridor_top_token_ids"],
        cpu_payload["corridor_top_token_ids"],
    )
    np.testing.assert_array_equal(
        gpu_payload["corridor_lengths"],
        cpu_payload["corridor_lengths"],
    )
    np.testing.assert_array_equal(
        gpu_payload["exemplar_positions"],
        cpu_payload["exemplar_positions"],
    )
    np.testing.assert_array_equal(
        gpu_payload["exemplar_selection_mask"],
        cpu_payload["exemplar_selection_mask"],
    )
    for field in (
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "exemplar_scores",
        "exemplar_source_top_log_probs",
        "exemplar_source_top_probs",
        "exemplar_source_top_mass",
        "exemplar_source_tail_mass",
        "exemplar_source_bucket_masses",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
    ):
        np.testing.assert_allclose(
            gpu_payload[field],
            cpu_payload[field],
            rtol=1e-6,
            atol=1e-6,
        )
    np.testing.assert_array_equal(
        gpu_payload["exemplar_source_effective_top_k"],
        cpu_payload["exemplar_source_effective_top_k"],
    )
    np.testing.assert_array_equal(
        gpu_payload["exemplar_source_top_token_ids"],
        cpu_payload["exemplar_source_top_token_ids"],
    )
    np.testing.assert_array_equal(
        gpu_payload["exemplar_source_top_selection_mask"],
        cpu_payload["exemplar_source_top_selection_mask"],
    )
    for field in (
        "score_example_ids",
        "score_selected_position",
        "score_top_token_id",
        "score_source_policy_ids",
        "score_lengths",
    ):
        np.testing.assert_array_equal(gpu_payload[field], cpu_payload[field])
    for row, positions in enumerate(gpu_payload["exemplar_positions"]):
        assert (gpu_payload["exemplar_selection_mask"][row, positions] == 1).all()

    summary = gpu_payload["source_policy_summary"]
    assert summary["exemplar_source_policy"] == source_policy
    assert summary["source_policy_kind"] == kind
    assert summary["source_policy_uses_bucketed_tail"] is bucketed
    assert summary["source_policy_dynamic_top_k"] is dynamic
    assert gpu_payload["schema_metadata"]["production_corridor_schema"] is True
    assert gpu_payload["schema_metadata"]["historical_parity_claimed"] is False
    assert (
        gpu_payload["schema_metadata"]["historical_reference_source"]
        == "gpu_torch_production"
    )
    assert gpu_payload["schema_metadata"]["exemplar_capture_mode_requested"] == (
        "one_pass_candidate"
    )
    assert gpu_payload["schema_metadata"]["exemplar_capture_mode_effective"] == (
        "one_pass_candidate"
    )
    assert gpu_payload["schema_metadata"]["exemplar_capture_mode_policy"] == (
        "explicit_one_pass_candidate_v1"
    )
    assert gpu_payload["schema_metadata"]["exemplar_candidate_scope"] == (
        "batch_all_examples"
    )
    assert gpu_payload["schema_metadata"]["corpus_level_exemplar_finalization"] is False
    assert (
        gpu_payload["schema_metadata"]["requires_second_pass_for_final_exemplars"]
        is False
    )


def test_gpu_corridor_two_pass_score_helper_is_batch_scale() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    config = _config(exemplar_capture_mode="two_pass_sparse_exemplar")
    logits_np = np.asarray(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                [0.5, 2.0, -0.5, -1.5, -2.0, -3.0],
                [1.0, 0.0, 3.0, -2.0, -3.0, -4.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 4.0, 1.0, -1.0, -2.0, -3.0],
                [1.0, 2.0, 4.0, 0.0, -1.0, -2.0],
            ],
        ],
        dtype=np.float32,
    )
    logits = torch.tensor(logits_np, dtype=torch.float32)

    payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_corridor_exemplar_score_reduce(torch, logits, config=config)
    )

    assert set(payload) == {
        "corridor_top_token_ids",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    }
    assert "corridor_top_probs" not in payload
    assert "exemplar_source_top_token_ids" not in payload
    assert "exemplar_source_bucket_masses" not in payload
    for field in (
        "corridor_lengths",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    ):
        assert payload[field].shape == (2,)
    for field in (
        "corridor_top_token_ids",
        "corridor_teacher_entropy",
        "corridor_confidence",
    ):
        assert payload[field].shape == (2, 3)
    assert (payload["score_source_policy_ids"] == 3).all()
    assert (payload["score_lengths"] == 3).all()
    assert np.isfinite(payload["score_max_entropy"]).all()
    assert np.isfinite(payload["score_mean_entropy"]).all()
    assert np.isfinite(payload["score_confidence_at_selected_position"]).all()

    selected = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_corridor_exemplar_selected_reduce(torch, logits, config=config)
    )
    assert selected["corridor_teacher_entropy"].shape == (2, 3)
    assert selected["exemplar_positions"].shape == (2, 2)


def test_compact_payload_to_numpy_keeps_mixed_corridor_source_and_score_fields() -> (
    None
):
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    payload = {
        "corridor_top_token_ids": _ArrayTensor(np.zeros((2, 3), dtype=np.int32)),
        "corridor_top_probs": _ArrayTensor(np.ones((2, 3), dtype=np.float32)),
        "corridor_teacher_entropy": _ArrayTensor(np.ones((2, 3), dtype=np.float32)),
        "corridor_confidence": _ArrayTensor(np.ones((2, 3), dtype=np.float32)),
        "corridor_lengths": _ArrayTensor(np.full(2, 3, dtype=np.int32)),
        "exemplar_positions": _ArrayTensor(np.zeros((2, 2), dtype=np.int32)),
        "exemplar_scores": _ArrayTensor(np.ones((2, 2), dtype=np.float32)),
        "exemplar_selection_mask": _ArrayTensor(np.ones((2, 3), dtype=np.int32)),
        "exemplar_source_policy_ids": _ArrayTensor(np.full((2, 3), 3, dtype=np.int32)),
        "exemplar_source_top_token_ids": _ArrayTensor(
            np.zeros((2, 3, 4), dtype=np.int32)
        ),
        "exemplar_source_top_log_probs": _ArrayTensor(
            np.ones((2, 3, 4), dtype=np.float32)
        ),
        "exemplar_source_top_probs": _ArrayTensor(np.ones((2, 3, 4), dtype=np.float32)),
        "exemplar_source_top_selection_mask": _ArrayTensor(
            np.ones((2, 3, 4), dtype=np.bool_)
        ),
        "exemplar_source_effective_top_k": _ArrayTensor(
            np.full((2, 3), 4, dtype=np.int32)
        ),
        "exemplar_source_top_mass": _ArrayTensor(np.ones((2, 3), dtype=np.float32)),
        "exemplar_source_tail_mass": _ArrayTensor(np.zeros((2, 3), dtype=np.float32)),
        "exemplar_source_bucket_masses": _ArrayTensor(
            np.zeros((2, 3, 2), dtype=np.float32)
        ),
        "score_example_ids": _ArrayTensor(np.arange(2, dtype=np.int32)),
        "score_max_entropy": _ArrayTensor(np.ones(2, dtype=np.float32)),
        "score_mean_entropy": _ArrayTensor(np.ones(2, dtype=np.float32)),
        "score_selected_position": _ArrayTensor(np.zeros(2, dtype=np.int32)),
        "score_top_token_id": _ArrayTensor(np.zeros(2, dtype=np.int32)),
        "score_selected_position_entropy": _ArrayTensor(np.ones(2, dtype=np.float32)),
        "score_confidence_at_selected_position": _ArrayTensor(
            np.ones(2, dtype=np.float32)
        ),
        "score_source_policy_ids": _ArrayTensor(np.full(2, 3, dtype=np.int32)),
        "score_lengths": _ArrayTensor(np.full(2, 3, dtype=np.int32)),
    }

    compact = gpu_torch._compact_payload_to_numpy(payload)

    assert "corridor_top_token_ids" in compact
    assert "exemplar_source_top_probs" in compact
    assert "score_example_ids" in compact
    assert compact["exemplar_source_top_probs"].shape == (2, 3, 4)
    assert compact["score_example_ids"].shape == (2,)


def _optional_torch_available() -> bool:
    try:
        import_module("torch")
    except ImportError:
        return False
    return True


class _ArrayTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _ArrayTensor:
        return self

    def float(self) -> _ArrayTensor:
        return _ArrayTensor(self._array.astype(np.float32))

    def to(self, device: str) -> _ArrayTensor:
        assert device == "cpu"
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _NumpyRejectingFloatTensor:
    def __init__(self) -> None:
        self.float_called = False

    def detach(self) -> _NumpyRejectingFloatTensor:
        return self

    def float(self) -> _NumpyRejectingFloatTensor:
        self.float_called = True
        return self

    def to(self, device: str) -> _NumpyRejectingFloatTensor:
        assert device == "cpu"
        return self

    def numpy(self) -> np.ndarray:
        if not self.float_called:
            raise TypeError("bfloat16 NumPy conversion is unsupported")
        return np.asarray([1.25], dtype=np.float32)
