from __future__ import annotations

import numpy as np
import pytest

from radjax_tome.backends import (
    CPUReferenceTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    list_backend_capabilities,
)


def _batch() -> TeacherBatchInput:
    return TeacherBatchInput(
        example_ids=("ex-1", "ex-2"),
        texts=("alpha", "beta"),
    )


def _config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "cpu_reference",
        "runtime_mode": "cpu",
        "cpu_orchestration_mode": "auto",
        "target_policy": "dense_logits",
        "sequence_length": 5,
        "batch_size": 99,
        "vocab_size": 11,
        "top_k": 4,
        "num_buckets": 3,
        "exemplar_top_n": 2,
        "dynamic_top_k_min": 2,
        "dynamic_top_k_max": 5,
        "dynamic_mass_threshold": 0.75,
        "dynamic_top_k_policy": "mass_threshold_v1",
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def test_cpu_reference_backend_created_through_registry() -> None:
    backend = create_backend(_config())

    assert backend.backend_id == "cpu_reference"
    assert backend.backend_family == "cpu_reference"
    assert backend.runtime_mode == "cpu"


def test_registry_lists_fake_and_cpu_reference_backends() -> None:
    backend_ids = [capability.backend_id for capability in list_backend_capabilities()]

    assert "fake_numpy" in backend_ids
    assert "cpu_reference" in backend_ids
    assert backend_ids == sorted(backend_ids)


def test_dense_logits_payload_shape_and_determinism() -> None:
    backend = create_backend(_config(target_policy="dense_logits"))

    first = backend.emit_batch(_batch())
    second = backend.emit_batch(_batch())

    assert first.payload["logits"].shape == (2, 5, 11)
    np.testing.assert_array_equal(first.input_ids, second.input_ids)
    np.testing.assert_array_equal(first.attention_mask, second.attention_mask)
    np.testing.assert_array_equal(first.payload["logits"], second.payload["logits"])
    assert first.metadata["capability_status"] == "supported_debug"
    assert first.metadata["configured_batch_size"] == 99
    assert first.metadata["actual_batch_size"] == 2


def test_topk_payload_fields_shapes_and_mass_accounting() -> None:
    backend = create_backend(_config(target_policy="topk_with_tail_v0", top_k=4))

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    } <= set(payload)
    assert payload["top_token_ids"].shape == (2, 5, 4)
    assert payload["top_log_probs"].shape == (2, 5, 4)
    assert payload["top_probs"].shape == (2, 5, 4)
    assert payload["top_mass"].shape == (2, 5)
    assert payload["tail_mass"].shape == (2, 5)
    assert payload["teacher_entropy"].shape == (2, 5)
    assert np.isfinite(payload["top_log_probs"]).all()
    assert np.isfinite(payload["top_probs"]).all()
    assert np.isfinite(payload["tail_mass"]).all()
    assert np.isfinite(payload["teacher_entropy"]).all()
    np.testing.assert_allclose(
        payload["tail_mass"],
        1.0 - payload["top_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.metadata["capability_status"] == "supported"
    assert result.metadata["requested_top_k"] == 4
    assert result.metadata["effective_top_k"] == 4


def test_topk_effective_top_k_clamps_to_vocab_size() -> None:
    backend = create_backend(
        _config(target_policy="topk_with_tail_v0", top_k=99, vocab_size=7)
    )

    result = backend.emit_batch(_batch())

    assert result.payload["top_token_ids"].shape == (2, 5, 7)
    assert result.metadata["requested_top_k"] == 99
    assert result.metadata["effective_top_k"] == 7


def test_cascaded_payload_shapes_and_bucket_mass_accounting() -> None:
    backend = create_backend(
        _config(target_policy="cascaded_soft_labels_v1", top_k=4, num_buckets=3)
    )

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    } <= set(payload)
    assert payload["bucket_masses"].shape == (2, 5, 3)
    assert np.isfinite(payload["bucket_masses"]).all()
    np.testing.assert_allclose(
        np.sum(payload["bucket_masses"], axis=-1),
        payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.metadata["capability_status"] == "supported"
    assert result.metadata["num_buckets"] == 3
    assert (
        result.metadata["bucket_policy"]
        == "contiguous_descending_tail_probability_mass"
    )


def test_dynamic_cascaded_payload_shapes_mask_and_mass_accounting() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=2,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.75,
            num_buckets=3,
        )
    )

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    } <= set(payload)
    assert payload["top_token_ids"].shape == (2, 5, 5)
    assert payload["top_log_probs"].shape == (2, 5, 5)
    assert payload["top_probs"].shape == (2, 5, 5)
    assert payload["top_selection_mask"].shape == (2, 5, 5)
    assert payload["top_selection_mask"].dtype == np.bool_
    assert payload["effective_top_k"].shape == (2, 5)
    assert np.issubdtype(payload["effective_top_k"].dtype, np.integer)
    assert payload["top_mass"].shape == (2, 5)
    assert payload["tail_mass"].shape == (2, 5)
    assert payload["bucket_masses"].shape == (2, 5, 3)
    assert payload["teacher_entropy"].shape == (2, 5)
    assert np.isfinite(payload["teacher_entropy"]).all()

    mask = payload["top_selection_mask"]
    np.testing.assert_array_equal(
        payload["effective_top_k"],
        np.sum(mask, axis=-1).astype(np.int32),
    )
    assert int(np.min(payload["effective_top_k"])) >= 2
    assert int(np.max(payload["effective_top_k"])) <= 5
    assert (payload["top_token_ids"][~mask] == 0).all()
    assert (payload["top_probs"][~mask] == 0.0).all()
    assert (payload["top_log_probs"][~mask] == 0.0).all()
    np.testing.assert_allclose(
        np.sum(payload["bucket_masses"], axis=-1),
        payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["top_mass"] + payload["tail_mass"],
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )

    not_max_clamped = payload["effective_top_k"] < 5
    assert (payload["top_mass"][not_max_clamped] >= 0.75).all()
    below_threshold = payload["top_mass"] < 0.75
    assert (payload["effective_top_k"][below_threshold] == 5).all()

    metadata = result.metadata
    assert metadata["capability_status"] == "supported"
    assert metadata["dynamic_top_k_policy"] == "mass_threshold_v1"
    assert metadata["dynamic_top_k_min_configured"] == 2
    assert metadata["dynamic_top_k_max_configured"] == 5
    assert metadata["dynamic_top_k_min_effective"] == 2
    assert metadata["dynamic_top_k_max_effective"] == 5
    assert metadata["dynamic_mass_threshold"] == 0.75
    assert metadata["num_buckets"] == 3
    assert metadata["bucket_policy"] == "contiguous_descending_tail_probability_mass"
    assert metadata["padding_policy"] == "pad_to_dynamic_top_k_max_effective"
    assert metadata["selection_mask_field"] == "top_selection_mask"
    assert "must be ignored" in metadata["masked_slot_policy"]
    assert metadata["top_selection_mask_semantics"] == (
        "true means explicit selected token"
    )
    assert metadata["effective_top_k_min_observed"] == int(
        np.min(payload["effective_top_k"])
    )
    assert metadata["effective_top_k_max_observed"] == int(
        np.max(payload["effective_top_k"])
    )
    assert metadata["effective_top_k_mean_observed"] == pytest.approx(
        float(np.mean(payload["effective_top_k"], dtype=np.float32))
    )


def test_dynamic_cascaded_min_k_is_honored_when_threshold_is_low() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=3,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.01,
        )
    )

    payload = backend.emit_batch(_batch()).payload

    assert (payload["effective_top_k"] >= 3).all()
    assert (payload["effective_top_k"] == 3).any()


def test_dynamic_cascaded_max_k_clamps_to_vocab_size() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            vocab_size=4,
            dynamic_top_k_min=2,
            dynamic_top_k_max=99,
            dynamic_mass_threshold=1.0,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.payload["top_token_ids"].shape == (2, 5, 4)
    assert (result.payload["effective_top_k"] == 4).all()
    assert result.metadata["dynamic_top_k_max_configured"] == 99
    assert result.metadata["dynamic_top_k_max_effective"] == 4
    assert result.metadata["dynamic_top_k_min_effective"] == 2


def test_corridor_exemplar_payload_and_metadata_are_deterministic() -> None:
    backend = create_backend(
        _config(target_policy="corridor_exemplar_v1", exemplar_top_n=2)
    )

    first = backend.emit_batch(_batch())
    second = backend.emit_batch(_batch())
    payload = first.payload

    assert first.target_policy == "corridor_exemplar_v1"
    assert {
        "corridor_records",
        "corridor_summary",
        "exemplar_records",
        "exemplar_summary",
        "mode_records",
        "corridor_top_token_ids",
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "exemplar_positions",
        "exemplar_scores",
        "exemplar_selection_mask",
    } <= set(payload)
    assert payload["corridor_records"]["record_count"] == 2
    assert payload["corridor_summary"]["record_count"] == 2
    assert payload["exemplar_records"]["record_count"] == 4
    assert payload["exemplar_summary"]["positions_per_example"] == 2
    assert payload["corridor_top_token_ids"].shape == (2, 5)
    assert payload["corridor_top_probs"].shape == (2, 5)
    assert payload["corridor_teacher_entropy"].shape == (2, 5)
    assert payload["corridor_confidence"].shape == (2, 5)
    assert payload["corridor_lengths"].shape == (2,)
    assert payload["exemplar_positions"].shape == (2, 2)
    assert payload["exemplar_scores"].shape == (2, 2)
    assert payload["exemplar_selection_mask"].shape == (2, 5)
    assert np.count_nonzero(payload["exemplar_selection_mask"]) == 4
    assert np.isfinite(payload["corridor_top_probs"]).all()
    assert np.isfinite(payload["corridor_teacher_entropy"]).all()
    assert np.isfinite(payload["exemplar_scores"]).all()

    for key in (
        "mode_records",
        "corridor_top_token_ids",
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "exemplar_positions",
        "exemplar_scores",
        "exemplar_selection_mask",
    ):
        np.testing.assert_array_equal(first.payload[key], second.payload[key])
    assert first.payload["corridor_records"] == second.payload["corridor_records"]
    assert first.payload["corridor_summary"] == second.payload["corridor_summary"]
    assert first.payload["exemplar_records"] == second.payload["exemplar_records"]
    assert first.payload["exemplar_summary"] == second.payload["exemplar_summary"]

    assert first.metadata["effective_runtime_mode"] == "cpu"
    assert first.metadata["effective_cpu_orchestration_mode"] == "serial"
    assert first.metadata["optimized_path_used"] is False
    assert first.metadata["fallback_used"] is False
    assert first.metadata["capability_status"] == "supported"
    assert first.metadata["corridor_policy"] == "deterministic_reference_corridor_v1"
    assert (
        first.metadata["exemplar_selection_policy"]
        == "deterministic_high_entropy_top_n_v1"
    )
    assert first.metadata["exemplar_records"] == 2


def test_cpu_reference_metadata_records_requested_and_effective_values() -> None:
    backend = create_backend(
        _config(
            cpu_orchestration_mode="staged",
            target_policy="topk_with_tail_v0",
        )
    )

    metadata = backend.emit_batch(_batch()).metadata

    assert metadata["requested_runtime_mode"] == "cpu"
    assert metadata["effective_runtime_mode"] == "cpu"
    assert metadata["requested_cpu_orchestration_mode"] == "staged"
    assert metadata["effective_cpu_orchestration_mode"] == "serial"
    assert metadata["requested_target_policy"] == "topk_with_tail_v0"
    assert metadata["effective_target_policy"] == "topk_with_tail_v0"
    assert metadata["backend_id"] == "cpu_reference"
    assert metadata["backend_family"] == "cpu_reference"
    assert metadata["runtime_kind"] == "cpu_reference"
    assert metadata["device_kind"] == "cpu"
    assert metadata["optimized_path_used"] is False
    assert metadata["fallback_used"] is False
    assert metadata["sequence_length"] == 5
    assert metadata["batch_size"] == 99
    assert metadata["vocab_size"] == 11


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("backend_id", "fake_numpy", "backend_id"),
        ("runtime_mode", "cpu_gpu", "runtime_mode"),
        ("sequence_length", 0, "sequence_length"),
        ("batch_size", 0, "batch_size"),
        ("vocab_size", 0, "vocab_size"),
        ("top_k", 0, "top_k"),
        ("num_buckets", 0, "num_buckets"),
        ("exemplar_top_n", 0, "exemplar_top_n"),
        ("dynamic_top_k_min", 0, "dynamic_top_k_min"),
        ("dynamic_top_k_max", 1, "dynamic_top_k_max"),
        ("dynamic_mass_threshold", 0.0, "dynamic_mass_threshold"),
        ("dynamic_mass_threshold", 1.1, "dynamic_mass_threshold"),
        ("dynamic_top_k_policy", "unknown", "dynamic_top_k_policy"),
    ),
)
def test_invalid_cpu_reference_config_values_fail_clearly(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config = _config(**{field: value})
        if field == "backend_id":
            CPUReferenceTeacherEmissionBackend(config)
        else:
            create_backend(config)


def test_cpu_reference_capabilities_mark_corridor_exemplar_supported() -> None:
    capabilities = {
        capability.target_policy: capability
        for capability in create_backend(_config()).capabilities()
    }

    corridor = capabilities["corridor_exemplar_v1"]
    assert corridor.status == "supported"
    assert corridor.implemented_now
    assert not corridor.optimized
    assert "Spec 3.3C.1 adds serial/reference CPU corridor/exemplar support" in (
        corridor.notes
    )


def test_cpu_reference_capabilities_mark_dynamic_cascaded_supported() -> None:
    capabilities = {
        capability.target_policy: capability
        for capability in create_backend(_config()).capabilities()
    }

    dynamic = capabilities["dynamic_cascaded_soft_labels_v1"]
    assert dynamic.status == "supported"
    assert dynamic.implemented_now
    assert not dynamic.optimized
    assert "Spec 3.3F6" in dynamic.notes
    assert "dynamic top-k explicit head plus bucketed-tail" in dynamic.notes
