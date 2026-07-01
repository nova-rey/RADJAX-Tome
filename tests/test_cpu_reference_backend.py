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


def test_unsupported_target_policy_fails_clearly() -> None:
    with pytest.raises(ValueError, match="supports dense_logits"):
        create_backend(_config(target_policy="corridor_exemplar_v1"))
