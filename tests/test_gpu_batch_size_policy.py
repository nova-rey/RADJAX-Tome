from __future__ import annotations

import numpy as np
import pytest

from radjax_tome.backends import (
    GPUTorchTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    gpu_batch_size_candidates,
    resolve_gpu_batch_size_policy,
)


def _cpu_config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dense_logits",
        "sequence_length": 5,
        "batch_size": 2,
        "vocab_size": 11,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def _gpu_config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "gpu_torch",
        "runtime_mode": "cpu_gpu",
        "target_policy": "dense_logits",
        "model_id": "missing-local-hf-model",
        "tokenizer_id": "missing-local-hf-model",
        "sequence_length": 4,
        "batch_size": 2,
        "vocab_size": 7,
        "local_files_only": True,
        "allow_downloads": False,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def _batch() -> TeacherBatchInput:
    return TeacherBatchInput(
        example_ids=("ex-1", "ex-2"),
        texts=("alpha", "beta"),
    )


@pytest.mark.parametrize("preset", (1, 2, 4, 8, 16, 32, 64))
def test_gpu_batch_size_preset_mode_accepts_allowed_presets(preset: int) -> None:
    config = _cpu_config(gpu_batch_size_preset=preset)

    metadata = resolve_gpu_batch_size_policy(config)

    assert metadata["gpu_batch_size_mode_requested"] == "preset"
    assert metadata["gpu_batch_size_mode_effective"] == "preset"
    assert metadata["requested_gpu_batch_size"] == preset
    assert metadata["effective_gpu_batch_size"] == preset
    assert metadata["gpu_batch_size_manual_override_used"] is False
    assert metadata["gpu_batch_size_candidates_tried"] == ()


def test_gpu_batch_size_invalid_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu_batch_size_preset"):
        create_backend(_cpu_config(gpu_batch_size_preset=3))


def test_gpu_batch_size_custom_accepts_positive_integer() -> None:
    metadata = resolve_gpu_batch_size_policy(
        _cpu_config(
            gpu_batch_size_mode="custom",
            gpu_batch_size_custom=12,
        )
    )

    assert metadata["gpu_batch_size_mode_effective"] == "custom"
    assert metadata["requested_gpu_batch_size"] == 12
    assert metadata["effective_gpu_batch_size"] == 12
    assert metadata["gpu_batch_size_manual_override_used"] is True
    assert metadata["gpu_batch_size_warning_emitted"] is False


def test_gpu_batch_size_custom_rejects_non_positive_integer() -> None:
    with pytest.raises(ValueError, match="gpu_batch_size_custom"):
        create_backend(
            _cpu_config(
                gpu_batch_size_mode="custom",
                gpu_batch_size_custom=0,
            )
        )


def test_gpu_batch_size_custom_over_threshold_warns_but_is_allowed() -> None:
    backend = create_backend(
        _cpu_config(
            gpu_batch_size_mode="custom",
            gpu_batch_size_custom=128,
        )
    )

    metadata = backend.emit_batch(_batch()).metadata

    assert metadata["effective_gpu_batch_size"] == 128
    assert metadata["gpu_batch_size_warning_emitted"] is True
    assert "custom batch size exceeds warning threshold" in str(
        metadata["gpu_batch_size_warning_reason"]
    )
    assert "target policy" in str(metadata["gpu_batch_size_warning_reason"])
    assert "exemplar capture mode" in str(metadata["gpu_batch_size_warning_reason"])


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected"),
    (
        (1, 64, (1, 2, 4, 8, 16, 32, 64)),
        (2, 32, (2, 4, 8, 16, 32)),
        (3, 64, (3, 6, 12, 24, 48, 64)),
    ),
)
def test_gpu_batch_size_auto_candidates_are_exponential(
    minimum: int,
    maximum: int,
    expected: tuple[int, ...],
) -> None:
    assert (
        gpu_batch_size_candidates(
            _cpu_config(
                gpu_batch_size_mode="auto",
                gpu_batch_size_auto_min=minimum,
                gpu_batch_size_auto_max=maximum,
            )
        )
        == expected
    )


def test_gpu_batch_size_auto_without_probe_requires_probe_at_minimum() -> None:
    metadata = resolve_gpu_batch_size_policy(_cpu_config(gpu_batch_size_mode="auto"))

    assert metadata["gpu_batch_size_mode_effective"] == "auto"
    assert metadata["effective_gpu_batch_size"] == 1
    assert metadata["gpu_batch_size_probe_required"] is True
    assert metadata["gpu_batch_size_candidates_tried"] == ()
    assert metadata["gpu_batch_size_auto_failed"] is False


def test_gpu_batch_size_auto_probe_chooses_last_good_batch() -> None:
    metadata = resolve_gpu_batch_size_policy(
        _cpu_config(gpu_batch_size_mode="auto"),
        probe_results=(
            {
                "candidate_batch_size": 8,
                "success": True,
                "time_per_example_seconds": 0.10,
                "write_time_seconds": 0.01,
            },
            {
                "candidate_batch_size": 16,
                "success": True,
                "time_per_example_seconds": 0.11,
                "write_time_seconds": 0.01,
            },
            {
                "candidate_batch_size": 32,
                "success": False,
                "failure_reason": "oom",
                "oom_or_device_failure": True,
            },
        ),
    )

    assert metadata["effective_gpu_batch_size"] == 16
    assert metadata["gpu_batch_size_last_good"] == 16
    assert metadata["gpu_batch_size_failure_at"] == 32
    assert metadata["gpu_batch_size_failure_reason"] == "oom"
    assert metadata["gpu_batch_size_candidates_tried"] == (8, 16, 32)


def test_gpu_batch_size_auto_midpoint_refinement_can_select_midpoint() -> None:
    metadata = resolve_gpu_batch_size_policy(
        _cpu_config(gpu_batch_size_mode="auto"),
        probe_results=(
            {
                "candidate_batch_size": 16,
                "success": True,
                "time_per_example_seconds": 0.10,
                "write_time_seconds": 0.01,
            },
            {
                "candidate_batch_size": 32,
                "success": False,
                "failure_reason": "time_per_example_regression",
            },
            {
                "candidate_batch_size": 24,
                "success": True,
                "time_per_example_seconds": 0.11,
                "write_time_seconds": 0.01,
            },
        ),
    )

    assert metadata["effective_gpu_batch_size"] == 24
    assert metadata["gpu_batch_size_last_good"] == 16
    assert metadata["gpu_batch_size_failure_at"] == 32


def test_gpu_batch_size_auto_first_candidate_failure_is_clear() -> None:
    metadata = resolve_gpu_batch_size_policy(
        _cpu_config(gpu_batch_size_mode="auto", gpu_batch_size_auto_min=4),
        probe_results=(
            {
                "candidate_batch_size": 4,
                "success": False,
                "failure_reason": "device_failure",
                "oom_or_device_failure": True,
            },
        ),
    )

    assert metadata["effective_gpu_batch_size"] == 4
    assert metadata["gpu_batch_size_last_good"] is None
    assert metadata["gpu_batch_size_failure_at"] == 4
    assert metadata["gpu_batch_size_failure_reason"] == "device_failure"
    assert metadata["gpu_batch_size_auto_failed"] is True


def test_cpu_reference_metadata_includes_batch_size_policy_and_measurements() -> None:
    backend = create_backend(_cpu_config())

    result = backend.emit_batch(_batch())
    payload_bytes = int(result.payload["logits"].nbytes)

    assert result.payload["logits"].shape == (2, 5, 11)
    assert result.metadata["effective_gpu_batch_size"] == 8
    assert result.metadata["gpu_batch_size_policy"] == "gpu_batch_size_policy_v1"
    assert result.metadata["measured_output_bytes_available"] is True
    assert result.metadata["measured_output_bytes"] == payload_bytes
    assert result.metadata["measured_compact_bytes_transferred_to_host"] == (
        payload_bytes
    )
    assert result.metadata["estimated_bytes_are_calibrated"] is False
    assert result.metadata["batch_size_policy_uses_estimates"] is True
    assert result.metadata["measured_gpu_peak_memory_available"] is False
    assert result.metadata["measured_gpu_peak_memory_bytes"] is None
    assert result.metadata["multidevice_enabled"] is False
    assert result.metadata["batch_partition_strategy"] == "single_device"


def test_gpu_torch_metadata_includes_batch_size_policy_without_probe() -> None:
    backend = GPUTorchTeacherEmissionBackend(_gpu_config(gpu_batch_size_mode="auto"))

    metadata = backend.metadata(actual_batch_size=2, effective_vocab_size=7)

    assert metadata["gpu_batch_size_mode_requested"] == "auto"
    assert metadata["gpu_batch_size_probe_required"] is True
    assert metadata["effective_gpu_batch_size"] == 1
    assert metadata["gpu_batch_size_target_policy"] == "dense_logits"
    assert metadata["gpu_batch_size_exemplar_source_policy"] == (
        "dynamic_cascaded_soft_labels_v1"
    )
    assert metadata["gpu_batch_size_exemplar_capture_mode"] == "one_pass_candidate"
    assert metadata["measured_output_bytes_available"] is False
    assert metadata["measured_output_bytes"] is None


def test_gpu_batch_size_policy_records_measured_payload_bytes() -> None:
    payload = {
        "top_token_ids": np.zeros((2, 4, 3), dtype=np.int32),
        "top_probs": np.zeros((2, 4, 3), dtype=np.float32),
    }

    metadata = resolve_gpu_batch_size_policy(_gpu_config(), payload=payload)

    assert metadata["measured_output_bytes_available"] is True
    assert metadata["measured_output_bytes"] == sum(
        int(value.nbytes) for value in payload.values()
    )
    assert metadata["estimated_to_measured_bytes_ratio"] is not None
