from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from radjax_tome.backends import (
    GPUTorchTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    check_gpu_torch_backend_available,
    create_backend,
    detect_torch_accelerator,
    list_backend_capabilities,
)
from radjax_tome.backends.gpu_torch import TeacherBackendUnsupportedPolicyError


def _config(**overrides: object) -> TeacherBackendConfig:
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
        texts=("hello", "world"),
    )


def test_backend_and_registry_imports_do_not_import_heavy_optional_deps() -> None:
    script = """
import importlib
import sys

for name in ("torch", "transformers", "jax"):
    sys.modules.pop(name, None)

importlib.import_module("radjax_tome.backends")
importlib.import_module("radjax_tome.backends.registry")

loaded = sorted(
    name for name in ("torch", "transformers", "jax") if name in sys.modules
)
print(",".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_gpu_torch_registered_with_compact_capabilities() -> None:
    capabilities = [
        capability
        for capability in list_backend_capabilities()
        if capability.backend_id == "gpu_torch"
    ]

    assert len(capabilities) == 5
    assert {capability.runtime_mode for capability in capabilities} == {"cpu_gpu"}
    statuses = {
        capability.target_policy: capability.status for capability in capabilities
    }
    implemented = {
        capability.target_policy: capability.implemented_now
        for capability in capabilities
    }
    optimized = {
        capability.target_policy: capability.optimized for capability in capabilities
    }
    assert statuses["dense_logits"] == "supported_debug"
    assert implemented["dense_logits"]
    assert not optimized["dense_logits"]
    assert statuses["topk_with_tail_v0"] == "optimized"
    assert implemented["topk_with_tail_v0"]
    assert optimized["topk_with_tail_v0"]
    assert statuses["cascaded_soft_labels_v1"] == "optimized"
    assert implemented["cascaded_soft_labels_v1"]
    assert optimized["cascaded_soft_labels_v1"]
    assert statuses["dynamic_cascaded_soft_labels_v1"] == "optimized"
    assert implemented["dynamic_cascaded_soft_labels_v1"]
    assert optimized["dynamic_cascaded_soft_labels_v1"]
    assert statuses["corridor_exemplar_v1"] == "historical_reference_exists"
    assert not implemented["corridor_exemplar_v1"]
    assert not optimized["corridor_exemplar_v1"]


def test_gpu_torch_constructs_without_loading_optional_deps() -> None:
    backend = create_backend(_config())

    assert backend.backend_id == "gpu_torch"
    assert backend.backend_family == "gpu_torch"
    assert backend.runtime_mode == "cpu_gpu"


@pytest.mark.parametrize("runtime_mode", ("cpu", "cpu_tpu"))
def test_gpu_torch_rejects_non_gpu_runtime_modes(runtime_mode: str) -> None:
    with pytest.raises(ValueError, match="runtime_mode"):
        GPUTorchTeacherEmissionBackend(_config(runtime_mode=runtime_mode))


@pytest.mark.parametrize("gpu_vocab_chunk_size", (0, -1))
def test_gpu_torch_rejects_invalid_vocab_chunk_size(
    gpu_vocab_chunk_size: int,
) -> None:
    with pytest.raises(ValueError, match="gpu_vocab_chunk_size"):
        GPUTorchTeacherEmissionBackend(
            _config(
                gpu_enable_vocab_chunking=True,
                gpu_vocab_chunk_size=gpu_vocab_chunk_size,
            )
        )


@pytest.mark.parametrize(
    "target_policy",
    (
        "dense_logits",
        "topk_with_tail_v0",
        "cascaded_soft_labels_v1",
        "dynamic_cascaded_soft_labels_v1",
    ),
)
def test_gpu_torch_accepts_f3_supported_policies(target_policy: str) -> None:
    backend = GPUTorchTeacherEmissionBackend(_config(target_policy=target_policy))

    assert backend.config.target_policy == target_policy


def test_gpu_torch_rejects_unimplemented_compact_policies() -> None:
    with pytest.raises(TeacherBackendUnsupportedPolicyError, match="not implemented"):
        GPUTorchTeacherEmissionBackend(_config(target_policy="corridor_exemplar_v1"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dynamic_top_k_min", 0, "dynamic_top_k_min"),
        ("dynamic_top_k_max", 0, "dynamic_top_k_max"),
        ("dynamic_mass_threshold", 0.0, "dynamic_mass_threshold"),
        ("dynamic_mass_threshold", 1.1, "dynamic_mass_threshold"),
        ("dynamic_top_k_policy", "unknown", "dynamic_top_k_policy"),
    ),
)
def test_gpu_torch_rejects_invalid_dynamic_config(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GPUTorchTeacherEmissionBackend(
            _config(
                target_policy="dynamic_cascaded_soft_labels_v1",
                **{field: value},
            )
        )


def test_detect_torch_accelerator_reports_missing_torch(monkeypatch) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "torch":
            raise ImportError("no torch")
        return import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        fake_import_module,
    )

    detection = detect_torch_accelerator()

    assert not detection.available
    assert detection.device_kind == "unavailable"
    assert detection.device is None
    assert detection.reason == "missing torch"


def test_detect_torch_accelerator_prefers_cuda_over_mps(monkeypatch) -> None:
    fake_torch = _fake_torch(cuda_available=True, mps_available=True)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        lambda name: fake_torch if name == "torch" else import_module(name),
    )

    detection = GPUTorchTeacherEmissionBackend.detect_device()

    assert detection.available
    assert detection.device_kind == "cuda"
    assert detection.device == "cuda"
    assert detection.cuda_available
    assert detection.mps_available


def test_detect_torch_accelerator_uses_mps_when_cuda_unavailable(
    monkeypatch,
) -> None:
    fake_torch = _fake_torch(cuda_available=False, mps_available=True)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        lambda name: fake_torch if name == "torch" else import_module(name),
    )

    detection = detect_torch_accelerator()

    assert detection.available
    assert detection.device_kind == "mps"
    assert detection.device == "mps"
    assert not detection.cuda_available
    assert detection.mps_available


def test_gpu_torch_no_accelerator_error_is_clear(monkeypatch) -> None:
    fake_torch = _fake_torch(cuda_available=False, mps_available=False)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        lambda name: fake_torch if name == "torch" else import_module(name),
    )
    backend = GPUTorchTeacherEmissionBackend(_config())

    with pytest.raises(RuntimeError) as exc_info:
        backend.emit_batch(_batch())

    message = str(exc_info.value)
    assert "gpu_torch" in message
    assert "cuda" in message
    assert "mps" in message
    assert "cpu_gpu" in message
    assert "no CPU fallback" in message


def test_gpu_torch_missing_transformers_error_is_clear(monkeypatch) -> None:
    fake_torch = _fake_torch(cuda_available=True, mps_available=False)

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            raise ImportError("no transformers")
        return import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        fake_import_module,
    )
    backend = GPUTorchTeacherEmissionBackend(_config())

    with pytest.raises(RuntimeError, match="teacher-hf.*transformers"):
        backend.emit_batch(_batch())


def test_gpu_torch_available_returns_false_without_device_or_model() -> None:
    assert not check_gpu_torch_backend_available(_config())


def test_gpu_torch_metadata_is_honest_without_emission() -> None:
    backend = GPUTorchTeacherEmissionBackend(_config())

    metadata = backend.metadata(actual_batch_size=2, effective_vocab_size=11)

    assert metadata["requested_runtime_mode"] == "cpu_gpu"
    assert metadata["effective_runtime_mode"] == "cpu_gpu"
    assert metadata["requested_target_policy"] == "dense_logits"
    assert metadata["effective_target_policy"] == "dense_logits"
    assert metadata["backend_id"] == "gpu_torch"
    assert metadata["backend_family"] == "gpu_torch"
    assert metadata["runtime_kind"] == "gpu_torch"
    assert metadata["capability_status"] == "supported_debug"
    assert metadata["optimized_path_used"] is False
    assert metadata["dense_debug_path"] is True
    assert metadata["dense_logits_transferred_to_host"] is True
    assert metadata["compact_reduction_used"] is False
    assert metadata["gpu_compact_reduction_implemented"] is False
    assert metadata["fallback_used"] is False
    assert metadata["configured_batch_size"] == 2
    assert metadata["actual_batch_size"] == 2
    assert metadata["sequence_length"] == 4
    assert metadata["configured_vocab_size"] == 7
    assert metadata["effective_vocab_size"] == 11
    assert metadata["local_files_only"] is True
    assert metadata["allow_downloads"] is False
    assert metadata["vocab_chunking_requested"] is False
    assert metadata["vocab_chunking_used"] is False
    assert metadata["gpu_vocab_chunk_size_requested"] is None
    assert metadata["gpu_vocab_chunk_size_effective"] is None
    assert metadata["gpu_vocab_chunks_per_batch"] == 1
    assert metadata["fallback_policy"] == "error"
    assert metadata["fallback_allowed"] is False
    assert metadata["fallback_handled_by"] == "none"
    assert metadata["diagnostic_status"] == "ok"
    assert metadata["failure_stage"] == "none"
    assert metadata["failure_reason"] is None


def test_gpu_torch_topk_metadata_records_compact_reduction() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(target_policy="topk_with_tail_v0", top_k=3)
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 3), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["requested_target_policy"] == "topk_with_tail_v0"
    assert metadata["effective_target_policy"] == "topk_with_tail_v0"
    assert metadata["capability_status"] == "optimized"
    assert metadata["optimized_path_used"] is True
    assert metadata["dense_debug_path"] is False
    assert metadata["dense_logits_transferred_to_host"] is False
    assert metadata["compact_reduction_used"] is True
    assert metadata["gpu_compact_reduction_implemented"] is True
    assert metadata["gpu_reduction_mode"] == "compact_topk_tail"
    assert metadata["requested_top_k"] == 3
    assert metadata["effective_top_k"] == 3
    assert metadata["compact_payload_arrays"] == [
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    ]
    assert metadata["compact_payload_fields"] == metadata["compact_payload_arrays"]
    assert metadata["compact_bytes_transferred_to_host"] == sum(
        int(value.nbytes) for value in compact_payload.values()
    )
    assert metadata["estimated_dense_logits_bytes"] == 2 * 4 * 11 * 4
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 11 * 4 * 3
    assert metadata["estimated_reducer_workspace_is_measured"] is False
    assert "effective_vocab_or_chunk" in metadata["estimated_reducer_workspace_formula"]
    assert metadata["estimated_dense_logits_dtype"] == "float32"
    assert metadata["vocab_chunking_requested"] is False
    assert metadata["vocab_chunking_used"] is False
    assert metadata["gpu_vocab_chunk_size_requested"] is None
    assert metadata["gpu_vocab_chunk_size_effective"] is None
    assert metadata["gpu_vocab_chunks_per_batch"] == 1


def test_gpu_torch_topk_chunked_metadata_records_workspace_estimate() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            target_policy="topk_with_tail_v0",
            top_k=3,
            gpu_enable_vocab_chunking=True,
            gpu_vocab_chunk_size=5,
        )
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 3), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["gpu_reduction_mode"] == "compact_topk_tail_chunked"
    assert metadata["vocab_chunking_requested"] is True
    assert metadata["vocab_chunking_used"] is True
    assert metadata["gpu_vocab_chunk_size_requested"] == 5
    assert metadata["gpu_vocab_chunk_size_effective"] == 5
    assert metadata["gpu_vocab_chunks_per_batch"] == 3
    assert metadata["estimated_dense_logits_bytes"] == 2 * 4 * 11 * 4
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 5 * 4 * 3
    assert metadata["estimated_reducer_workspace_is_measured"] is False


def test_gpu_torch_cascaded_metadata_records_bucket_reduction() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(target_policy="cascaded_soft_labels_v1", top_k=3, num_buckets=2)
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 3), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "bucket_masses": np.ones((2, 4, 2), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["requested_target_policy"] == "cascaded_soft_labels_v1"
    assert metadata["effective_target_policy"] == "cascaded_soft_labels_v1"
    assert metadata["capability_status"] == "optimized"
    assert metadata["optimized_path_used"] is True
    assert metadata["dense_debug_path"] is False
    assert metadata["dense_logits_transferred_to_host"] is False
    assert metadata["compact_reduction_used"] is True
    assert metadata["gpu_compact_reduction_implemented"] is True
    assert metadata["gpu_reduction_mode"] == "compact_cascaded_soft_labels"
    assert metadata["requested_top_k"] == 3
    assert metadata["effective_top_k"] == 3
    assert metadata["num_buckets"] == 2
    assert metadata["bucket_policy"] == "contiguous_descending_tail_probability_mass"
    assert metadata["compact_payload_arrays"] == [
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    ]
    assert metadata["compact_payload_fields"] == metadata["compact_payload_arrays"]
    assert metadata["compact_bytes_transferred_to_host"] == sum(
        int(value.nbytes) for value in compact_payload.values()
    )
    assert metadata["estimated_dense_logits_bytes"] == 2 * 4 * 11 * 4
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 11 * 4 * 3
    assert metadata["estimated_reducer_workspace_is_measured"] is False
    assert metadata["estimated_dense_logits_dtype"] == "float32"
    assert metadata["vocab_chunking_requested"] is False
    assert metadata["vocab_chunking_used"] is False
    assert metadata["gpu_vocab_chunk_size_requested"] is None
    assert metadata["gpu_vocab_chunk_size_effective"] is None
    assert metadata["gpu_vocab_chunks_per_batch"] == 1


def test_gpu_torch_cascaded_chunking_request_is_not_overclaimed() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            target_policy="cascaded_soft_labels_v1",
            top_k=3,
            num_buckets=2,
            gpu_enable_vocab_chunking=True,
            gpu_vocab_chunk_size=5,
        )
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 3), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 3), dtype=np.float32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "bucket_masses": np.ones((2, 4, 2), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["vocab_chunking_requested"] is True
    assert metadata["vocab_chunking_used"] is False
    assert (
        metadata["vocab_chunking_reason"]
        == "exact_bucket_policy_requires_full_probability_workspace"
    )
    assert metadata["gpu_vocab_chunk_size_requested"] == 5
    assert metadata["gpu_vocab_chunk_size_effective"] is None
    assert metadata["gpu_vocab_chunks_per_batch"] == 1
    assert metadata["gpu_reduction_mode"] == "compact_cascaded_soft_labels"
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 11 * 4 * 3
    assert metadata["dense_logits_transferred_to_host"] is False


def test_gpu_torch_dynamic_cascaded_metadata_records_compact_reduction() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=2,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.75,
            num_buckets=3,
        )
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 5), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 5), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 5), dtype=np.float32),
        "top_selection_mask": np.zeros((2, 4, 5), dtype=bool),
        "effective_top_k": np.full((2, 4), 2, dtype=np.int32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "bucket_masses": np.ones((2, 4, 3), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["requested_target_policy"] == "dynamic_cascaded_soft_labels_v1"
    assert metadata["effective_target_policy"] == "dynamic_cascaded_soft_labels_v1"
    assert metadata["capability_status"] == "optimized"
    assert metadata["optimized_path_used"] is True
    assert metadata["dense_debug_path"] is False
    assert metadata["dense_logits_transferred_to_host"] is False
    assert metadata["compact_reduction_used"] is True
    assert metadata["gpu_compact_reduction_implemented"] is True
    assert metadata["gpu_reduction_mode"] == "compact_dynamic_cascaded_soft_labels"
    assert metadata["dynamic_top_k_policy"] == "mass_threshold_v1"
    assert metadata["dynamic_top_k_min_configured"] == 2
    assert metadata["dynamic_top_k_max_configured"] == 5
    assert metadata["dynamic_top_k_min_effective"] == 2
    assert metadata["dynamic_top_k_max_effective"] == 5
    assert metadata["dynamic_mass_threshold"] == 0.75
    assert metadata["selection_mask_field"] == "top_selection_mask"
    assert metadata["top_selection_mask_semantics"] == (
        "true means explicit selected token"
    )
    assert metadata["dynamic_head_selection_vectorized"] is True
    assert metadata["dynamic_tail_bucket_vectorized"] is True
    assert metadata["padding_policy"] == "pad_to_dynamic_top_k_max_effective"
    assert "must be ignored" in metadata["masked_slot_policy"]
    assert metadata["effective_top_k_min_observed"] == 2
    assert metadata["effective_top_k_max_observed"] == 2
    assert metadata["effective_top_k_mean_observed"] == pytest.approx(2.0)
    assert metadata["num_buckets"] == 3
    assert metadata["bucket_policy"] == "contiguous_descending_tail_probability_mass"
    assert metadata["compact_payload_arrays"] == [
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    ]
    assert metadata["compact_payload_fields"] == metadata["compact_payload_arrays"]
    assert metadata["compact_bytes_transferred_to_host"] == sum(
        int(value.nbytes) for value in compact_payload.values()
    )
    assert metadata["estimated_dense_logits_bytes"] == 2 * 4 * 11 * 4
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 11 * 4 * 3
    assert metadata["estimated_reducer_workspace_is_measured"] is False
    assert metadata["fallback_used"] is False
    assert metadata["fallback_policy"] == "error"
    assert metadata["failure_stage"] == "none"
    assert metadata["failure_reason"] is None


def test_gpu_torch_dynamic_cascaded_chunking_request_is_not_overclaimed() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=2,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.75,
            num_buckets=3,
            gpu_enable_vocab_chunking=True,
            gpu_vocab_chunk_size=5,
        )
    )
    compact_payload = {
        "top_token_ids": np.zeros((2, 4, 5), dtype=np.int32),
        "top_log_probs": np.zeros((2, 4, 5), dtype=np.float32),
        "top_probs": np.zeros((2, 4, 5), dtype=np.float32),
        "top_selection_mask": np.zeros((2, 4, 5), dtype=bool),
        "effective_top_k": np.full((2, 4), 2, dtype=np.int32),
        "top_mass": np.zeros((2, 4), dtype=np.float32),
        "tail_mass": np.ones((2, 4), dtype=np.float32),
        "bucket_masses": np.ones((2, 4, 3), dtype=np.float32),
        "teacher_entropy": np.zeros((2, 4), dtype=np.float32),
    }

    metadata = backend.metadata(
        actual_batch_size=2,
        effective_vocab_size=11,
        compact_payload=compact_payload,
        estimated_dense_logits_dtype="float32",
    )

    assert metadata["vocab_chunking_requested"] is True
    assert metadata["vocab_chunking_used"] is False
    assert (
        metadata["vocab_chunking_reason"]
        == "dynamic_exact_bucket_policy_requires_full_probability_workspace"
    )
    assert metadata["gpu_vocab_chunk_size_requested"] == 5
    assert metadata["gpu_vocab_chunk_size_effective"] is None
    assert metadata["gpu_vocab_chunks_per_batch"] == 1
    assert metadata["gpu_reduction_mode"] == "compact_dynamic_cascaded_soft_labels"
    assert metadata["estimated_reducer_workspace_bytes"] == 2 * 4 * 11 * 4 * 3


def test_gpu_topk_tail_reducer_shapes_and_mass_accounting() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0],
                [0.5, 2.0, -0.5, -1.5, -2.0],
                [1.0, 0.0, 3.0, -2.0, -3.0],
                [2.0, 1.0, 0.0, -1.0, -2.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0],
                [0.0, 4.0, 1.0, -1.0, -2.0],
                [1.0, 2.0, 4.0, 0.0, -1.0],
                [0.0, -1.0, -2.0, 3.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )

    payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_topk_tail_reduce(torch, logits, top_k=3)
    )

    assert payload["top_token_ids"].shape == (2, 4, 3)
    assert payload["top_log_probs"].shape == (2, 4, 3)
    assert payload["top_probs"].shape == (2, 4, 3)
    assert payload["top_mass"].shape == (2, 4)
    assert payload["tail_mass"].shape == (2, 4)
    assert payload["teacher_entropy"].shape == (2, 4)
    assert np.isfinite(payload["top_log_probs"]).all()
    assert np.isfinite(payload["top_probs"]).all()
    np.testing.assert_allclose(
        payload["tail_mass"],
        1.0 - payload["top_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    for value in payload.values():
        assert isinstance(value, np.ndarray)


def test_gpu_topk_chunked_reducer_matches_unchunked() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                [0.5, 2.0, -0.5, -1.5, -2.0, -3.0],
                [1.0, 0.0, 3.0, -2.0, -3.0, -4.0],
                [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 4.0, 1.0, -1.0, -2.0, -3.0],
                [1.0, 2.0, 4.0, 0.0, -1.0, -2.0],
                [0.0, -1.0, -2.0, 3.0, 1.0, -3.0],
            ],
        ],
        dtype=torch.float32,
    )

    unchunked = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_topk_tail_reduce(torch, logits, top_k=3)
    )
    chunked = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_topk_tail_reduce(
            torch,
            logits,
            top_k=3,
            vocab_chunk_size=2,
        )
    )

    assert chunked["top_token_ids"].shape == (2, 4, 3)
    np.testing.assert_array_equal(chunked["top_token_ids"], unchunked["top_token_ids"])
    for field in (
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    ):
        np.testing.assert_allclose(
            chunked[field],
            unchunked[field],
            rtol=1e-6,
            atol=1e-6,
        )


def test_gpu_cascaded_reducer_shapes_and_bucket_mass_accounting() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                [0.5, 2.0, -0.5, -1.5, -2.0, -3.0],
                [1.0, 0.0, 3.0, -2.0, -3.0, -4.0],
                [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 4.0, 1.0, -1.0, -2.0, -3.0],
                [1.0, 2.0, 4.0, 0.0, -1.0, -2.0],
                [0.0, -1.0, -2.0, 3.0, 1.0, -3.0],
            ],
        ],
        dtype=torch.float32,
    )

    payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_cascaded_reduce(
            torch,
            logits,
            top_k=3,
            num_buckets=2,
        )
    )

    assert payload["top_token_ids"].shape == (2, 4, 3)
    assert payload["top_log_probs"].shape == (2, 4, 3)
    assert payload["top_probs"].shape == (2, 4, 3)
    assert payload["top_mass"].shape == (2, 4)
    assert payload["tail_mass"].shape == (2, 4)
    assert payload["bucket_masses"].shape == (2, 4, 2)
    assert payload["teacher_entropy"].shape == (2, 4)
    assert np.isfinite(payload["bucket_masses"]).all()
    np.testing.assert_allclose(
        payload["tail_mass"],
        1.0 - payload["top_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["bucket_masses"].sum(axis=-1),
        payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    for value in payload.values():
        assert isinstance(value, np.ndarray)


def test_gpu_cascaded_nonchunked_reuses_probability_workspace(monkeypatch) -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    calls = {"count": 0}
    original_log_softmax = torch.nn.functional.log_softmax

    def counting_log_softmax(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return original_log_softmax(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "log_softmax", counting_log_softmax)
    logits = torch.tensor(
        [[[4.0, 1.0, 0.0, -1.0, -2.0, -3.0]]],
        dtype=torch.float32,
    )

    gpu_torch._gpu_cascaded_reduce(torch, logits, top_k=2, num_buckets=2)

    assert calls["count"] == 1


def test_gpu_torch_real_dense_emission_when_local_model_and_device_exist() -> None:
    model_dir = os.environ.get("RADJAX_TOME_TEST_HF_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        pytest.skip("RADJAX_TOME_TEST_HF_MODEL_DIR is not configured")
    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    if not detect_torch_accelerator().available:
        pytest.skip("no CUDA or MPS Torch accelerator is available")
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            model_id=model_dir,
            tokenizer_id=model_dir,
            vocab_size=1,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.backend_id == "gpu_torch"
    assert result.runtime_mode == "cpu_gpu"
    assert result.target_policy == "dense_logits"
    assert result.input_ids.shape == (2, 4)
    assert result.attention_mask.shape == (2, 4)
    assert result.payload["logits"].shape[:2] == (2, 4)
    assert result.payload["logits"].__class__.__module__.startswith("numpy")
    assert result.metadata["runtime_kind"] == "gpu_torch"
    assert result.metadata["device_kind"] in {"cuda", "mps"}
    assert result.metadata["torch_device"] in {"cuda", "mps"}
    assert result.metadata["model_id"] == model_dir
    assert result.metadata["tokenizer_id"] == model_dir
    assert result.metadata["configured_vocab_size"] == 1
    assert result.metadata["effective_vocab_size"] >= 1
    assert result.metadata["optimized_path_used"] is False
    assert result.metadata["dense_debug_path"] is True
    assert result.metadata["dense_logits_transferred_to_host"] is True
    assert result.metadata["compact_reduction_used"] is False
    assert result.metadata["gpu_compact_reduction_implemented"] is False
    assert result.metadata["fallback_used"] is False
    for value in result.payload.values():
        if isinstance(value, np.ndarray):
            assert value.__class__.__module__.startswith("numpy")


def test_gpu_torch_real_topk_emission_when_local_model_and_device_exist() -> None:
    model_dir = os.environ.get("RADJAX_TOME_TEST_HF_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        pytest.skip("RADJAX_TOME_TEST_HF_MODEL_DIR is not configured")
    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    if not detect_torch_accelerator().available:
        pytest.skip("no CUDA or MPS Torch accelerator is available")
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            model_id=model_dir,
            tokenizer_id=model_dir,
            target_policy="topk_with_tail_v0",
            vocab_size=1,
            top_k=3,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.backend_id == "gpu_torch"
    assert result.runtime_mode == "cpu_gpu"
    assert result.target_policy == "topk_with_tail_v0"
    assert result.input_ids.shape == (2, 4)
    assert result.attention_mask.shape == (2, 4)
    assert result.payload["top_token_ids"].shape[:2] == (2, 4)
    assert (
        result.payload["top_log_probs"].shape == result.payload["top_token_ids"].shape
    )
    assert result.payload["top_probs"].shape == result.payload["top_token_ids"].shape
    assert result.payload["top_mass"].shape == (2, 4)
    assert result.payload["tail_mass"].shape == (2, 4)
    assert result.payload["teacher_entropy"].shape == (2, 4)
    assert np.isfinite(result.payload["top_log_probs"]).all()
    assert np.isfinite(result.payload["top_probs"]).all()
    np.testing.assert_allclose(
        result.payload["tail_mass"],
        1.0 - result.payload["top_mass"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.metadata["runtime_kind"] == "gpu_torch"
    assert result.metadata["device_kind"] in {"cuda", "mps"}
    assert result.metadata["optimized_path_used"] is True
    assert result.metadata["dense_debug_path"] is False
    assert result.metadata["dense_logits_transferred_to_host"] is False
    assert result.metadata["compact_reduction_used"] is True
    assert result.metadata["gpu_compact_reduction_implemented"] is True
    assert result.metadata["gpu_reduction_mode"] == "compact_topk_tail"
    assert result.metadata["compact_bytes_transferred_to_host"] > 0
    assert result.metadata["estimated_dense_logits_bytes"] > 0
    for value in result.payload.values():
        if isinstance(value, np.ndarray):
            assert value.__class__.__module__.startswith("numpy")


def test_gpu_torch_real_cascaded_emission_when_local_model_and_device_exist() -> None:
    model_dir = os.environ.get("RADJAX_TOME_TEST_HF_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        pytest.skip("RADJAX_TOME_TEST_HF_MODEL_DIR is not configured")
    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    if not detect_torch_accelerator().available:
        pytest.skip("no CUDA or MPS Torch accelerator is available")
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            model_id=model_dir,
            tokenizer_id=model_dir,
            target_policy="cascaded_soft_labels_v1",
            vocab_size=1,
            top_k=3,
            num_buckets=2,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.backend_id == "gpu_torch"
    assert result.runtime_mode == "cpu_gpu"
    assert result.target_policy == "cascaded_soft_labels_v1"
    assert result.input_ids.shape == (2, 4)
    assert result.attention_mask.shape == (2, 4)
    assert result.payload["top_token_ids"].shape[:2] == (2, 4)
    assert (
        result.payload["top_log_probs"].shape == result.payload["top_token_ids"].shape
    )
    assert result.payload["top_probs"].shape == result.payload["top_token_ids"].shape
    assert result.payload["top_mass"].shape == (2, 4)
    assert result.payload["tail_mass"].shape == (2, 4)
    assert result.payload["bucket_masses"].shape == (2, 4, 2)
    assert result.payload["teacher_entropy"].shape == (2, 4)
    assert np.isfinite(result.payload["bucket_masses"]).all()
    np.testing.assert_allclose(
        result.payload["bucket_masses"].sum(axis=-1),
        result.payload["tail_mass"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.metadata["runtime_kind"] == "gpu_torch"
    assert result.metadata["device_kind"] in {"cuda", "mps"}
    assert result.metadata["optimized_path_used"] is True
    assert result.metadata["dense_debug_path"] is False
    assert result.metadata["dense_logits_transferred_to_host"] is False
    assert result.metadata["compact_reduction_used"] is True
    assert result.metadata["gpu_compact_reduction_implemented"] is True
    assert result.metadata["gpu_reduction_mode"] == "compact_cascaded_soft_labels"
    assert result.metadata["num_buckets"] == 2
    assert (
        result.metadata["bucket_policy"]
        == "contiguous_descending_tail_probability_mass"
    )
    assert result.metadata["compact_bytes_transferred_to_host"] > 0
    assert result.metadata["estimated_dense_logits_bytes"] > 0
    for value in result.payload.values():
        if isinstance(value, np.ndarray):
            assert value.__class__.__module__.startswith("numpy")


def test_gpu_torch_real_dynamic_cascaded_emission_when_model_and_device_exist() -> None:
    model_dir = os.environ.get("RADJAX_TOME_TEST_HF_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        pytest.skip("RADJAX_TOME_TEST_HF_MODEL_DIR is not configured")
    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    if not detect_torch_accelerator().available:
        pytest.skip("no CUDA or MPS Torch accelerator is available")
    backend = GPUTorchTeacherEmissionBackend(
        _config(
            model_id=model_dir,
            tokenizer_id=model_dir,
            target_policy="dynamic_cascaded_soft_labels_v1",
            vocab_size=1,
            dynamic_top_k_min=1,
            dynamic_top_k_max=3,
            dynamic_mass_threshold=0.75,
            num_buckets=2,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.backend_id == "gpu_torch"
    assert result.runtime_mode == "cpu_gpu"
    assert result.target_policy == "dynamic_cascaded_soft_labels_v1"
    assert result.input_ids.shape == (2, 4)
    assert result.attention_mask.shape == (2, 4)
    assert result.payload["top_token_ids"].shape[:2] == (2, 4)
    assert (
        result.payload["top_log_probs"].shape == result.payload["top_token_ids"].shape
    )
    assert result.payload["top_probs"].shape == result.payload["top_token_ids"].shape
    assert result.payload["top_selection_mask"].shape == (
        result.payload["top_token_ids"].shape
    )
    assert result.payload["top_selection_mask"].dtype == np.bool_
    assert result.payload["effective_top_k"].shape == (2, 4)
    assert result.payload["top_mass"].shape == (2, 4)
    assert result.payload["tail_mass"].shape == (2, 4)
    assert result.payload["bucket_masses"].shape == (2, 4, 2)
    assert result.payload["teacher_entropy"].shape == (2, 4)
    np.testing.assert_allclose(
        result.payload["bucket_masses"].sum(axis=-1),
        result.payload["tail_mass"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.metadata["runtime_kind"] == "gpu_torch"
    assert result.metadata["device_kind"] in {"cuda", "mps"}
    assert result.metadata["optimized_path_used"] is True
    assert result.metadata["dense_debug_path"] is False
    assert result.metadata["dense_logits_transferred_to_host"] is False
    assert result.metadata["compact_reduction_used"] is True
    assert result.metadata["gpu_compact_reduction_implemented"] is True
    assert (
        result.metadata["gpu_reduction_mode"] == "compact_dynamic_cascaded_soft_labels"
    )
    assert result.metadata["dynamic_top_k_policy"] == "mass_threshold_v1"
    assert result.metadata["compact_bytes_transferred_to_host"] > 0
    assert result.metadata["estimated_dense_logits_bytes"] > 0
    for value in result.payload.values():
        if isinstance(value, np.ndarray):
            assert value.__class__.__module__.startswith("numpy")


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> object:
    return SimpleNamespace(
        __version__="0.test",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )


def _optional_dependencies_available() -> bool:
    try:
        import_module("torch")
        import_module("transformers")
    except ImportError:
        return False
    return True


def _optional_torch_available() -> bool:
    try:
        import_module("torch")
    except ImportError:
        return False
    return True
