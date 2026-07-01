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


def test_gpu_torch_registered_with_dense_debug_capability() -> None:
    capabilities = [
        capability
        for capability in list_backend_capabilities()
        if capability.backend_id == "gpu_torch"
    ]

    assert len(capabilities) == 4
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
    for policy in (
        "topk_with_tail_v0",
        "cascaded_soft_labels_v1",
        "corridor_exemplar_v1",
    ):
        assert statuses[policy] == "historical_reference_exists"
        assert not implemented[policy]
        assert not optimized[policy]


def test_gpu_torch_constructs_without_loading_optional_deps() -> None:
    backend = create_backend(_config())

    assert backend.backend_id == "gpu_torch"
    assert backend.backend_family == "gpu_torch"
    assert backend.runtime_mode == "cpu_gpu"


@pytest.mark.parametrize("runtime_mode", ("cpu", "cpu_tpu"))
def test_gpu_torch_rejects_non_gpu_runtime_modes(runtime_mode: str) -> None:
    with pytest.raises(ValueError, match="runtime_mode"):
        GPUTorchTeacherEmissionBackend(_config(runtime_mode=runtime_mode))


@pytest.mark.parametrize(
    "target_policy",
    ("topk_with_tail_v0", "cascaded_soft_labels_v1", "corridor_exemplar_v1"),
)
def test_gpu_torch_rejects_compact_policies_for_f1(target_policy: str) -> None:
    with pytest.raises(ValueError, match="dense_logits"):
        GPUTorchTeacherEmissionBackend(_config(target_policy=target_policy))


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
