from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from radjax_tome.backends import (
    GPUTorchTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    check_gpu_torch_backend_available,
    diagnose_gpu_torch_backend,
)
from radjax_tome.backends.gpu_torch import (
    TeacherBackendDeviceError,
    TeacherBackendUnsupportedPolicyError,
    TorchAcceleratorDetection,
    _wrap_device_error,
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


def test_gpu_torch_diagnostics_reports_missing_torch_without_raising(
    monkeypatch,
) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "torch":
            raise ImportError("no torch")
        return import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        fake_import_module,
    )

    diagnostics = diagnose_gpu_torch_backend(_config())

    assert diagnostics["backend_id"] == "gpu_torch"
    assert diagnostics["runtime_mode"] == "cpu_gpu"
    assert diagnostics["target_policy"] == "dense_logits"
    assert diagnostics["dependency_status"] == "missing_torch"
    assert diagnostics["torch_available"] is False
    assert diagnostics["transformers_available"] is False
    assert diagnostics["accelerator_available"] is False
    assert diagnostics["can_emit"] is False
    assert diagnostics["failure_stage"] == "missing_dependency"
    assert "gpu_torch" in str(diagnostics["failure_reason"])
    assert "torch" in str(diagnostics["failure_reason"])
    assert "teacher-hf" in str(diagnostics["failure_reason"])
    assert diagnostics["fallback_used"] is False
    assert diagnostics["fallback_allowed"] is False


def test_gpu_torch_diagnostics_reports_missing_transformers_without_raising(
    monkeypatch,
) -> None:
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

    diagnostics = GPUTorchTeacherEmissionBackend.diagnostics(_config())

    assert diagnostics["dependency_status"] == "missing_transformers"
    assert diagnostics["torch_available"] is True
    assert diagnostics["transformers_available"] is False
    assert diagnostics["accelerator_available"] is True
    assert diagnostics["device_kind"] == "cuda"
    assert diagnostics["torch_device"] == "cuda"
    assert diagnostics["can_emit"] is False
    assert diagnostics["failure_stage"] == "missing_dependency"
    assert "transformers" in str(diagnostics["failure_reason"])
    assert "teacher-hf" in str(diagnostics["failure_reason"])


def test_gpu_torch_diagnostics_reports_no_accelerator_and_available_agrees(
    monkeypatch,
) -> None:
    fake_torch = _fake_torch(cuda_available=False, mps_available=False)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        lambda name: fake_torch if name == "torch" else import_module(name),
    )

    diagnostics = diagnose_gpu_torch_backend(_config())

    assert diagnostics["torch_available"] is True
    assert diagnostics["transformers_available"] is False
    assert diagnostics["accelerator_available"] is False
    assert diagnostics["cuda_available"] is False
    assert diagnostics["mps_available"] is False
    assert diagnostics["failure_stage"] == "no_accelerator"
    assert "cpu_gpu" in str(diagnostics["failure_reason"])
    assert "cuda" in str(diagnostics["failure_reason"])
    assert "mps" in str(diagnostics["failure_reason"])
    assert "no CPU fallback" in str(diagnostics["failure_reason"])
    assert check_gpu_torch_backend_available(_config()) is False


def test_gpu_torch_diagnostics_reports_missing_local_model_without_download(
    monkeypatch,
) -> None:
    fake_torch = _fake_torch(cuda_available=True, mps_available=False)
    calls: list[tuple[str, str, bool]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, *, local_files_only: bool) -> object:
            calls.append(("tokenizer", tokenizer_id, local_files_only))
            return object()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, *, local_files_only: bool) -> object:
            calls.append(("model", model_id, local_files_only))
            raise OSError("missing local model")

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        return import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        fake_import_module,
    )

    diagnostics = diagnose_gpu_torch_backend(_config())

    assert diagnostics["failure_stage"] == "model_load"
    assert diagnostics["tokenizer_available"] is True
    assert diagnostics["model_available"] is False
    assert diagnostics["local_files_only"] is True
    assert diagnostics["allow_downloads"] is False
    assert "model_id" in str(diagnostics["failure_reason"])
    assert "tokenizer_id" in str(diagnostics["failure_reason"])
    assert "local_files_only=True" in str(diagnostics["failure_reason"])
    assert "allow_downloads=False" in str(diagnostics["failure_reason"])
    assert calls == [
        ("tokenizer", "missing-local-hf-model", True),
        ("model", "missing-local-hf-model", True),
    ]


def test_gpu_torch_fallback_policy_auto_requires_orchestrator(
    monkeypatch,
) -> None:
    fake_torch = _fake_torch(cuda_available=False, mps_available=False)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        lambda name: fake_torch if name == "torch" else import_module(name),
    )
    config = _config(fallback_policy="auto")
    diagnostics = diagnose_gpu_torch_backend(config)

    assert diagnostics["fallback_policy"] == "auto"
    assert diagnostics["fallback_allowed"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["can_emit"] is False
    assert "orchestrator" in str(diagnostics["failure_reason"])
    assert "gpu_torch did not fall back" in str(diagnostics["failure_reason"])

    backend = GPUTorchTeacherEmissionBackend(config)
    with pytest.raises(RuntimeError) as exc_info:
        backend.emit_batch(_batch())
    message = str(exc_info.value)
    assert "no CPU fallback" in message
    assert "orchestrator" in message


def test_gpu_torch_unsupported_policy_is_structured() -> None:
    config = _config(target_policy="corridor_exemplar_v1")

    with pytest.raises(TeacherBackendUnsupportedPolicyError) as exc_info:
        GPUTorchTeacherEmissionBackend(config)

    message = str(exc_info.value)
    assert "gpu_torch" in message
    assert "target_policy" in message
    assert "corridor_exemplar_v1" in message
    assert "not implemented" in message

    diagnostics = diagnose_gpu_torch_backend(config)
    assert diagnostics["can_emit"] is False
    assert diagnostics["failure_stage"] == "unsupported_target"
    assert "corridor_exemplar_v1" in str(diagnostics["failure_reason"])


def test_gpu_torch_diagnostics_include_chunking_fields_for_invalid_config() -> None:
    diagnostics = diagnose_gpu_torch_backend(
        _config(gpu_enable_vocab_chunking=True, gpu_vocab_chunk_size=-1)
    )

    assert diagnostics["can_emit"] is False
    assert diagnostics["failure_stage"] == "invalid_config"
    assert diagnostics["gpu_enable_vocab_chunking"] is True
    assert diagnostics["gpu_vocab_chunk_size_requested"] == -1
    assert diagnostics["gpu_vocab_chunk_size_effective"] is None
    assert diagnostics["vocab_chunking_used"] is False


def test_gpu_torch_wraps_input_transfer_device_failure(monkeypatch) -> None:
    fake_torch = _fake_torch(cuda_available=True, mps_available=False)
    fake_transformers = _fake_transformers(
        tokenizer=_FakeTokenizer(_FailingTransferTensor()),
        model=_FakeModel(),
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        return import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.gpu_torch.import_module",
        fake_import_module,
    )

    backend = GPUTorchTeacherEmissionBackend(_config())

    with pytest.raises(TeacherBackendDeviceError) as exc_info:
        backend.emit_batch(_batch())

    message = str(exc_info.value)
    assert "gpu_torch" in message
    assert "input tensor transfer to device" in message
    assert "device_kind='cuda'" in message
    assert "torch_device='cuda'" in message
    assert "out of memory or device failure" in message
    assert "no CPU fallback" in message


def test_gpu_torch_wraps_mps_unsupported_operation() -> None:
    error = _wrap_device_error(
        "model forward",
        RuntimeError("operator is not implemented for MPS"),
        TorchAcceleratorDetection(
            available=True,
            device_kind="mps",
            device="mps",
            reason="mps available",
            mps_available=True,
        ),
    )

    message = str(error)
    assert "gpu_torch" in message
    assert "model forward" in message
    assert "mps" in message
    assert "unsupported operation" in message
    assert "no CPU fallback" in message


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> object:
    return SimpleNamespace(
        __version__="0.test",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
        no_grad=lambda: _NoGrad(),
    )


def _fake_transformers(*, tokenizer: object, model: object) -> object:
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, *, local_files_only: bool) -> object:
            return tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, *, local_files_only: bool) -> object:
            return model

    return SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self, tensor: object) -> None:
        self._tensor = tensor

    def __call__(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"input_ids": self._tensor, "attention_mask": self._tensor}


class _FakeModel:
    def eval(self) -> None:
        return None

    def to(self, device: str) -> _FakeModel:
        return self


class _FailingTransferTensor:
    def to(self, device: str) -> object:
        raise RuntimeError("CUDA out of memory")
