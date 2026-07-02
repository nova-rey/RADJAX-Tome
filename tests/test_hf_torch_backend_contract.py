from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from radjax_tome.backends import (
    HFTorchTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    list_backend_capabilities,
)


def _config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "hf_torch",
        "runtime_mode": "cpu",
        "target_policy": "dense_logits",
        "model_id": "missing-local-hf-model",
        "tokenizer_id": "missing-local-hf-model",
        "sequence_length": 4,
        "batch_size": 2,
        "vocab_size": 7,
        "top_k": 3,
        "num_buckets": 2,
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


def test_hf_torch_registered_and_capabilities_are_import_safe() -> None:
    capabilities = [
        capability
        for capability in list_backend_capabilities()
        if capability.backend_id == "hf_torch"
    ]

    assert len(capabilities) == 5
    assert {capability.target_policy for capability in capabilities} == {
        "dense_logits",
        "topk_with_tail_v0",
        "cascaded_soft_labels_v1",
        "dynamic_cascaded_soft_labels_v1",
        "corridor_exemplar_v1",
    }
    assert not any(capability.optimized for capability in capabilities)
    assert not any(capability.runtime_mode == "cpu_gpu" for capability in capabilities)
    statuses = {
        capability.target_policy: capability.status for capability in capabilities
    }
    assert statuses["dense_logits"] == "supported_debug"
    assert statuses["topk_with_tail_v0"] == "supported"
    assert statuses["cascaded_soft_labels_v1"] == "supported"
    assert statuses["dynamic_cascaded_soft_labels_v1"] == "planned"
    assert statuses["corridor_exemplar_v1"] == "planned"
    implemented = {
        capability.target_policy: capability.implemented_now
        for capability in capabilities
    }
    assert not implemented["dynamic_cascaded_soft_labels_v1"]


def test_hf_torch_backend_constructs_without_loading_optional_deps() -> None:
    backend = create_backend(_config())

    assert backend.backend_id == "hf_torch"
    assert backend.backend_family == "hf_torch"
    assert backend.runtime_mode == "cpu"


def test_hf_torch_rejects_gpu_runtime_mode() -> None:
    with pytest.raises(ValueError, match="runtime_mode"):
        HFTorchTeacherEmissionBackend(_config(runtime_mode="cpu_gpu"))


def test_hf_torch_missing_optional_dependency_error_is_clear(monkeypatch) -> None:
    original_import_module = import_module

    def fake_import_module(name: str):
        if name == "torch":
            raise ImportError("no torch")
        return original_import_module(name)

    monkeypatch.setattr(
        "radjax_tome.backends.hf_torch.import_module",
        fake_import_module,
    )
    backend = HFTorchTeacherEmissionBackend(_config())

    with pytest.raises(RuntimeError, match="teacher-hf.*torch.*transformers"):
        backend.emit_batch(_batch())


def test_hf_torch_local_model_unavailable_error_is_clear() -> None:
    backend = HFTorchTeacherEmissionBackend(_config())

    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    with pytest.raises(RuntimeError, match="local torch/transformers model"):
        backend.emit_batch(_batch())


def test_hf_torch_available_returns_false_for_missing_local_model() -> None:
    assert not HFTorchTeacherEmissionBackend.available(_config())


@pytest.mark.parametrize(
    "target_policy",
    ("dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"),
)
def test_hf_torch_real_emission_when_local_model_is_configured(
    target_policy: str,
) -> None:
    model_dir = os.environ.get("RADJAX_TOME_TEST_HF_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        pytest.skip("RADJAX_TOME_TEST_HF_MODEL_DIR is not configured")
    if not _optional_dependencies_available():
        pytest.skip("optional torch/transformers dependencies are not installed")
    backend = HFTorchTeacherEmissionBackend(
        _config(
            model_id=model_dir,
            tokenizer_id=model_dir,
            target_policy=target_policy,
            vocab_size=1,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.backend_id == "hf_torch"
    assert result.runtime_mode == "cpu"
    assert result.target_policy == target_policy
    assert result.input_ids.shape == (2, 4)
    assert result.attention_mask.shape == (2, 4)
    assert result.metadata["runtime_kind"] == "hf_torch"
    assert result.metadata["device_kind"] == "cpu"
    assert result.metadata["model_id"] == model_dir
    assert result.metadata["tokenizer_id"] == model_dir
    assert result.metadata["configured_vocab_size"] == 1
    assert result.metadata["effective_vocab_size"] >= 1
    assert result.metadata["optimized_path_used"] is False
    assert result.metadata["fallback_used"] is False
    for value in result.payload.values():
        if isinstance(value, np.ndarray):
            assert value.__class__.__module__.startswith("numpy")
    if target_policy == "dense_logits":
        assert result.payload["logits"].shape[:2] == (2, 4)
    elif target_policy == "topk_with_tail_v0":
        assert "top_token_ids" in result.payload
        assert "tail_mass" in result.payload
    else:
        assert "bucket_masses" in result.payload


def _optional_dependencies_available() -> bool:
    try:
        import_module("torch")
        import_module("transformers")
    except ImportError:
        return False
    return True
