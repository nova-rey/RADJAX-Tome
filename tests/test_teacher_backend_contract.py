from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from radjax_tome.backends import (
    BackendCapability,
    CPUReferenceTeacherEmissionBackend,
    FakeNumpyTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    list_backend_capabilities,
    register_backend,
)


def test_backend_package_imports_without_heavy_optional_dependencies() -> None:
    script = """
import importlib
import sys

for name in ("torch", "transformers", "jax"):
    sys.modules.pop(name, None)

importlib.import_module("radjax_tome.backends")
importlib.import_module("radjax_tome.backends.registry")
importlib.import_module("radjax_tome.backends.fake")
importlib.import_module("radjax_tome.backends.cpu")

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


def test_backend_config_and_batch_input_construct() -> None:
    config = TeacherBackendConfig(
        backend_id="fake_numpy",
        runtime_mode="cpu",
        cpu_orchestration_mode="serial",
        target_policy="dense_logits",
        model_id="fake-model",
        tokenizer_id="fake-tokenizer",
        sequence_length=4,
        batch_size=2,
        vocab_size=7,
        top_k=3,
        num_buckets=2,
        local_files_only=True,
        allow_downloads=False,
        fallback_policy="error",
    )
    batch = TeacherBatchInput(
        example_ids=("ex-1", "ex-2"),
        texts=("alpha", "beta"),
    )

    assert config.backend_id == "fake_numpy"
    assert config.sequence_length == 4
    assert config.top_k == 3
    assert config.num_buckets == 2
    assert batch.example_ids == ("ex-1", "ex-2")
    assert batch.texts == ("alpha", "beta")


def test_batch_input_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        TeacherBatchInput(example_ids=("ex-1",), texts=("alpha", "beta"))


def test_fake_backend_created_through_registry_emits_deterministic_logits() -> None:
    config = TeacherBackendConfig(
        backend_id="fake_numpy",
        sequence_length=4,
        batch_size=2,
        vocab_size=7,
    )
    batch = TeacherBatchInput(
        example_ids=("ex-1", "ex-2"),
        texts=("alpha", "beta"),
    )
    backend = create_backend(config)

    first = backend.emit_batch(batch)
    second = backend.emit_batch(batch)

    assert first.backend_id == "fake_numpy"
    assert first.runtime_mode == "cpu"
    assert first.target_policy == "dense_logits"
    assert first.input_ids.shape == (config.batch_size, config.sequence_length)
    assert first.attention_mask.shape == (config.batch_size, config.sequence_length)
    assert first.payload["logits"].shape == (
        config.batch_size,
        config.sequence_length,
        config.vocab_size,
    )
    np.testing.assert_array_equal(first.input_ids, second.input_ids)
    np.testing.assert_array_equal(first.attention_mask, second.attention_mask)
    np.testing.assert_array_equal(first.payload["logits"], second.payload["logits"])
    assert first.metadata["requested_runtime_mode"] == "cpu"
    assert first.metadata["effective_runtime_mode"] == "cpu"
    assert first.metadata["requested_target_policy"] == "dense_logits"
    assert first.metadata["effective_target_policy"] == "dense_logits"
    assert first.metadata["backend_id"] == "fake_numpy"
    assert first.metadata["backend_family"] == "fake_numpy"
    assert first.metadata["runtime_kind"] == "fake_numpy"
    assert first.metadata["device_kind"] == "cpu"
    assert first.metadata["optimized_path_used"] is False
    assert first.metadata["fallback_used"] is False


def test_registry_lists_default_backend_capabilities() -> None:
    capabilities = list_backend_capabilities()
    fake_capabilities = [
        capability
        for capability in capabilities
        if capability.backend_id == "fake_numpy"
    ]
    cpu_capabilities = [
        capability
        for capability in capabilities
        if capability.backend_id == "cpu_reference"
    ]

    assert len(fake_capabilities) == 1
    assert len(cpu_capabilities) == 4
    capability = fake_capabilities[0]
    assert capability.backend_family == "fake_numpy"
    assert capability.runtime_mode == "cpu"
    assert capability.target_policy == "dense_logits"
    assert capability.status == "supported_debug"
    assert capability.implemented_now
    assert not capability.optimized
    assert "Spec 3.3B fake backend proves" in capability.notes
    assert [capability.backend_id for capability in capabilities] == sorted(
        capability.backend_id for capability in capabilities
    )


def test_duplicate_and_unknown_backend_ids_fail_clearly() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_backend(FakeNumpyTeacherEmissionBackend)

    with pytest.raises(ValueError, match="already registered"):
        register_backend(CPUReferenceTeacherEmissionBackend)

    with pytest.raises(ValueError, match="unknown teacher backend"):
        create_backend(TeacherBackendConfig(backend_id="missing_backend"))


def test_capability_model_rejects_optimized_without_implementation() -> None:
    with pytest.raises(ValueError, match="optimized capabilities"):
        BackendCapability(
            backend_id="bad",
            backend_family="bad",
            runtime_mode="cpu",
            target_policy="dense_logits",
            status="optimized",
            optimized=True,
            implemented_now=False,
            notes="invalid",
        )
