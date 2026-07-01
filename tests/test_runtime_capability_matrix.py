from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_SENTENCE = (
    "Runtime mode chooses where computation happens. Target policy chooses what "
    "comes back. The capability matrix says what is supported. Metadata proves "
    "what actually happened. The writer stays backend-neutral."
)
RUNTIME_MODES = {"cpu", "cpu_gpu", "cpu_tpu"}
CPU_ORCHESTRATION_MODES = {"auto", "serial", "staged"}
TARGET_POLICIES = {
    "dense_logits",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
    "corridor_exemplar_v1",
}
SUPPORT_STATUSES = {
    "unsupported",
    "planned",
    "supported",
    "supported_debug",
    "optimized",
    "historical_reference_exists",
}
BACKEND_FAMILIES = {
    "fake_numpy",
    "cpu_reference",
    "hf_torch",
    "gpu_torch",
    "jax_tpu",
}


def test_runtime_backend_doc_defines_architecture_vocabulary() -> None:
    path = ROOT / "docs" / "TOME_RUNTIME_BACKENDS.md"
    text = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert ARCHITECTURE_SENTENCE in text
    assert "internal contract wall" in text
    assert "CPU front-end/orchestrator responsibilities" in text
    assert "Runtime backend responsibilities" in text
    for value in sorted(RUNTIME_MODES):
        assert f"`{value}`" in text
    for value in sorted(CPU_ORCHESTRATION_MODES):
        assert f"`{value}`" in text
    for value in sorted(TARGET_POLICIES):
        assert f"`{value}`" in text
    assert (
        "Dense logits are useful for reference, debug, and very small corpora" in text
    )
    assert "not the main optimization target" in text
    assert "There must be no silent accelerator-to-CPU fallback" in text


def test_runtime_capability_matrix_is_deterministic_and_complete() -> None:
    path = ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json"
    raw = path.read_text(encoding="utf-8")
    matrix = json.loads(raw)

    assert path.is_file()
    assert raw.endswith("\n")
    assert raw == json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    assert matrix["kind"] == "radjax_tome_runtime_capability_matrix"
    assert matrix["version"] == 1
    assert RUNTIME_MODES <= set(matrix["runtime_modes"])
    assert CPU_ORCHESTRATION_MODES <= set(matrix["cpu_orchestration_modes"])
    assert TARGET_POLICIES <= set(matrix["target_policies"])
    assert SUPPORT_STATUSES <= set(matrix["support_statuses"])
    assert BACKEND_FAMILIES <= set(matrix["backend_families"])
    assert matrix["orchestration"]["implemented_now"]
    assert matrix["orchestration"]["default_mode"] == "auto"
    assert set(matrix["orchestration"]["supported_modes"]) == {
        "auto",
        "serial",
        "staged",
    }
    assert not matrix["orchestration"]["staged_is_performance_optimized"]
    assert "public builder" in " ".join(matrix["non_goals"])


def test_runtime_capability_matrix_does_not_overclaim_accelerators() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = matrix["capabilities"]

    assert capabilities
    for capability in capabilities:
        assert {
            "backend_family",
            "runtime_mode",
            "target_policy",
            "status",
            "optimized",
            "implemented_now",
            "notes",
        } <= set(capability)
        assert capability["runtime_mode"] in RUNTIME_MODES
        assert capability["target_policy"] in TARGET_POLICIES
        assert capability["status"] in SUPPORT_STATUSES
        if capability["optimized"]:
            assert capability["implemented_now"]

    cpu_tpu_claims = [
        capability
        for capability in capabilities
        if capability["runtime_mode"] == "cpu_tpu"
    ]
    jax_tpu_claims = [
        capability
        for capability in capabilities
        if capability["backend_family"] == "jax_tpu"
    ]
    assert cpu_tpu_claims
    assert jax_tpu_claims
    assert not any(capability["implemented_now"] for capability in cpu_tpu_claims)
    assert not any(capability["optimized"] for capability in jax_tpu_claims)

    fallback = matrix["fallback_policy"]
    assert fallback["explicit_unsupported_request_should_fail"]
    assert fallback["auto_runtime_may_choose_supported_fallback"]
    assert fallback["fallback_must_be_recorded_in_metadata"]
    assert not fallback["silent_accelerator_to_cpu_fallback_allowed"]


def test_runtime_capability_matrix_reflects_contract_skeleton_only() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = matrix["capabilities"]
    fake_dense = [
        capability
        for capability in capabilities
        if capability["backend_family"] == "fake_numpy"
        and capability["target_policy"] == "dense_logits"
    ]
    assert len(fake_dense) == 1
    assert fake_dense[0]["implemented_now"]
    assert fake_dense[0]["status"] == "supported_debug"
    assert (
        "Spec 3.3B fake backend proves the backend contract" in fake_dense[0]["notes"]
    )
    assert "public builder has not migrated" in fake_dense[0]["notes"]

    assert not any(
        capability["implemented_now"]
        for capability in capabilities
        if capability["runtime_mode"] in {"cpu_gpu", "cpu_tpu"}
    )
    assert not any(
        "Spec 3.3B" in capability["notes"]
        for capability in capabilities
        if capability["backend_family"] in {"hf_torch", "gpu_torch", "jax_tpu"}
    )


def test_runtime_capability_matrix_reflects_cpu_reference_backend() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = {
        (capability["backend_family"], capability["target_policy"]): capability
        for capability in matrix["capabilities"]
    }

    dense = capabilities[("cpu_reference", "dense_logits")]
    topk = capabilities[("cpu_reference", "topk_with_tail_v0")]
    cascaded = capabilities[("cpu_reference", "cascaded_soft_labels_v1")]
    corridor = capabilities[("cpu_reference", "corridor_exemplar_v1")]

    assert dense["implemented_now"]
    assert dense["status"] == "supported_debug"
    assert not dense["optimized"]
    assert topk["implemented_now"]
    assert topk["status"] == "supported"
    assert not topk["optimized"]
    assert cascaded["implemented_now"]
    assert cascaded["status"] == "supported"
    assert not cascaded["optimized"]
    assert corridor["implemented_now"]
    assert corridor["status"] == "supported"
    assert not corridor["optimized"]
    assert (
        "Spec 3.3C.1 adds serial/reference CPU corridor/exemplar support"
        in (corridor["notes"])
    )
    assert "public builder has not migrated" in dense["notes"]

    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["runtime_mode"] in {"cpu_gpu", "cpu_tpu"}
    )


def test_runtime_capability_matrix_documents_orchestration() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    orchestration = matrix["orchestration"]

    assert orchestration["implemented_now"]
    assert orchestration["supported_modes"] == ["serial", "staged", "auto"]
    assert orchestration["default_mode"] == "auto"
    assert orchestration["staged_is_performance_optimized"] is False
    assert "pipeline-shaped" in orchestration["notes"]
    assert "historical optimizer port" in orchestration["notes"]
    non_goals = " ".join(matrix["non_goals"])
    assert "public builder" in non_goals
    assert "performance optimized" in non_goals
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["runtime_mode"] in {"cpu_gpu", "cpu_tpu"}
    )


def test_runtime_capability_matrix_reflects_hf_torch_backend_contract() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = {
        (capability["backend_family"], capability["target_policy"]): capability
        for capability in matrix["capabilities"]
    }

    dense = capabilities[("hf_torch", "dense_logits")]
    topk = capabilities[("hf_torch", "topk_with_tail_v0")]
    cascaded = capabilities[("hf_torch", "cascaded_soft_labels_v1")]
    corridor = capabilities[("hf_torch", "corridor_exemplar_v1")]

    assert dense["implemented_now"]
    assert dense["status"] == "supported_debug"
    assert not dense["optimized"]
    assert (
        "Spec 3.3E HF Torch backend emits real causal-LM dense logits"
        in (dense["notes"])
    )
    assert topk["implemented_now"]
    assert topk["status"] == "supported"
    assert not topk["optimized"]
    assert cascaded["implemented_now"]
    assert cascaded["status"] == "supported"
    assert not cascaded["optimized"]
    assert not corridor["implemented_now"]
    assert corridor["status"] == "planned"
    assert not corridor["optimized"]
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["runtime_mode"] in {"cpu_gpu", "cpu_tpu"}
    )
