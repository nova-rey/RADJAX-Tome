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
    "dynamic_cascaded_soft_labels_v1",
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
    assert "Spec 3.3F5 adds runtime diagnostics and error hardening" in text
    assert "fallback_policy=auto" in text
    assert "orchestrator signal only" in text
    assert "does not emit CPU results for an explicit `cpu_gpu` request" in text
    assert "dynamic_cascaded_soft_labels_v1" in text
    assert "dynamic top-k explicit head" in text
    assert "bucketed tail" in text
    assert "Future corridor/exemplar schema" in text


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
        if capability["runtime_mode"] == "cpu_tpu"
    )
    assert not any(
        capability["implemented_now"]
        for capability in capabilities
        if capability["backend_family"] == "gpu_torch"
        and capability["target_policy"]
        not in {"dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"}
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
    dynamic = capabilities[("cpu_reference", "dynamic_cascaded_soft_labels_v1")]
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
    assert dynamic["implemented_now"]
    assert dynamic["status"] == "supported"
    assert not dynamic["optimized"]
    assert "Spec 3.3F6" in dynamic["notes"]
    assert "dynamic top-k explicit head plus bucketed tail" in dynamic["notes"]
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
        if capability["runtime_mode"] == "cpu_tpu"
    )
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["backend_family"] == "gpu_torch"
        and capability["target_policy"]
        not in {"dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"}
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
        if capability["runtime_mode"] == "cpu_tpu"
    )
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["backend_family"] == "gpu_torch"
        and capability["target_policy"]
        not in {"dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"}
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
    dynamic = capabilities[("hf_torch", "dynamic_cascaded_soft_labels_v1")]
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
    assert not dynamic["implemented_now"]
    assert dynamic["status"] == "planned"
    assert not dynamic["optimized"]
    assert "Spec 3.3F6" in dynamic["notes"]
    assert not corridor["implemented_now"]
    assert corridor["status"] == "planned"
    assert not corridor["optimized"]
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["runtime_mode"] == "cpu_tpu"
    )
    assert not any(
        capability["implemented_now"]
        for capability in matrix["capabilities"]
        if capability["backend_family"] == "gpu_torch"
        and capability["target_policy"]
        not in {"dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"}
    )


def test_runtime_capability_matrix_reflects_gpu_torch_cascaded_reducer() -> None:
    matrix = json.loads(
        (ROOT / "docs" / "TOME_RUNTIME_CAPABILITY_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = {
        (capability["backend_family"], capability["target_policy"]): capability
        for capability in matrix["capabilities"]
    }

    dense = capabilities[("gpu_torch", "dense_logits")]
    topk = capabilities[("gpu_torch", "topk_with_tail_v0")]
    cascaded = capabilities[("gpu_torch", "cascaded_soft_labels_v1")]
    dynamic = capabilities[("gpu_torch", "dynamic_cascaded_soft_labels_v1")]
    corridor = capabilities[("gpu_torch", "corridor_exemplar_v1")]

    assert dense["runtime_mode"] == "cpu_gpu"
    assert dense["implemented_now"]
    assert dense["status"] == "supported_debug"
    assert not dense["optimized"]
    assert "Spec 3.3F1 gpu_torch emits dense debug HF logits" in dense["notes"]
    assert "transfers dense logits back to host" in dense["notes"]
    assert "Spec 3.3F5" in dense["notes"]
    assert "no backend-local CPU fallback" in dense["notes"]
    assert topk["runtime_mode"] == "cpu_gpu"
    assert topk["implemented_now"]
    assert topk["status"] == "optimized"
    assert topk["optimized"]
    assert "Spec 3.3F2 gpu_torch computes GPU compact top-k/tail" in topk["notes"]
    assert "compact payload arrays" in topk["notes"]
    assert "optional vocab chunking" in topk["notes"]
    assert "memory/workspace metadata" in topk["notes"]
    assert "fallback semantics" in topk["notes"]
    assert cascaded["runtime_mode"] == "cpu_gpu"
    assert cascaded["implemented_now"]
    assert cascaded["status"] == "optimized"
    assert cascaded["optimized"]
    assert "Spec 3.3F3 gpu_torch computes GPU compact cascaded" in cascaded["notes"]
    assert "bucket masses" in cascaded["notes"]
    assert "compact payload arrays" in cascaded["notes"]
    assert "Spec 3.3F4.1" in cascaded["notes"]
    assert "requested cascaded chunking honestly" in cascaded["notes"]
    assert "full probability workspace" in cascaded["notes"]
    assert "shared probability workspace reuse" in cascaded["notes"]
    assert "structured runtime diagnostics" in cascaded["notes"]
    assert dynamic["runtime_mode"] == "cpu_gpu"
    assert dynamic["status"] == "planned"
    assert not dynamic["implemented_now"]
    assert not dynamic["optimized"]
    assert "Spec 3.3F7" in dynamic["notes"]
    assert corridor["runtime_mode"] == "cpu_gpu"
    assert corridor["status"] == "historical_reference_exists"
    assert not corridor["implemented_now"]
    assert not corridor["optimized"]

    non_goals = " ".join(matrix["non_goals"])
    assert "Do not silently fall back to CPU" in non_goals
    assert "Spec 3.3F6" in non_goals
    assert "backend-local CPU fallback" in non_goals
    assert "measured peak GPU memory" in non_goals
