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
