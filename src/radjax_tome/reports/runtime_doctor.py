from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_tome.backends.base import (
    TeacherBackendConfig,
    resolve_exemplar_capture_policy,
    resolve_gpu_batch_size_policy,
)
from radjax_tome.backends.registry import list_backend_capabilities
from radjax_tome.io.json import write_json

RUNTIME_DOCTOR_REPORT_SCHEMA = "runtime_doctor_report_v1"


def build_runtime_doctor_report(
    config: TeacherBackendConfig,
    *,
    exemplar_selector_policy: str = "multi_leaderboard_exemplar_selector_v1",
    exemplar_selection_enabled: bool = False,
    exemplar_fulfillment_policy: str = "auto",
) -> dict[str, Any]:
    """Build a lightweight backend preflight report."""
    capability = _matching_capability(config)
    capture = resolve_exemplar_capture_policy(config)
    batch = resolve_gpu_batch_size_policy(config, payload=None)
    target_supported = bool(capability and capability.implemented_now)

    report: dict[str, Any] = {
        "report_schema": RUNTIME_DOCTOR_REPORT_SCHEMA,
        "backend_id": config.backend_id,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id,
        "local_files_only": config.local_files_only,
        "allow_downloads": config.allow_downloads,
        "fallback_policy": config.fallback_policy,
        "fallback_allowed": config.fallback_policy == "auto",
        "fallback_handled_by": "none",
        "can_emit": target_supported,
        "failure_stage": "none" if target_supported else "unsupported_target",
        "failure_reason": None
        if target_supported
        else _unsupported_target_reason(config),
        "target_policy_supported": target_supported,
        "capability_status": capability.status if capability else "unsupported",
        "optimized": bool(capability and capability.optimized),
        "implemented_now": bool(capability and capability.implemented_now),
        "exemplar_capture_mode_requested": capture["exemplar_capture_mode_requested"],
        "exemplar_capture_mode_effective": capture["exemplar_capture_mode_effective"],
        "exemplar_source_policy": config.exemplar_source_policy,
        "exemplar_second_pass_source_policy": (
            config.exemplar_second_pass_source_policy
        ),
        "exemplar_selector_policy": exemplar_selector_policy,
        "exemplar_selection_enabled": exemplar_selection_enabled,
        "exemplar_fulfillment_policy": exemplar_fulfillment_policy,
        "gpu_batch_size_mode_requested": batch["gpu_batch_size_mode_requested"],
        "gpu_batch_size_mode_effective": batch["gpu_batch_size_mode_effective"],
        "effective_gpu_batch_size": batch["effective_gpu_batch_size"],
        "gpu_batch_size_probe_required": batch["gpu_batch_size_probe_required"],
        "gpu_batch_size_warning_emitted": batch["gpu_batch_size_warning_emitted"],
        "gpu_batch_size_warning_reason": batch["gpu_batch_size_warning_reason"],
        "batch_partition_strategy": batch["batch_partition_strategy"],
        "multidevice_enabled": batch["multidevice_enabled"],
    }

    if config.backend_id == "gpu_torch":
        report.update(_gpu_torch_diagnostics(config))
    else:
        report.update(
            {
                "torch_available": None,
                "transformers_available": None,
                "accelerator_available": None,
                "device_kind": "cpu",
                "torch_device": None,
                "cuda_available": None,
                "mps_available": None,
                "torch_version": None,
                "model_available": None,
                "tokenizer_available": None,
                "dependency_status": "not_required",
            }
        )

    report["remediation_hint"] = remediation_hint_for_failure(report, config)
    return report


def write_runtime_doctor_report(report: Mapping[str, Any], path: Path) -> None:
    write_json(path, dict(report))


def render_runtime_doctor_summary(report: Mapping[str, Any]) -> list[str]:
    dependency_status = report.get("dependency_status")
    accelerator = report.get("device_kind")
    accelerator_status = accelerator or _availability(
        report.get("accelerator_available")
    )
    return [
        f"backend={report.get('backend_id')}",
        f"runtime_mode={report.get('runtime_mode')}",
        f"target_policy={report.get('target_policy')}",
        f"torch={_availability(report.get('torch_available'))}",
        f"transformers={_availability(report.get('transformers_available'))}",
        f"accelerator={accelerator_status}",
        f"dependency_status={dependency_status}",
        f"can_emit={_bool_text(report.get('can_emit'))}",
        f"failure_stage={report.get('failure_stage')}",
        f"failure_reason={report.get('failure_reason')}",
        f"remediation={report.get('remediation_hint')}",
        f"fallback_allowed={_bool_text(report.get('fallback_allowed'))}",
        f"fallback_handled_by={report.get('fallback_handled_by')}",
    ]


def remediation_hint_for_failure(
    report: Mapping[str, Any],
    config: TeacherBackendConfig,
) -> str | None:
    stage = str(report.get("failure_stage") or "none")
    status = str(report.get("dependency_status") or "")
    if stage == "none" and bool(report.get("can_emit", False)):
        return None
    if status == "missing_torch":
        return "Install the teacher-hf optional dependency group with torch."
    if status == "missing_transformers":
        return "Install the teacher-hf optional dependency group with transformers."
    if stage == "no_accelerator":
        return "Run on a CUDA or MPS host, or choose cpu_reference."
    if stage == "model_load":
        if config.local_files_only or not config.allow_downloads:
            return (
                "Provide local model/tokenizer files or rerun with downloads "
                "explicitly enabled."
            )
        return "Check model_id, tokenizer_id, and local Hugging Face cache access."
    if stage == "unsupported_target":
        return (
            "Choose a backend/runtime/target-policy combination in the "
            "capability matrix."
        )
    if stage == "invalid_config":
        return "Fix the backend configuration values reported in failure_reason."
    if config.fallback_policy == "auto" and config.backend_id == "gpu_torch":
        return (
            "fallback_policy=auto is recorded, but this builder does not silently "
            "swap gpu_torch to a CPU backend."
        )
    return "Review failure_stage and failure_reason before building."


def _matching_capability(config: TeacherBackendConfig) -> Any | None:
    for capability in list_backend_capabilities():
        if (
            capability.backend_id == config.backend_id
            and capability.runtime_mode == config.runtime_mode
            and capability.target_policy == config.target_policy
        ):
            return capability
    return None


def _gpu_torch_diagnostics(config: TeacherBackendConfig) -> dict[str, Any]:
    from radjax_tome.backends.gpu_torch import diagnose_gpu_torch_backend

    diagnostics = diagnose_gpu_torch_backend(config)
    if diagnostics.get("failure_stage") == "missing_dependency":
        dependency_status = diagnostics.get("dependency_status")
        diagnostics["failure_stage"] = str(dependency_status or "missing_dependency")
    diagnostics.setdefault("fallback_handled_by", "none")
    diagnostics.setdefault("model_available", False)
    diagnostics.setdefault("tokenizer_available", False)
    return diagnostics


def _unsupported_target_reason(config: TeacherBackendConfig) -> str:
    target_policy = config.target_policy
    return (
        f"{config.backend_id} does not implement target_policy={target_policy!r} "
        f"for runtime_mode={config.runtime_mode!r}"
    )


def _availability(value: object) -> str:
    if value is True:
        return "available"
    if value is False:
        return "unavailable"
    return "not_required"


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"
