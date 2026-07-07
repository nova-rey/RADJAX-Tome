from __future__ import annotations

import platform
from collections.abc import Mapping
from importlib import import_module
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
    install = build_gpu_install_diagnostics()

    report: dict[str, Any] = {
        "report_schema": RUNTIME_DOCTOR_REPORT_SCHEMA,
        "python_version": install["python_version"],
        "platform": install["platform"],
        "radjax_tome_import": install["radjax_tome_import"],
        "backend_id": config.backend_id,
        "teacher_backend": config.backend_id,
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
        "gpu_install_diagnostics": install,
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

    report.update(_install_top_level_fields(install))
    report["remediation_hints"] = remediation_hints_for_failure(report, config)
    report["remediation_hint"] = (
        report["remediation_hints"][0] if report["remediation_hints"] else None
    )
    report["recommended_commands"] = recommended_commands_for_report(report)
    return report


def build_gpu_install_diagnostics() -> dict[str, Any]:
    torch_info = _module_info("torch")
    transformers_info = _module_info("transformers")
    jax_info = _module_info("jax")
    torch_module = torch_info["module"]
    cuda_available = False
    cuda_device_count = 0
    cuda_device_names: list[str] = []
    torch_cuda_version = None
    cuda_runtime_version = None
    mps_available = False
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        cuda_available = _safe_bool_call(getattr(cuda, "is_available", None))
        cuda_device_count = _safe_int_call(getattr(cuda, "device_count", None))
        cuda_device_names = _safe_cuda_device_names(cuda, cuda_device_count)
        version = getattr(torch_module, "version", None)
        torch_cuda_version = getattr(version, "cuda", None)
        cuda_runtime_version = torch_cuda_version
        backends = getattr(torch_module, "backends", None)
        mps = getattr(backends, "mps", None)
        mps_available = _safe_bool_call(getattr(mps, "is_available", None))
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "radjax_tome_import": _radjax_tome_import_status(),
        "torch_available": torch_info["available"],
        "torch_version": torch_info["version"],
        "torch_import_error": torch_info["error"],
        "transformers_available": transformers_info["available"],
        "transformers_version": transformers_info["version"],
        "transformers_import_error": transformers_info["error"],
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "cuda_runtime_version": cuda_runtime_version,
        "torch_cuda_version": torch_cuda_version,
        "mps_available": mps_available,
        "jax_available": jax_info["available"],
        "jax_version": jax_info["version"],
        "recommended_install_extra": "gpu-teacher",
    }


def write_runtime_doctor_report(report: Mapping[str, Any], path: Path) -> None:
    write_json(path, dict(report))


def render_runtime_doctor_summary(report: Mapping[str, Any]) -> list[str]:
    dependency_status = report.get("dependency_status")
    accelerator = report.get("device_kind")
    accelerator_status = accelerator or _availability(
        report.get("accelerator_available")
    )
    lines = [
        f"python_version={report.get('python_version')}",
        f"platform={report.get('platform')}",
        f"radjax_tome_import={report.get('radjax_tome_import')}",
        f"torch_available={_bool_text(report.get('torch_available'))}",
        f"torch_version={report.get('torch_version')}",
        f"transformers_available={_bool_text(report.get('transformers_available'))}",
        f"transformers_version={report.get('transformers_version')}",
        f"torch.cuda.is_available={_bool_text(report.get('cuda_available'))}",
        f"cuda_device_count={report.get('cuda_device_count')}",
        *[
            f"cuda_device_{index}_name={name}"
            for index, name in enumerate(report.get("cuda_device_names", ()) or ())
        ],
        f"cuda_runtime_version={report.get('cuda_runtime_version')}",
        f"torch_cuda_version={report.get('torch_cuda_version')}",
        f"mps_available={_bool_text(report.get('mps_available'))}",
        f"jax_available={_bool_text(report.get('jax_available'))}",
        f"backend={report.get('backend_id')}",
        f"teacher_backend={report.get('teacher_backend')}",
        f"runtime_mode={report.get('runtime_mode')}",
        f"target_policy={report.get('target_policy')}",
        f"local_files_only={_bool_text(report.get('local_files_only'))}",
        f"allow_downloads={_bool_text(report.get('allow_downloads'))}",
        f"fallback_policy={report.get('fallback_policy')}",
        f"capability_status={report.get('capability_status')}",
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
    remediation_hints = report.get("remediation_hints", ())
    recommended_commands = report.get("recommended_commands", ())
    lines.extend(f"remediation_hint={hint}" for hint in remediation_hints)
    lines.extend(f"recommended={command}" for command in recommended_commands)
    return lines


def remediation_hints_for_failure(
    report: Mapping[str, Any],
    config: TeacherBackendConfig,
) -> list[str]:
    stage = str(report.get("failure_stage") or "none")
    status = str(report.get("dependency_status") or "")
    if stage == "none" and bool(report.get("can_emit", False)):
        return []
    if status == "missing_torch":
        return ['Install GPU teacher dependencies: pip install -e ".[gpu-teacher]"']
    if status == "missing_transformers":
        return ['Install GPU teacher dependencies: pip install -e ".[gpu-teacher]"']
    if stage == "no_accelerator":
        return [
            "Install/check NVIDIA driver and a CUDA-enabled PyTorch build, "
            "or run on an MPS host, or choose cpu_reference."
        ]
    if stage == "model_load":
        if config.local_files_only or not config.allow_downloads:
            return [
                "Provide or cache the model locally; RADJAX-Tome did not "
                "download a model."
            ]
        return ["Check model_id, tokenizer_id, and local Hugging Face cache access."]
    if stage == "unsupported_target":
        return [
            "Choose a backend/runtime/target-policy combination in the "
            "capability matrix."
        ]
    if stage == "invalid_config":
        return ["Fix the backend configuration values reported in failure_reason."]
    if config.fallback_policy == "auto" and config.backend_id == "gpu_torch":
        return [
            "fallback_policy=auto is recorded, but this builder does not silently "
            "swap gpu_torch to a CPU backend."
        ]
    return ["Review failure_stage and failure_reason before building."]


def remediation_hint_for_failure(
    report: Mapping[str, Any],
    config: TeacherBackendConfig,
) -> str | None:
    hints = remediation_hints_for_failure(report, config)
    return hints[0] if hints else None


def recommended_commands_for_report(report: Mapping[str, Any]) -> list[str]:
    commands = [
        (
            "radjax-tome model inspect --model-path /models/MODEL "
            "--output teacher_model_provenance.json"
        ),
        "radjax-tome corpus build --input ./sources --output ./corpus_out",
        (
            "radjax-tome build --teacher-backend gpu_torch --runtime-mode cpu_gpu "
            "--teacher-model /models/MODEL --teacher-model-provenance "
            "./teacher_model_provenance.json --dataset ./corpus_out/corpus.jsonl "
            "--corpus-manifest ./corpus_out/corpus_manifest.json --output ./gpu_tome"
        ),
        (
            "radjax-tome parity --left ./cpu_artifact --right ./gpu_tome "
            "--output parity_report.json"
        ),
    ]
    if report.get("dependency_status") in {"missing_torch", "missing_transformers"}:
        return ['pip install -e ".[gpu-teacher]"', *commands]
    return commands


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


def _install_top_level_fields(install: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "torch_available": install["torch_available"],
        "torch_version": install["torch_version"],
        "transformers_available": install["transformers_available"],
        "transformers_version": install["transformers_version"],
        "cuda_available": install["cuda_available"],
        "cuda_device_count": install["cuda_device_count"],
        "cuda_device_names": install["cuda_device_names"],
        "cuda_runtime_version": install["cuda_runtime_version"],
        "torch_cuda_version": install["torch_cuda_version"],
        "mps_available": install["mps_available"],
        "jax_available": install["jax_available"],
    }


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


def _module_info(name: str) -> dict[str, Any]:
    try:
        module = import_module(name)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "module": None,
            "version": None,
        }
    return {
        "available": True,
        "error": None,
        "module": module,
        "version": getattr(module, "__version__", None),
    }


def _radjax_tome_import_status() -> str:
    try:
        import_module("radjax_tome")
    except Exception as exc:
        return f"unavailable:{exc}"
    return "ok"


def _safe_bool_call(function: Any) -> bool:
    if not callable(function):
        return False
    try:
        return bool(function())
    except Exception:
        return False


def _safe_int_call(function: Any) -> int:
    if not callable(function):
        return 0
    try:
        return int(function())
    except Exception:
        return 0


def _safe_cuda_device_names(cuda: Any, count: int) -> list[str]:
    get_name = getattr(cuda, "get_device_name", None)
    if not callable(get_name):
        return []
    names: list[str] = []
    for index in range(count):
        try:
            names.append(str(get_name(index)))
        except Exception:
            names.append(f"unavailable:{index}")
    return names
