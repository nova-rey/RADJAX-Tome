"""Teacher backend interfaces and fake/synthetic implementations."""

from __future__ import annotations

from typing import Any

from radjax_tome._lazy_exports import (
    LazyExportMap,
    lazy_export_names,
    resolve_lazy_export,
)
from radjax_tome.backends.base import (
    MIN_CORRIDOR_STAT_TOP_K,
    BackendCapability,
    CpuOrchestrationMode,
    FallbackPolicy,
    RuntimeMode,
    SupportStatus,
    TargetPolicy,
    TeacherBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionBackend,
    TeacherEmissionResult,
    TeacherTargetEmitter,
    gpu_batch_size_candidates,
    resolve_gpu_batch_size_policy,
    validate_gpu_batch_size_policy_config,
)
from radjax_tome.backends.gpu_torch import (
    GPUTorchTeacherEmissionBackend,
    TorchAcceleratorDetection,
    check_gpu_torch_backend_available,
    detect_torch_accelerator,
    diagnose_gpu_torch_backend,
)

_LAZY_EXPORTS: LazyExportMap = {
    "CPUReferenceTeacherEmissionBackend": (
        "radjax_tome.backends.cpu",
        "CPUReferenceTeacherEmissionBackend",
    ),
    "emit_teacher_target_store": (
        "radjax_tome.backends.emission",
        "emit_teacher_target_store",
    ),
    "FakeNumpyTeacherEmissionBackend": (
        "radjax_tome.backends.fake",
        "FakeNumpyTeacherEmissionBackend",
    ),
    "FakeTeacherBackend": ("radjax_tome.backends.fake", "FakeTeacherBackend"),
    "HFTorchTeacherEmissionBackend": (
        "radjax_tome.backends.hf_torch",
        "HFTorchTeacherEmissionBackend",
    ),
    "BackendBatchEnvelope": (
        "radjax_tome.backends.orchestration",
        "BackendBatchEnvelope",
    ),
    "BackendRunConfig": ("radjax_tome.backends.orchestration", "BackendRunConfig"),
    "BackendRunResult": ("radjax_tome.backends.orchestration", "BackendRunResult"),
    "run_backend_batches": (
        "radjax_tome.backends.orchestration",
        "run_backend_batches",
    ),
    "create_backend": ("radjax_tome.backends.registry", "create_backend"),
    "list_backend_capabilities": (
        "radjax_tome.backends.registry",
        "list_backend_capabilities",
    ),
    "register_backend": ("radjax_tome.backends.registry", "register_backend"),
    "SyntheticTeacherBackend": (
        "radjax_tome.backends.synthetic",
        "SyntheticTeacherBackend",
    ),
    "HFTeacherExportConfig": (
        "radjax_tome.backends.hf_export",
        "HFTeacherExportConfig",
    ),
    "HFTeacherExportMetadata": (
        "radjax_tome.backends.hf_export",
        "HFTeacherExportMetadata",
    ),
    "build_hf_export_metadata": (
        "radjax_tome.backends.hf_export",
        "build_hf_export_metadata",
    ),
    "read_hf_export_metadata": (
        "radjax_tome.backends.hf_export",
        "read_hf_export_metadata",
    ),
    "validate_hf_export_config": (
        "radjax_tome.backends.hf_export",
        "validate_hf_export_config",
    ),
    "write_hf_export_metadata": (
        "radjax_tome.backends.hf_export",
        "write_hf_export_metadata",
    ),
    "DEFAULT_HF_SPECIMEN_MODEL_ID": (
        "radjax_tome.backends.hf_specimen",
        "DEFAULT_HF_SPECIMEN_MODEL_ID",
    ),
    "HFTeacherSpecimenConfig": (
        "radjax_tome.backends.hf_specimen",
        "HFTeacherSpecimenConfig",
    ),
    "HFTeacherSpecimenSmokeResult": (
        "radjax_tome.backends.hf_specimen",
        "HFTeacherSpecimenSmokeResult",
    ),
    "HFTeacherSpecimenSwapReport": (
        "radjax_tome.backends.hf_specimen",
        "HFTeacherSpecimenSwapReport",
    ),
    "build_hf_teacher_specimen_dry_run": (
        "radjax_tome.backends.hf_specimen",
        "build_hf_teacher_specimen_dry_run",
    ),
    "build_hf_teacher_specimen_swap_report": (
        "radjax_tome.backends.hf_specimen",
        "build_hf_teacher_specimen_swap_report",
    ),
    "read_hf_teacher_specimen_report": (
        "radjax_tome.backends.hf_specimen",
        "read_hf_teacher_specimen_report",
    ),
    "run_hf_teacher_specimen_smoke": (
        "radjax_tome.backends.hf_specimen",
        "run_hf_teacher_specimen_smoke",
    ),
    "validate_hf_teacher_specimen_config": (
        "radjax_tome.backends.hf_specimen",
        "validate_hf_teacher_specimen_config",
    ),
    "write_hf_teacher_specimen_report": (
        "radjax_tome.backends.hf_specimen",
        "write_hf_teacher_specimen_report",
    ),
    "QwenPolicyEntry": ("radjax_tome.backends.qwen_policy", "QwenPolicyEntry"),
    "QwenPolicyMap": ("radjax_tome.backends.qwen_policy", "QwenPolicyMap"),
    "QwenResolution": ("radjax_tome.backends.qwen_policy", "QwenResolution"),
    "load_qwen_policy": ("radjax_tome.backends.qwen_policy", "load_qwen_policy"),
    "resolve_qwen_policy": (
        "radjax_tome.backends.qwen_policy",
        "resolve_qwen_policy",
    ),
    "resolve_qwen_policy_map": (
        "radjax_tome.backends.qwen_policy",
        "resolve_qwen_policy_map",
    ),
}

__all__ = [
    "BackendCapability",
    "BackendBatchEnvelope",
    "BackendRunConfig",
    "BackendRunResult",
    "CPUReferenceTeacherEmissionBackend",
    "CpuOrchestrationMode",
    "FallbackPolicy",
    "MIN_CORRIDOR_STAT_TOP_K",
    "FakeNumpyTeacherEmissionBackend",
    "FakeTeacherBackend",
    "DEFAULT_HF_SPECIMEN_MODEL_ID",
    "GPUTorchTeacherEmissionBackend",
    "HFTeacherExportConfig",
    "HFTeacherExportMetadata",
    "HFTeacherSpecimenConfig",
    "HFTeacherSpecimenSmokeResult",
    "HFTeacherSpecimenSwapReport",
    "HFTorchTeacherEmissionBackend",
    "QwenPolicyEntry",
    "QwenPolicyMap",
    "QwenResolution",
    "RuntimeMode",
    "SupportStatus",
    "SyntheticTeacherBackend",
    "TargetPolicy",
    "TeacherBackend",
    "TeacherBackendConfig",
    "TeacherBatchInput",
    "TeacherEmissionBackend",
    "TeacherEmissionResult",
    "TeacherTargetEmitter",
    "TorchAcceleratorDetection",
    "build_hf_export_metadata",
    "build_hf_teacher_specimen_dry_run",
    "build_hf_teacher_specimen_swap_report",
    "check_gpu_torch_backend_available",
    "create_backend",
    "diagnose_gpu_torch_backend",
    "detect_torch_accelerator",
    "emit_teacher_target_store",
    "gpu_batch_size_candidates",
    "list_backend_capabilities",
    "load_qwen_policy",
    "read_hf_export_metadata",
    "read_hf_teacher_specimen_report",
    "resolve_gpu_batch_size_policy",
    "resolve_qwen_policy",
    "resolve_qwen_policy_map",
    "run_hf_teacher_specimen_smoke",
    "run_backend_batches",
    "register_backend",
    "validate_hf_export_config",
    "validate_gpu_batch_size_policy_config",
    "validate_hf_teacher_specimen_config",
    "write_hf_export_metadata",
    "write_hf_teacher_specimen_report",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return lazy_export_names(globals(), _LAZY_EXPORTS)
