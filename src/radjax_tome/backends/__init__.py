"""Teacher backend interfaces and fake/synthetic implementations."""

from radjax_tome.backends.base import (
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
)
from radjax_tome.backends.emission import emit_teacher_target_store
from radjax_tome.backends.fake import (
    FakeNumpyTeacherEmissionBackend,
    FakeTeacherBackend,
)
from radjax_tome.backends.hf_export import (
    HFTeacherExportConfig,
    HFTeacherExportMetadata,
    build_hf_export_metadata,
    read_hf_export_metadata,
    validate_hf_export_config,
    write_hf_export_metadata,
)
from radjax_tome.backends.hf_specimen import (
    DEFAULT_HF_SPECIMEN_MODEL_ID,
    HFTeacherSpecimenConfig,
    HFTeacherSpecimenSmokeResult,
    HFTeacherSpecimenSwapReport,
    build_hf_teacher_specimen_dry_run,
    build_hf_teacher_specimen_swap_report,
    read_hf_teacher_specimen_report,
    run_hf_teacher_specimen_smoke,
    validate_hf_teacher_specimen_config,
    write_hf_teacher_specimen_report,
)
from radjax_tome.backends.qwen_policy import (
    QwenPolicyEntry,
    QwenPolicyMap,
    QwenResolution,
    load_qwen_policy,
    resolve_qwen_policy,
    resolve_qwen_policy_map,
)
from radjax_tome.backends.registry import (
    create_backend,
    list_backend_capabilities,
    register_backend,
)
from radjax_tome.backends.synthetic import SyntheticTeacherBackend

__all__ = [
    "BackendCapability",
    "CpuOrchestrationMode",
    "FallbackPolicy",
    "FakeNumpyTeacherEmissionBackend",
    "FakeTeacherBackend",
    "DEFAULT_HF_SPECIMEN_MODEL_ID",
    "HFTeacherExportConfig",
    "HFTeacherExportMetadata",
    "HFTeacherSpecimenConfig",
    "HFTeacherSpecimenSmokeResult",
    "HFTeacherSpecimenSwapReport",
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
    "build_hf_export_metadata",
    "build_hf_teacher_specimen_dry_run",
    "build_hf_teacher_specimen_swap_report",
    "create_backend",
    "emit_teacher_target_store",
    "list_backend_capabilities",
    "load_qwen_policy",
    "read_hf_export_metadata",
    "read_hf_teacher_specimen_report",
    "resolve_qwen_policy",
    "resolve_qwen_policy_map",
    "run_hf_teacher_specimen_smoke",
    "register_backend",
    "validate_hf_export_config",
    "validate_hf_teacher_specimen_config",
    "write_hf_export_metadata",
    "write_hf_teacher_specimen_report",
]
