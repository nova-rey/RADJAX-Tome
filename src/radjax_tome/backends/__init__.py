"""Teacher backend interfaces and fake/synthetic implementations."""

from radjax_tome.backends.base import TeacherBackend, TeacherTargetEmitter
from radjax_tome.backends.emission import emit_teacher_target_store
from radjax_tome.backends.fake import FakeTeacherBackend
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
from radjax_tome.backends.synthetic import SyntheticTeacherBackend

__all__ = [
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
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "TeacherTargetEmitter",
    "build_hf_export_metadata",
    "build_hf_teacher_specimen_dry_run",
    "build_hf_teacher_specimen_swap_report",
    "emit_teacher_target_store",
    "load_qwen_policy",
    "read_hf_export_metadata",
    "read_hf_teacher_specimen_report",
    "resolve_qwen_policy",
    "resolve_qwen_policy_map",
    "run_hf_teacher_specimen_smoke",
    "validate_hf_export_config",
    "validate_hf_teacher_specimen_config",
    "write_hf_export_metadata",
    "write_hf_teacher_specimen_report",
]
