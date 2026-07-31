"""Compatibility-only backend export registry.

These research/frozen surfaces remain reachable through the public package
façade without making the initializer itself own a research dependency edge.
"""

from radjax_tome._lazy_exports import LazyExportMap

COMPATIBILITY_EXPORTS: LazyExportMap = {
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
