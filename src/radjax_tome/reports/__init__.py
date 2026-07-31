"""Producer-side report schemas for RADJAX-Tome."""

from __future__ import annotations

from typing import Any

from radjax_tome._lazy_exports import (
    lazy_export_names,
    resolve_lazy_export,
)
from radjax_tome.reports.compatibility_exports import COMPATIBILITY_EXPORTS
from radjax_tome.reports.metadata_sanity import (
    ARTIFACT_METADATA_SANITY_REPORT_SCHEMA,
    METADATA_SANITY_REPORT_FILENAME,
    build_artifact_metadata_sanity_report,
    render_artifact_metadata_sanity_summary,
    write_artifact_metadata_sanity_report,
)
from radjax_tome.reports.parity import (
    TOME_PARITY_REPORT_FILENAME,
    TOME_PARITY_REPORT_SCHEMA,
    TomeParityConfig,
    TomeParityReport,
    compare_tome_artifacts,
    write_tome_parity_report,
)
from radjax_tome.reports.rendering import markdown_table, status_line
from radjax_tome.reports.run_plan import (
    GPU_RUN_PLAN_FILENAME,
    GPU_RUN_PLAN_SCHEMA,
    GPURunPlanConfig,
    build_gpu_run_plan,
    render_gpu_run_plan_summary,
    run_gpu_torch_auto_batch_probe,
    write_gpu_run_plan,
)
from radjax_tome.reports.runtime_doctor import (
    RUNTIME_DOCTOR_REPORT_SCHEMA,
    build_runtime_doctor_report,
    render_runtime_doctor_summary,
    write_runtime_doctor_report,
)
from radjax_tome.reports.writers import write_json_report, write_markdown_report

_LAZY_EXPORTS = COMPATIBILITY_EXPORTS

__all__ = [
    "ARTIFACT_METADATA_SANITY_REPORT_SCHEMA",
    "BaselineArmReport",
    "FingerprintArcReport",
    "FingerprintArtifactByteBudget",
    "FingerprintBaselineComparisonReport",
    "FingerprintQualityPerByteReport",
    "GPU_RUN_PLAN_FILENAME",
    "GPU_RUN_PLAN_SCHEMA",
    "GPURunPlanConfig",
    "METADATA_SANITY_REPORT_FILENAME",
    "QualityPerByteDelta",
    "REQUIRED_ARC2_FLAGS",
    "RUNTIME_DOCTOR_REPORT_SCHEMA",
    "TOME_PARITY_REPORT_FILENAME",
    "TOME_PARITY_REPORT_SCHEMA",
    "TomeParityConfig",
    "TomeParityReport",
    "build_artifact_metadata_sanity_report",
    "build_fingerprint_arc_report",
    "build_gpu_run_plan",
    "build_quality_per_byte_delta",
    "build_runtime_doctor_report",
    "compare_tome_artifacts",
    "read_fingerprint_arc_report",
    "read_fingerprint_baseline_report",
    "read_fingerprint_quality_report",
    "render_artifact_metadata_sanity_summary",
    "render_fingerprint_arc_summary",
    "render_fingerprint_baseline_summary",
    "render_fingerprint_quality_summary",
    "render_gpu_run_plan_summary",
    "render_runtime_doctor_summary",
    "run_gpu_torch_auto_batch_probe",
    "markdown_table",
    "status_line",
    "write_artifact_metadata_sanity_report",
    "write_fingerprint_arc_report",
    "write_fingerprint_baseline_report",
    "write_fingerprint_quality_report",
    "write_json_report",
    "write_markdown_report",
    "write_gpu_run_plan",
    "write_runtime_doctor_report",
    "write_tome_parity_report",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return lazy_export_names(globals(), _LAZY_EXPORTS)
