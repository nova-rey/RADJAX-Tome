"""Producer-side report schemas for RADJAX-Tome."""

from radjax_tome.reports.arc import (
    REQUIRED_ARC2_FLAGS,
    FingerprintArcReport,
    build_fingerprint_arc_report,
    read_fingerprint_arc_report,
    render_fingerprint_arc_summary,
    write_fingerprint_arc_report,
)
from radjax_tome.reports.baseline import (
    BaselineArmReport,
    FingerprintBaselineComparisonReport,
    read_fingerprint_baseline_report,
    render_fingerprint_baseline_summary,
    write_fingerprint_baseline_report,
)
from radjax_tome.reports.fingerprint_quality import (
    FingerprintArtifactByteBudget,
    FingerprintQualityPerByteReport,
    QualityPerByteDelta,
    build_quality_per_byte_delta,
    read_fingerprint_quality_report,
    render_fingerprint_quality_summary,
    write_fingerprint_quality_report,
)
from radjax_tome.reports.metadata_sanity import (
    ARTIFACT_METADATA_SANITY_REPORT_SCHEMA,
    METADATA_SANITY_REPORT_FILENAME,
    build_artifact_metadata_sanity_report,
    render_artifact_metadata_sanity_summary,
    write_artifact_metadata_sanity_report,
)
from radjax_tome.reports.rendering import markdown_table, status_line
from radjax_tome.reports.runtime_doctor import (
    RUNTIME_DOCTOR_REPORT_SCHEMA,
    build_runtime_doctor_report,
    render_runtime_doctor_summary,
    write_runtime_doctor_report,
)
from radjax_tome.reports.writers import write_json_report, write_markdown_report

__all__ = [
    "ARTIFACT_METADATA_SANITY_REPORT_SCHEMA",
    "BaselineArmReport",
    "FingerprintArcReport",
    "FingerprintArtifactByteBudget",
    "FingerprintBaselineComparisonReport",
    "FingerprintQualityPerByteReport",
    "METADATA_SANITY_REPORT_FILENAME",
    "QualityPerByteDelta",
    "REQUIRED_ARC2_FLAGS",
    "RUNTIME_DOCTOR_REPORT_SCHEMA",
    "build_artifact_metadata_sanity_report",
    "build_fingerprint_arc_report",
    "build_quality_per_byte_delta",
    "build_runtime_doctor_report",
    "read_fingerprint_arc_report",
    "read_fingerprint_baseline_report",
    "read_fingerprint_quality_report",
    "render_artifact_metadata_sanity_summary",
    "render_fingerprint_arc_summary",
    "render_fingerprint_baseline_summary",
    "render_fingerprint_quality_summary",
    "render_runtime_doctor_summary",
    "markdown_table",
    "status_line",
    "write_artifact_metadata_sanity_report",
    "write_fingerprint_arc_report",
    "write_fingerprint_baseline_report",
    "write_fingerprint_quality_report",
    "write_json_report",
    "write_markdown_report",
    "write_runtime_doctor_report",
]
