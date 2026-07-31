"""Compatibility-only research report exports."""

from radjax_tome._lazy_exports import LazyExportMap

COMPATIBILITY_EXPORTS: LazyExportMap = {
    "REQUIRED_ARC2_FLAGS": ("radjax_tome.reports.arc", "REQUIRED_ARC2_FLAGS"),
    "FingerprintArcReport": ("radjax_tome.reports.arc", "FingerprintArcReport"),
    "build_fingerprint_arc_report": (
        "radjax_tome.reports.arc",
        "build_fingerprint_arc_report",
    ),
    "read_fingerprint_arc_report": (
        "radjax_tome.reports.arc",
        "read_fingerprint_arc_report",
    ),
    "render_fingerprint_arc_summary": (
        "radjax_tome.reports.arc",
        "render_fingerprint_arc_summary",
    ),
    "write_fingerprint_arc_report": (
        "radjax_tome.reports.arc",
        "write_fingerprint_arc_report",
    ),
    "BaselineArmReport": ("radjax_tome.reports.baseline", "BaselineArmReport"),
    "FingerprintBaselineComparisonReport": (
        "radjax_tome.reports.baseline",
        "FingerprintBaselineComparisonReport",
    ),
    "read_fingerprint_baseline_report": (
        "radjax_tome.reports.baseline",
        "read_fingerprint_baseline_report",
    ),
    "render_fingerprint_baseline_summary": (
        "radjax_tome.reports.baseline",
        "render_fingerprint_baseline_summary",
    ),
    "write_fingerprint_baseline_report": (
        "radjax_tome.reports.baseline",
        "write_fingerprint_baseline_report",
    ),
    "FingerprintArtifactByteBudget": (
        "radjax_tome.reports.fingerprint_quality",
        "FingerprintArtifactByteBudget",
    ),
    "FingerprintQualityPerByteReport": (
        "radjax_tome.reports.fingerprint_quality",
        "FingerprintQualityPerByteReport",
    ),
    "QualityPerByteDelta": (
        "radjax_tome.reports.fingerprint_quality",
        "QualityPerByteDelta",
    ),
    "build_quality_per_byte_delta": (
        "radjax_tome.reports.fingerprint_quality",
        "build_quality_per_byte_delta",
    ),
    "read_fingerprint_quality_report": (
        "radjax_tome.reports.fingerprint_quality",
        "read_fingerprint_quality_report",
    ),
    "render_fingerprint_quality_summary": (
        "radjax_tome.reports.fingerprint_quality",
        "render_fingerprint_quality_summary",
    ),
    "write_fingerprint_quality_report": (
        "radjax_tome.reports.fingerprint_quality",
        "write_fingerprint_quality_report",
    ),
}
