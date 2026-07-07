from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

ARTIFACT_METADATA_SANITY_REPORT_SCHEMA = "artifact_metadata_sanity_report_v1"
METADATA_SANITY_REPORT_FILENAME = "metadata_sanity_report.json"

_SUMMARY_FIELDS = (
    "target_type",
    "target_policy",
    "backend_id",
    "effective_backend_id",
    "runtime_mode",
    "effective_runtime_mode",
    "fallback_used",
    "fallback_policy",
    "fallback_handled_by",
    "capability_status",
    "optimized_path_used",
    "dense_logits_transferred_to_host",
    "compact_reduction_used",
    "gpu_compact_reduction_implemented",
    "gpu_reduction_mode",
    "compact_payload_fields",
    "compact_payload_arrays",
    "compact_bytes_transferred_to_host",
    "estimated_dense_logits_bytes",
    "estimated_reducer_workspace_bytes",
    "estimated_reducer_workspace_is_measured",
    "exemplar_capture_mode_requested",
    "exemplar_capture_mode_effective",
    "exemplar_capture_policy",
    "exemplar_capture_stage",
    "exemplar_candidate_scope",
    "corpus_level_exemplar_finalization",
    "requires_second_pass_for_final_exemplars",
    "rerun_teacher_for_selected_examples",
    "exemplar_selection_enabled",
    "exemplar_selector_policy",
    "exemplar_selection_manifest_path",
    "exemplar_selection_manifest_schema",
    "exemplar_fulfillment_policy",
    "selection_application",
    "deduplication_policy",
    "duplicate_candidate_count",
    "backfill_success_count",
    "score_aware_budget_trimming",
    "budget_trimming_policy",
    "budget_applied",
    "production_global_selector",
    "semantic_diversity_used",
    "utility_calibrated",
    "gpu_batch_size_mode_requested",
    "gpu_batch_size_mode_effective",
    "effective_gpu_batch_size",
    "gpu_batch_size_warning_emitted",
    "batch_size_policy_uses_estimates",
    "estimated_bytes_are_calibrated",
    "measured_output_bytes_available",
    "measured_output_bytes",
    "measured_compact_bytes_transferred_to_host",
    "estimated_to_measured_bytes_ratio",
    "multidevice_enabled",
    "batch_partition_strategy",
)


def build_artifact_metadata_sanity_report(path: Path) -> dict[str, Any]:
    artifact_path = Path(path)
    blockers: list[str] = []
    warnings: list[str] = []
    metadata = _read_optional_json(artifact_path / "metadata.json", blockers)
    params = _normalise_mapping(metadata.get("target_params", {}))
    target_type = _string_or_none(metadata.get("target_type"))
    target_policy = _string_or_none(params.get("target_policy")) or target_type
    report: dict[str, Any] = {
        "report_schema": ARTIFACT_METADATA_SANITY_REPORT_SCHEMA,
        "artifact_path": str(artifact_path),
        "status": "fail" if blockers else "pass",
        "blockers": blockers,
        "warnings": warnings,
        "target_type": target_type,
        "target_policy": target_policy,
        "backend_id": _string_or_none(
            params.get("requested_backend_id") or params.get("backend_id")
        ),
        "effective_backend_id": _string_or_none(
            params.get("effective_backend_id") or params.get("backend_id")
        ),
        "runtime_mode": _string_or_none(
            params.get("requested_runtime_mode") or params.get("runtime_mode")
        ),
        "effective_runtime_mode": _string_or_none(
            params.get("effective_runtime_mode") or params.get("runtime_mode")
        ),
        "fallback_used": _bool_or_none(params.get("fallback_used")),
        "fallback_policy": _string_or_none(params.get("fallback_policy")),
        "fallback_handled_by": _string_or_none(
            params.get("fallback_handled_by") or "none"
        ),
        "capability_status": _string_or_none(params.get("capability_status")),
        "optimized_path_used": _bool_or_none(params.get("optimized_path_used")),
    }
    for field in _SUMMARY_FIELDS:
        report.setdefault(field, params.get(field))

    _add_selection_manifest_summary(artifact_path, report, params, warnings)
    _check_score_pass_metadata(report, params, blockers)
    _check_corridor_metadata(report, params, blockers)
    _check_selector_metadata(report, params, blockers, warnings)
    _check_backend_runtime_metadata(report, params, blockers)
    _check_batch_metadata(report, params, blockers)
    _warn_for_missing_claimed_metadata(report, params, warnings)

    report["status"] = _status(blockers, warnings)
    return report


def write_artifact_metadata_sanity_report(
    report: Mapping[str, Any],
    path: Path,
) -> None:
    write_json(path, dict(report))


def render_artifact_metadata_sanity_summary(report: Mapping[str, Any]) -> list[str]:
    return [
        f"metadata_sanity_status={report.get('status')}",
        f"metadata_sanity_blockers={len(report.get('blockers', ()))}",
        f"metadata_sanity_warnings={len(report.get('warnings', ()))}",
        f"metadata_sanity_target={report.get('target_type')}",
        f"metadata_sanity_backend={report.get('backend_id')}->{report.get('effective_backend_id')}",
        f"metadata_sanity_capture_stage={report.get('exemplar_capture_stage')}",
        f"metadata_sanity_selection_manifest={report.get('exemplar_selection_manifest_path')}",
        f"metadata_sanity_deduplication_policy={report.get('deduplication_policy')}",
        f"metadata_sanity_production_global_selector={report.get('production_global_selector')}",
        f"metadata_sanity_semantic_diversity_used={report.get('semantic_diversity_used')}",
        f"metadata_sanity_utility_calibrated={report.get('utility_calibrated')}",
    ]


def _read_optional_json(path: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        return read_json_object(path)
    except ValueError as exc:
        blockers.append(str(exc))
        return {}


def _normalise_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _normalise_value(item) for key, item in value.items()}


def _normalise_value(value: object) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    try:
        if any(char in stripped for char in (".", "e", "E")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return value


def _add_selection_manifest_summary(
    artifact_path: Path,
    report: dict[str, Any],
    params: Mapping[str, Any],
    warnings: list[str],
) -> None:
    manifest_name = _string_or_none(params.get("exemplar_selection_manifest_path"))
    if manifest_name is None:
        return
    manifest_path = artifact_path / manifest_name
    if not manifest_path.is_file():
        warnings.append(
            f"selection manifest path is recorded but missing: {manifest_name}"
        )
        return
    manifest = read_json_object(manifest_path)
    report.setdefault(
        "exemplar_selection_manifest_schema", manifest.get("schema_version")
    )
    for report_key, manifest_key in (
        ("exemplar_selector_policy", "selection_policy"),
        ("exemplar_fulfillment_policy", "fulfillment_policy"),
        ("selection_application", "selection_application"),
        ("deduplication_policy", "deduplication_policy"),
        ("duplicate_candidate_count", "duplicate_candidate_count"),
        ("backfill_success_count", "backfill_success_count"),
        ("score_aware_budget_trimming", "score_aware_budget_trimming"),
        ("budget_trimming_policy", "budget_trimming_policy"),
        ("budget_applied", "budget_applied"),
        ("production_global_selector", "production_global_selector"),
        ("semantic_diversity_used", "semantic_diversity_used"),
        ("utility_calibrated", "utility_calibrated"),
    ):
        if report.get(report_key) is None and manifest_key in manifest:
            report[report_key] = manifest[manifest_key]


def _check_score_pass_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    blockers: list[str],
) -> None:
    if report.get("target_type") != "corridor_exemplar_score_pass_v1":
        return
    if report.get("exemplar_capture_stage") != "score_pass":
        blockers.append(
            "score-pass target must record exemplar_capture_stage=score_pass"
        )
    if _present(params, "production_corridor_schema") and _bool_or_none(
        params.get("production_corridor_schema")
    ):
        blockers.append("score-pass target must not claim production_corridor_schema")
    if report.get("requires_second_pass_for_final_exemplars") is not True:
        blockers.append(
            "score-pass target must require a second pass for final exemplars"
        )


def _check_corridor_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    blockers: list[str],
) -> None:
    if report.get("target_type") != "corridor_exemplar_v1":
        return
    if _present(params, "schema_version") and params.get("schema_version") != (
        "corridor_exemplar_v1"
    ):
        blockers.append("corridor target schema_version must be corridor_exemplar_v1")
    if (
        _present(params, "production_corridor_schema")
        and _bool_or_none(params.get("production_corridor_schema")) is not True
    ):
        blockers.append("corridor target must truthfully claim production schema")


def _check_selector_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> None:
    for field in (
        "production_global_selector",
        "semantic_diversity_used",
        "utility_calibrated",
    ):
        if _bool_or_none(report.get(field)) is True:
            blockers.append(f"{field} must remain false until implemented")

    fulfillment = report.get("exemplar_fulfillment_policy")
    if fulfillment == "rerun_selected_capture":
        if _bool_or_none(params.get("rerun_manifest_ready")) is not True:
            blockers.append("rerun fulfillment requires rerun_manifest_ready=true")
        if not _present(params, "selected_pass_rerun_performed"):
            blockers.append(
                "rerun fulfillment requires selected_pass_rerun_performed metadata"
            )
    if (
        fulfillment == "select_from_existing_capture"
        and _bool_or_none(params.get("selected_from_existing_capture")) is not True
    ):
        blockers.append(
            "select_from_existing_capture requires selected_from_existing_capture=true"
        )

    if _bool_or_none(report.get("exemplar_selection_enabled")) is True:
        for field in (
            "exemplar_selector_policy",
            "exemplar_selection_manifest_path",
            "deduplication_policy",
            "exemplar_fulfillment_policy",
        ):
            if report.get(field) in {None, ""}:
                warnings.append(f"selection metadata missing {field}")


def _check_backend_runtime_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    blockers: list[str],
) -> None:
    effective_backend = report.get("effective_backend_id")
    if effective_backend == "gpu_torch":
        if report.get("effective_runtime_mode") != "cpu_gpu":
            blockers.append("gpu_torch effective backend requires cpu_gpu runtime")
        if _bool_or_none(report.get("fallback_used")) is True:
            blockers.append(
                "gpu_torch effective backend cannot also claim fallback_used"
            )

    requested_backend = report.get("backend_id")
    if requested_backend == "gpu_torch" and effective_backend != "gpu_torch":
        explicit_fallback = (
            _bool_or_none(report.get("fallback_used")) is True
            and report.get("fallback_policy") == "auto"
            and report.get("fallback_handled_by") not in {None, "none", ""}
        )
        if not explicit_fallback:
            blockers.append(
                "requested gpu_torch with different effective backend needs "
                "explicit fallback metadata"
            )

    if (
        _present(params, "fallback_used")
        and _bool_or_none(params.get("fallback_used")) is True
        and report.get("fallback_handled_by") in {None, "none", ""}
    ):
        blockers.append("fallback_used requires fallback_handled_by metadata")


def _check_batch_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    blockers: list[str],
) -> None:
    multidevice = _bool_or_none(report.get("multidevice_enabled"))
    if (
        multidevice is False
        and report.get("batch_partition_strategy") != "single_device"
    ):
        blockers.append(
            "multidevice_enabled=false requires batch_partition_strategy=single_device"
        )
    if _present(params, "gpu_batch_size_mode_requested") and report.get(
        "effective_gpu_batch_size"
    ) in {None, ""}:
        blockers.append("GPU batch metadata must include effective_gpu_batch_size")


def _warn_for_missing_claimed_metadata(
    report: Mapping[str, Any],
    params: Mapping[str, Any],
    warnings: list[str],
) -> None:
    backend_routed = report.get("backend_id") is not None or _present(
        params, "artifact_emission_path"
    )
    if not backend_routed:
        return
    claimed_fields = (
        "effective_backend_id",
        "effective_runtime_mode",
        "fallback_used",
        "fallback_policy",
        "optimized_path_used",
        "gpu_batch_size_mode_requested",
        "gpu_batch_size_mode_effective",
        "effective_gpu_batch_size",
        "batch_partition_strategy",
        "multidevice_enabled",
    )
    for field in claimed_fields:
        if report.get(field) is None:
            warnings.append(f"backend metadata missing {field}")


def _status(blockers: list[str], warnings: list[str]) -> str:
    if blockers:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def _present(mapping: Mapping[str, Any], key: str) -> bool:
    return key in mapping and mapping[key] is not None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
