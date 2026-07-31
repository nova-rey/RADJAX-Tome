"""parity ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .payloads import _payload_scalar_summary
from .reporting import _parity_timing_fields


def compare_exemplar_delivery_artifacts(
    path_a: Path,
    path_b: Path,
    *,
    output: Path | None = None,
    atol: float = 1e-6,
    require_selection_match: bool = False,
) -> dict[str, Any]:
    left = _artifact_selection(path_a)
    right = _artifact_selection(path_b)
    blockers: list[str] = []
    warnings: list[str] = []
    selection_differences: list[str] = []
    entropy_allowed_tolerance = max(
        float(atol),
        _artifact_entropy_tolerance(left),
        _artifact_entropy_tolerance(right),
    )
    entropy_deltas = [
        abs(float(left_score) - float(right_score))
        for left_score, right_score in zip(
            left["scores"], right["scores"], strict=False
        )
        if np.isfinite(left_score) and np.isfinite(right_score)
    ]
    entropy_absolute_delta = max(entropy_deltas, default=None)
    entropy_parity_status = "pass"
    if len(left["scores"]) != len(right["scores"]):
        entropy_parity_status = "fail"
    elif any(
        not np.isfinite(left_score) or not np.isfinite(right_score)
        for left_score, right_score in zip(
            left["scores"], right["scores"], strict=False
        )
    ):
        entropy_parity_status = "fail"
    elif entropy_absolute_delta is not None and (
        entropy_absolute_delta > entropy_allowed_tolerance
    ):
        entropy_parity_status = "fail"
    coordinate_exact_match = (
        left["ids"] == right["ids"]
        and left["positions"] == right["positions"]
        and left.get("source_coordinates", []) == right.get("source_coordinates", [])
    )
    top_token_exact_match = left.get("top_token_ids", []) == right.get(
        "top_token_ids", []
    )
    if left["ids"] != right["ids"]:
        selection_differences.append("selected example IDs differ")
    if left["positions"] != right["positions"]:
        selection_differences.append("selected positions differ")
    if left["ranks"] != right["ranks"]:
        selection_differences.append("selected score ranks differ")
    for index, (left_score, right_score) in enumerate(
        zip(left["scores"], right["scores"], strict=False)
    ):
        if not np.isfinite(left_score) or not np.isfinite(right_score):
            selection_differences.append(
                f"selected score is nonfinite at rank {index + 1}"
            )
            break
        if abs(left_score - right_score) > entropy_allowed_tolerance:
            selection_differences.append(f"selected score differs at rank {index + 1}")
            break
    if not coordinate_exact_match:
        selection_differences.append("selected source coordinates differ")
    if not top_token_exact_match:
        selection_differences.append("selected top-token identities differ")
    if left["mode_keys"] != right["mode_keys"]:
        selection_differences.append("selected mode keys differ")
    if entropy_parity_status == "fail":
        blockers.append("selected entropy parity exceeds quantization tolerance")
    if not coordinate_exact_match and (
        require_selection_match
        or left.get("source_coordinates")
        or right.get("source_coordinates")
    ):
        blockers.append("selected source coordinates do not match exactly")
    if not top_token_exact_match and (
        require_selection_match
        or left.get("top_token_ids")
        or right.get("top_token_ids")
    ):
        blockers.append("selected top-token identities do not match exactly")
    if require_selection_match:
        blockers.extend(selection_differences)
    elif selection_differences:
        warnings.append(
            "selected identities differ; structural parity checks remain authoritative"
        )
    if not all(status == "linked" for status in left["assignment_statuses"]):
        blockers.append("Path A selected corridor assignments are not linked")
    if not all(status == "linked" for status in right["assignment_statuses"]):
        blockers.append("Path B selected corridor assignments are not linked")
    if left["payload_shapes"] != right["payload_shapes"]:
        blockers.append("compressed exemplar payload shapes differ")
    if left["corridor_shape"] != right["corridor_shape"]:
        blockers.append("corridor artifact shapes differ")
    if left["corridor_mode_policy"] != right["corridor_mode_policy"]:
        blockers.append("corridor mode policies differ")
    if left["corridor_mode_count"] != right["corridor_mode_count"]:
        blockers.append("corridor mode counts differ")
    if left["corridor_tracked_stats"] != right["corridor_tracked_stats"]:
        blockers.append("corridor tracked stats differ")
    if left["corridor_mode_table"] != right["corridor_mode_table"]:
        blockers.append("corridor mode tables differ")
    if (
        left["corridor_assignment_storage_kind"]
        != right["corridor_assignment_storage_kind"]
    ):
        blockers.append("corridor assignment storage kinds differ")
    for label, artifact in (("Path A", left), ("Path B", right)):
        if artifact["report"].get("corridor_artifact_built") is not True:
            blockers.append(f"{label} did not build corridor artifacts")
        if artifact["report"].get("corridor_modes_built") is not True:
            blockers.append(f"{label} did not build corridor modes")
        if int(artifact["report"].get("corridor_mode_count") or 0) < 1:
            blockers.append(f"{label} corridor mode count is zero")
    if right["report"].get("non_selected_exemplar_payload_retained") is True:
        blockers.append("Path B retained non-selected exemplar payloads")
    left_retained = int(left["report"].get("final_retained_bytes") or 0)
    right_retained = int(right["report"].get("final_retained_bytes") or 0)
    left_pruned = left["report"].get("non_selected_exemplar_payload_retained") is False
    if right_retained >= left_retained and not left_pruned:
        warnings.append(
            "Path B retained bytes are not smaller than unpruned Path A bytes"
        )
    report = {
        "schema_version": EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA,
        "status": "fail" if blockers else "warn" if warnings else "pass",
        "blockers": blockers,
        "warnings": warnings,
        "path_a": str(path_a),
        "path_b": str(path_b),
        "selection_match_required": require_selection_match,
        "selection_differences": selection_differences,
        "selected_example_ids_match": left["ids"] == right["ids"],
        "selected_positions_match": left["positions"] == right["positions"],
        "selected_score_ranks_match": left["ranks"] == right["ranks"],
        "coordinate_exact_match": coordinate_exact_match,
        "top_token_exact_match": top_token_exact_match,
        "entropy_absolute_delta": entropy_absolute_delta,
        "entropy_allowed_tolerance": entropy_allowed_tolerance,
        "entropy_parity_status": entropy_parity_status,
        "entropy_deltas": entropy_deltas,
        "selected_mode_keys_match": left["mode_keys"] == right["mode_keys"],
        "selected_corridor_mode_ids_match": left["mode_keys"] == right["mode_keys"],
        "selected_corridor_assignments_linked": all(
            status == "linked"
            for status in (*left["assignment_statuses"], *right["assignment_statuses"])
        ),
        "payload_shape_compatible": left["payload_shapes"] == right["payload_shapes"],
        "corridor_artifact_shape_match": left["corridor_shape"]
        == right["corridor_shape"],
        "corridor_mode_policy_match": left["corridor_mode_policy"]
        == right["corridor_mode_policy"],
        "corridor_mode_count_match": left["corridor_mode_count"]
        == right["corridor_mode_count"],
        "corridor_tracked_stats_match": left["corridor_tracked_stats"]
        == right["corridor_tracked_stats"],
        "corridor_mode_table_match": left["corridor_mode_table"]
        == right["corridor_mode_table"],
        "corridor_assignment_storage_kind_match": left[
            "corridor_assignment_storage_kind"
        ]
        == right["corridor_assignment_storage_kind"],
        "path_a_corridor_artifact_built": left["report"].get("corridor_artifact_built"),
        "path_b_corridor_artifact_built": right["report"].get(
            "corridor_artifact_built"
        ),
        "path_a_corridor_mode_count": left["report"].get("corridor_mode_count"),
        "path_b_corridor_mode_count": right["report"].get("corridor_mode_count"),
        "path_a_corridor_mode_policy": left["corridor_mode_policy"],
        "path_b_corridor_mode_policy": right["corridor_mode_policy"],
        "path_a_corridor_assignment_storage_kind": left[
            "corridor_assignment_storage_kind"
        ],
        "path_b_corridor_assignment_storage_kind": right[
            "corridor_assignment_storage_kind"
        ],
        "path_a_retained_bytes": left_retained,
        "path_b_retained_bytes": right_retained,
        "path_a_teacher_rerun_count": left["report"].get("teacher_rerun_count"),
        "path_b_teacher_rerun_count": right["report"].get("teacher_rerun_count"),
        "path_b_non_selected_exemplar_payload_retained": right["report"].get(
            "non_selected_exemplar_payload_retained"
        ),
    }
    timing = _parity_timing_fields(left["report"], right["report"])
    if timing:
        report.update(timing)
    if output is not None:
        write_json(output, report)
    return report


def _artifact_selection(path: Path) -> dict[str, Any]:
    report = read_json_object(path / EXEMPLAR_DELIVERY_REPORT_FILENAME)
    selected = _read_selected_exemplars(
        path / "leaderboards" / SELECTED_EXEMPLARS_FILENAME,
        [],
    )
    payloads = _read_selected_payloads(path / "selected_exemplars", [])
    corridor_modes = _corridor_modes_payload(path)
    return {
        "report": report,
        "ids": [item.get("selected_example_id") for item in selected],
        "positions": [item.get("selected_position") for item in selected],
        "ranks": [item.get("rank") for item in selected],
        "scores": [float(item.get("selected_score") or 0.0) for item in selected],
        "top_token_ids": [item.get("source_top_token_id") for item in selected],
        "source_coordinates": [
            [
                item.get("source_shard_id"),
                item.get("source_row"),
                item.get("source_position"),
            ]
            for item in selected
        ],
        "mode_keys": [item.get("corridor_mode_id") for item in selected],
        "assignment_statuses": [
            item.get("corridor_assignment_status") for item in selected
        ],
        "payload_shapes": [_payload_shape(item) for item in payloads],
        "corridor_shape": _corridor_artifact_shape(path),
        "corridor_mode_policy": corridor_modes.get("mode_policy"),
        "corridor_mode_count": corridor_modes.get("mode_count"),
        "corridor_tracked_stats": corridor_modes.get("tracked_stats", []),
        "corridor_mode_table": _normalized_mode_table(corridor_modes),
        "corridor_assignment_storage_kind": report.get(
            "corridor_assignment_storage_kind"
        ),
        "entropy_quantization_step": report.get(
            "entropy_quantization_step",
            ENTROPY_PARITY_QUANTIZATION_STEP,
        ),
    }


def _artifact_entropy_tolerance(artifact: Mapping[str, Any]) -> float:
    value = artifact.get("entropy_quantization_step")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ENTROPY_PARITY_QUANTIZATION_STEP
    if not np.isfinite(numeric) or numeric < 0.0:
        return ENTROPY_PARITY_QUANTIZATION_STEP
    return numeric


def _corridor_artifact_shape(path: Path) -> tuple[str, ...]:
    return tuple(
        relative_path
        for relative_path in (
            "corridors/corridor_summary.json",
            "corridors/corridor_fingerprints.json",
            "corridors/corridor_modes.json",
            "corridors/mode_assignments.json",
        )
        if (path / relative_path).is_file()
    )


def _corridor_modes_payload(path: Path) -> dict[str, Any]:
    try:
        return read_json_object(path / "corridors" / "corridor_modes.json")
    except (OSError, ValueError):
        return {}


def _normalized_mode_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    modes = payload.get("modes", [])
    if not isinstance(modes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for mode in modes:
        if not isinstance(mode, dict):
            continue
        normalized.append(
            {
                "mode_id": mode.get("mode_id"),
                "mode_key": mode.get("mode_key"),
                "record_count": mode.get("record_count"),
                "bounds": mode.get("bounds"),
            }
        )
    return normalized


def _payload_shape(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "top_token_ids": len(payload.get("top_token_ids", [])),
        "top_log_probs": len(payload.get("top_log_probs", [])),
        "top_probs": len(payload.get("top_probs", [])),
        "top_selection_mask": len(payload.get("top_selection_mask", [])),
        "bucket_masses": len(payload.get("bucket_masses", [])),
    }


def _read_selected_exemplars(
    path: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    try:
        payload = read_json_object(path)
    except (OSError, ValueError) as exc:
        blockers.append(f"selected_exemplars.json invalid: {exc}")
        return []
    selected = payload.get("selected_exemplars", [])
    if not isinstance(selected, list):
        blockers.append("selected_exemplars.json selected_exemplars must be a list")
        return []
    return [item for item in selected if isinstance(item, dict)]


def _read_selected_payloads(
    selected_dir: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    if not selected_dir.is_dir():
        blockers.append("selected_exemplars directory missing")
        return []
    payloads: list[dict[str, Any]] = []
    for path in sorted(selected_dir.glob("selected-exemplars-*.json")):
        try:
            payload = read_json_object(path)
        except (OSError, ValueError) as exc:
            blockers.append(f"{path.name} invalid: {exc}")
            continue
        records = payload.get("selected_exemplars", [])
        if isinstance(records, list):
            payloads.extend(item for item in records if isinstance(item, dict))
    if not payloads:
        blockers.append("selected exemplar payloads are missing")
    return payloads


def _read_selected_payload_summaries(
    selected_dir: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    """Validate payload shards one at a time, retaining only scalar state."""

    if not selected_dir.is_dir():
        blockers.append("selected_exemplars directory missing")
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(selected_dir.glob("selected-exemplars-*.json")):
        try:
            envelope = read_json_object(path)
        except (OSError, ValueError) as exc:
            blockers.append(f"{path.name} invalid: {exc}")
            continue
        records = envelope.get("selected_exemplars", [])
        if not isinstance(records, list):
            blockers.append(f"{path.name} selected_exemplars is invalid")
            continue
        for item in records:
            if not isinstance(item, dict):
                blockers.append(f"{path.name} selected exemplar is invalid")
                continue
            summary = _payload_scalar_summary(
                item,
                record_index=int(envelope.get("record_index", len(summaries))),
            )
            summary["payload_hash"] = envelope.get("payload_hash")
            top_token_id = item.get("top_token_ids", [None])
            if isinstance(top_token_id, list):
                summary["top_token_ids"] = top_token_id[:1]
            for key in (
                "top_log_probs",
                "top_probs",
                "top_selection_mask",
                "bucket_masses",
            ):
                if key in item:
                    value = item[key]
                    summary[key] = value[:1] if key != "bucket_masses" else value
            summaries.append(summary)
    if not summaries:
        blockers.append("selected exemplar payloads are missing")
    return summaries


def _metadata_int(artifact_dir: Path, key: str) -> int | None:
    try:
        metadata = read_json_object(artifact_dir / "metadata.json")
    except (OSError, ValueError):
        return None
    value = metadata.get(key)
    return int(value) if value is not None else None


__all__ = [name for name in globals() if not name.startswith("__")]
