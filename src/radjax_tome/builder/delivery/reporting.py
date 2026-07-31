"""reporting ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .payloads import _primary_budget


def _delivery_timing_fields(
    config: ExemplarDeliveryConfig,
    *,
    num_examples: int,
    num_selected_payloads: int,
    selected_example_count: int,
    delivery_wall_seconds: float,
    selection_wall_seconds: float,
    payload_wall_seconds: float,
    pruning_wall_seconds: float,
) -> dict[str, Any]:
    path_key = (
        "path_b_wall_seconds"
        if config.delivery_path == TWO_PASS_RERUN_SELECTED
        else "path_a_wall_seconds"
    )
    teacher_rerun_wall_seconds = (
        payload_wall_seconds if config.delivery_path == TWO_PASS_RERUN_SELECTED else 0.0
    )
    return {
        "timing_enabled": True,
        "delivery_wall_seconds": delivery_wall_seconds,
        path_key: delivery_wall_seconds,
        "selection_wall_seconds": selection_wall_seconds,
        "selected_payload_materialization_wall_seconds": payload_wall_seconds,
        "pruning_wall_seconds": pruning_wall_seconds,
        "teacher_rerun_wall_seconds": teacher_rerun_wall_seconds,
        "teacher_rerun_examples_per_second": _rate(
            selected_example_count,
            teacher_rerun_wall_seconds,
        ),
        "examples_per_second": _rate(num_examples, delivery_wall_seconds),
        "selected_payloads_per_second": _rate(
            num_selected_payloads,
            payload_wall_seconds,
        ),
        "timing_claims_not_made": {
            "no_speed_parity_requirement": True,
            "no_performance_regression_gate": True,
            "timing_is_environment_specific": True,
        },
    }


def _parity_timing_fields(
    path_a_report: dict[str, Any],
    path_b_report: dict[str, Any],
) -> dict[str, Any]:
    if not (path_a_report.get("timing_enabled") or path_b_report.get("timing_enabled")):
        return {}
    path_a_wall = _float_or_none(
        path_a_report.get("path_a_wall_seconds")
        or path_a_report.get("delivery_wall_seconds")
    )
    path_b_wall = _float_or_none(
        path_b_report.get("path_b_wall_seconds")
        or path_b_report.get("delivery_wall_seconds")
    )
    ratio = (
        path_b_wall / path_a_wall
        if path_a_wall is not None and path_b_wall is not None and path_a_wall > 0
        else None
    )
    return {
        "timing_enabled": True,
        "path_a_wall_seconds": path_a_wall,
        "path_b_wall_seconds": path_b_wall,
        "path_b_over_path_a_wall_ratio": ratio,
        "faster_path": _faster_path(path_a_wall, path_b_wall),
        "path_a_examples_per_second": path_a_report.get("examples_per_second"),
        "path_b_examples_per_second": path_b_report.get("examples_per_second"),
        "path_a_selected_payloads_per_second": path_a_report.get(
            "selected_payloads_per_second"
        ),
        "path_b_selected_payloads_per_second": path_b_report.get(
            "selected_payloads_per_second"
        ),
        "timing_claims_not_made": {
            "no_speed_parity_requirement": True,
            "no_performance_regression_gate": True,
            "timing_is_environment_specific": True,
        },
    }


def _faster_path(
    path_a_wall: float | None,
    path_b_wall: float | None,
) -> str:
    if path_a_wall is None or path_b_wall is None:
        return "unknown"
    if path_a_wall == path_b_wall:
        return "tie"
    return "path_a" if path_a_wall < path_b_wall else "path_b"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _selected_board_summary(
    selected_payloads: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_board = {
        board_id: [
            item for item in selected_payloads if item.get("selected_board") == board_id
        ]
        for board_id in (
            PRIMARY_SELECTED_BOARD,
            LONG_TAIL_UNCERTAINTY_BOARD,
            PERVERSE_TAIL_DIAGNOSTIC_BOARD,
        )
    }
    semantic_counts: dict[str, int] = {}
    long_tail_counts: dict[str, int] = {}
    source_score_board_counts: dict[str, int] = {}
    for item in selected_payloads:
        tag = str(item.get("semantic_tail_tag") or "unknown_open_class_tail")
        semantic_counts[tag] = semantic_counts.get(tag, 0) + 1
        long_tail_class = str(item.get("long_tail_class") or "normal")
        long_tail_counts[long_tail_class] = long_tail_counts.get(long_tail_class, 0) + 1
    for record in selected_records:
        board = str(record.get("mode_key") or "unassigned")
        source_score_board_counts[board] = source_score_board_counts.get(board, 0) + 1
    return {
        "primary_count": len(by_board[PRIMARY_SELECTED_BOARD]),
        "long_tail_uncertainty_count": len(by_board[LONG_TAIL_UNCERTAINTY_BOARD]),
        "perverse_tail_diagnostic_count": len(by_board[PERVERSE_TAIL_DIAGNOSTIC_BOARD]),
        "total_selected_count": len(selected_payloads),
        "semantic_tail_class_counts": dict(sorted(semantic_counts.items())),
        "long_tail_class_counts": dict(sorted(long_tail_counts.items())),
        "source_score_board_counts": dict(sorted(source_score_board_counts.items())),
    }


def _long_tail_observations(summary: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    for class_name, key in (
        ("long_tail", "long_tail_count"),
        ("very_long_tail", "very_long_tail_count"),
        ("suspicious_flat", "suspicious_flat_count"),
        (
            "full_vocab_or_near_full_vocab",
            "full_vocab_or_near_full_vocab_count",
        ),
    ):
        count = int(summary.get(key) or 0)
        if count:
            observations.append(f"selected exemplars classified {class_name}: {count}")
    return observations


def _leaderboard_report(
    manifest: dict[str, Any],
    *,
    selected_records: list[dict[str, Any]],
    config: ExemplarDeliveryConfig,
    created_at: str,
    long_tail_summary: dict[str, Any],
    selected_board_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "selected_exemplar_leaderboard_report_v1",
        "created_at": created_at,
        "delivery_path": config.delivery_path,
        "selection_policy": manifest.get("selection_policy"),
        "score_policy": config.score_policy,
        "leaderboard_capacity": config.leaderboard_capacity,
        "selected_exemplar_budget": config.selected_exemplar_budget,
        "selected_exemplar_fraction": config.selected_exemplar_fraction,
        "num_candidates_seen": manifest.get("num_candidates_seen"),
        "num_board_winners": manifest.get("num_board_winners"),
        "num_selected_exemplars": len(selected_records),
        "long_tail_summary": long_tail_summary,
        "selected_board_summary": selected_board_summary,
        "primary_selected_exemplar_budget": _primary_budget(config),
        "long_tail_side_board_cap": config.long_tail_side_board_cap,
        "perverse_tail_side_board_cap": config.perverse_tail_side_board_cap,
        "include_long_tail_in_primary": config.include_long_tail_in_primary,
        "include_perverse_tail_in_primary": config.include_perverse_tail_in_primary,
        "reject_perverse_exemplars": config.reject_perverse_exemplars,
        "boards": manifest.get("boards", []),
    }


def _elapsed(started_at: float) -> float:
    return max(0.0, perf_counter() - started_at)


def _rate(count: int, wall_seconds: float) -> float | None:
    if wall_seconds <= 0:
        return None
    return float(count) / wall_seconds


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [name for name in globals() if not name.startswith("__")]
