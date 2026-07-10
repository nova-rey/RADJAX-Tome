from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from radjax_tome.backends import (
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
)
from radjax_tome.builder.corridor_artifacts import (
    build_corridor_artifacts,
    validate_corridor_artifacts,
)
from radjax_tome.builder.exemplar_selection import (
    PATH_A_FULFILLMENT_POLICY,
    PATH_B_FULFILLMENT_POLICY,
    build_exemplar_selection_manifest,
)
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.targets.store import TeacherTargetStore

EXEMPLAR_DELIVERY_REPORT_FILENAME = "delivery_report.json"
EXEMPLAR_DELIVERY_REPORT_SCHEMA = "selected_exemplar_delivery_report_v1"
EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA = "exemplar_delivery_parity_report_v1"
LEADERBOARD_REPORT_FILENAME = "leaderboard_report.json"
SELECTED_EXEMPLARS_FILENAME = "selected_exemplars.json"
SELECTED_LINKAGE_MISMATCH = (
    "selected exemplar linkage mismatch: selected record/payload does not match "
    "source candidate coordinate"
)

ONE_PASS_PRUNED_CANDIDATE = "one_pass_pruned_candidate"
TWO_PASS_RERUN_SELECTED = "two_pass_rerun_selected"
EXEMPLAR_SCORE_POLICY = "entropy_top_n_v1"
DeliveryProgressCallback = Callable[[dict[str, Any]], None]

_REQUIRED_SELECTED_PAYLOAD_FIELDS = (
    "selected_example_id",
    "selected_position",
    "selected_score",
    "score_selected_position_entropy",
    "score_top_token_id",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_score",
    "source_top_token_id",
    "source_score_policy",
    "selected_policy",
    "source_delivery_path",
    "top_token_ids",
    "top_log_probs",
    "top_probs",
    "top_selection_mask",
    "effective_top_k",
    "top_mass",
    "tail_mass",
    "bucket_masses",
    "teacher_entropy",
    "sequence_length",
    "vocab_size",
    "num_buckets",
    "dynamic_top_k",
    "corridor_mode_id",
    "corridor_fingerprint_id",
    "corridor_assignment_status",
)

_ONE_PASS_CANDIDATE_PAYLOAD_ARRAYS = (
    "exemplar_source_policy_ids",
    "exemplar_source_top_token_ids",
    "exemplar_source_top_log_probs",
    "exemplar_source_top_probs",
    "exemplar_source_top_selection_mask",
    "exemplar_source_effective_top_k",
    "exemplar_source_top_mass",
    "exemplar_source_tail_mass",
    "exemplar_source_bucket_masses",
)


@dataclass(frozen=True)
class ExemplarDeliveryConfig:
    artifact_dir: Path
    dataset_path: Path
    delivery_path: str = TWO_PASS_RERUN_SELECTED
    selection_enabled: bool = False
    leaderboard_capacity: int = 16
    selected_exemplar_budget: int | None = None
    selected_exemplar_fraction: float | None = None
    retain_unselected_exemplar_payloads: bool = True
    score_policy: str = EXEMPLAR_SCORE_POLICY
    sequence_length: int = 16
    vocab_size: int = 32
    top_k: int = 8
    num_buckets: int = 4
    max_examples: int | None = None
    backend_config: TeacherBackendConfig | None = None
    selected_rerun_batch_size: int = 1
    track_timing: bool = False
    progress_callback: DeliveryProgressCallback | None = None


def materialize_selected_exemplar_delivery(
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    _validate_delivery_config(config)
    created_at = _now()
    delivery_started = perf_counter()
    store = TeacherTargetStore.open(config.artifact_dir)
    examples = _load_examples(
        config.dataset_path,
        max_examples=store.metadata.num_examples,
    )
    fulfillment_policy = (
        PATH_B_FULFILLMENT_POLICY
        if config.delivery_path == TWO_PASS_RERUN_SELECTED
        else PATH_A_FULFILLMENT_POLICY
    )
    selection_started = perf_counter()
    manifest = build_exemplar_selection_manifest(
        store,
        examples=examples,
        batch_size=_batch_size_from_store(store),
        capture_mode=_capture_mode_for_delivery(config.delivery_path),
        fulfillment_policy=fulfillment_policy,
        board_capacity=config.leaderboard_capacity,
        budget_examples=config.selected_exemplar_budget,
        budget_fraction=config.selected_exemplar_fraction,
        created_at=created_at,
        canonical_score_fields_only=True,
        use_score_pass_fields=config.delivery_path == TWO_PASS_RERUN_SELECTED,
    )
    selected_records = _flatten_selected_records(
        manifest,
        delivery_path=config.delivery_path,
    )
    selected_example_count = len(
        {record["selected_example_id"] for record in selected_records}
    )
    selection_wall_seconds = _elapsed(selection_started)
    payload_started = perf_counter()
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        _notify_delivery_progress(
            config,
            phase="selected_rerun",
            event="started",
            selected_examples_processed=0,
            selected_examples_total=selected_example_count,
        )
    selected_payloads = _selected_payloads(
        selected_records,
        store=store,
        examples=examples,
        config=config,
    )
    payload_wall_seconds = _elapsed(payload_started)
    output = config.artifact_dir
    corridors_dir = output / "corridors"
    leaderboards_dir = output / "leaderboards"
    selected_dir = output / "selected_exemplars"
    leaderboards_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    corridor_result = build_corridor_artifacts(
        output_dir=output,
        examples=examples,
        selected_records=selected_records,
        selected_payloads=selected_payloads,
        delivery_path=config.delivery_path,
        non_selected_exemplar_payload_retained=(
            config.retain_unselected_exemplar_payloads
        ),
        progress_callback=config.progress_callback,
    )
    pruning_started = perf_counter()
    pruned_candidate_payload_bytes = _prune_path_a_candidate_payload_arrays(
        store,
        retain=config.retain_unselected_exemplar_payloads,
        enabled=config.delivery_path == ONE_PASS_PRUNED_CANDIDATE,
    )
    pruning_wall_seconds = _elapsed(pruning_started)
    temporary_candidate_bytes = _materialize_path_a_temp_cache(
        output,
        selected_payloads=selected_payloads,
        retain=config.retain_unselected_exemplar_payloads,
        enabled=config.delivery_path == ONE_PASS_PRUNED_CANDIDATE,
    )
    leaderboard_report = _leaderboard_report(
        manifest,
        selected_records=selected_records,
        config=config,
        created_at=created_at,
    )
    selected_exemplars = {
        "schema_version": "selected_exemplars_v1",
        "created_at": created_at,
        "delivery_path": config.delivery_path,
        "score_policy": config.score_policy,
        "selected_exemplars": selected_records,
    }
    write_json(leaderboards_dir / LEADERBOARD_REPORT_FILENAME, leaderboard_report)
    write_json(leaderboards_dir / SELECTED_EXEMPLARS_FILENAME, selected_exemplars)
    write_json(
        selected_dir / "selected-exemplars-00000.json",
        {
            "schema_version": "selected_exemplar_payload_shard_v1",
            "delivery_path": config.delivery_path,
            "selected_exemplars": selected_payloads,
        },
    )

    retained_bytes = _tree_bytes(corridors_dir) + _tree_bytes(leaderboards_dir)
    retained_bytes += _tree_bytes(selected_dir)
    report = {
        "schema_version": EXEMPLAR_DELIVERY_REPORT_SCHEMA,
        "status": "pass",
        "blockers": [],
        "warnings": [],
        "created_at": created_at,
        "completed_at": _now(),
        "selection_enabled": config.selection_enabled,
        "delivery_path": config.delivery_path,
        "dataset_path": str(config.dataset_path),
        "score_policy": config.score_policy,
        "num_examples_scored": store.metadata.num_examples,
        "num_positions_scored": store.metadata.num_examples
        * store.metadata.sequence_length,
        "num_selected_exemplars": len(selected_payloads),
        "selected_example_count": selected_example_count,
        "selected_rerun_example_ids": (
            _unique_selected_example_ids(selected_records)
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else []
        ),
        "selected_exemplar_payload_retained": bool(selected_payloads),
        "non_selected_exemplar_payload_retained": (
            config.retain_unselected_exemplar_payloads
        ),
        "teacher_rerun_count": (
            selected_example_count
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else 0
        ),
        "selected_rerun_batch_size": (
            config.selected_rerun_batch_size
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else None
        ),
        "selected_payload_source": (
            "backend_dynamic_cascaded_soft_labels_v1"
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else "one_pass_candidate_shard_capture"
        ),
        "temporary_candidate_bytes": temporary_candidate_bytes,
        "pruned_candidate_payload_bytes": pruned_candidate_payload_bytes,
        "final_retained_bytes": retained_bytes,
        "leaderboard_report_path": str(leaderboards_dir / LEADERBOARD_REPORT_FILENAME),
        "selected_exemplars_path": str(leaderboards_dir / SELECTED_EXEMPLARS_FILENAME),
        "selected_payload_shard_count": 1,
        "claims_not_made": {
            "no_dense_logits_retained": True,
            "no_student_training_quality_claim": True,
            "no_path_b_quality_parity_without_report": True,
        },
    }
    report.update(corridor_result.report_fields())
    if config.retain_unselected_exemplar_payloads:
        report["status"] = "fail"
        report["blockers"] = ["non-selected exemplar payload retention is enabled"]
    if config.track_timing:
        delivery_wall_seconds = _elapsed(delivery_started)
        report.update(
            _delivery_timing_fields(
                config,
                num_examples=store.metadata.num_examples,
                num_selected_payloads=len(selected_payloads),
                selected_example_count=selected_example_count,
                delivery_wall_seconds=delivery_wall_seconds,
                selection_wall_seconds=selection_wall_seconds,
                payload_wall_seconds=payload_wall_seconds,
                pruning_wall_seconds=pruning_wall_seconds,
            )
        )
    write_json(output / EXEMPLAR_DELIVERY_REPORT_FILENAME, report)
    return report


def validate_selected_exemplar_delivery(
    artifact_dir: Path,
) -> tuple[list[str], list[str]]:
    report_path = artifact_dir / EXEMPLAR_DELIVERY_REPORT_FILENAME
    selected_dir = artifact_dir / "selected_exemplars"
    leaderboards_dir = artifact_dir / "leaderboards"
    if not report_path.exists() and not selected_dir.exists():
        return [], []
    blockers: list[str] = []
    warnings: list[str] = []
    if not report_path.is_file():
        return ["delivery_report.json missing"], warnings
    try:
        report = read_json_object(report_path)
    except (OSError, ValueError) as exc:
        return [f"delivery_report.json invalid: {exc}"], warnings
    if report.get("non_selected_exemplar_payload_retained") is True:
        blockers.append("non_selected_exemplar_payload_retained=true")
    if report.get("selection_enabled") is True:
        if int(report.get("num_selected_exemplars") or 0) <= 0:
            blockers.append("selected exemplar count is zero")
    selected_path = leaderboards_dir / SELECTED_EXEMPLARS_FILENAME
    selected = _read_selected_exemplars(selected_path, blockers)
    payloads = _read_selected_payloads(selected_dir, blockers)
    expected_selected_count = int(report.get("num_selected_exemplars") or 0)
    if selected and len(selected) != expected_selected_count:
        blockers.append("selected_exemplars.json count does not match delivery report")
    if payloads and len(payloads) != expected_selected_count:
        blockers.append("selected payload count does not match delivery report")
    _validate_selected_record_payload_linkage(
        artifact_dir,
        selected_records=selected,
        selected_payloads=payloads,
        blockers=blockers,
    )
    sequence_length = _metadata_int(artifact_dir, "sequence_length")
    selected_ids = {str(item.get("selected_example_id")) for item in selected}
    payload_ids = {str(item.get("selected_example_id")) for item in payloads}
    _validate_selected_ids_against_dataset(report, selected_ids, blockers)
    missing_payload_ids = selected_ids.difference(payload_ids)
    if missing_payload_ids:
        blockers.append(
            "selected_exemplars.json references selections without payloads: "
            + ", ".join(sorted(missing_payload_ids))
        )
    for item in selected:
        position = int(item.get("selected_position", -1))
        if sequence_length is not None and not 0 <= position < sequence_length:
            blockers.append(f"selected position outside sequence length: {position}")
    for payload in payloads:
        missing = [
            field for field in _REQUIRED_SELECTED_PAYLOAD_FIELDS if field not in payload
        ]
        if missing:
            blockers.append(
                "selected exemplar payload missing compressed teacher target fields: "
                + ", ".join(missing)
            )
    if report.get("delivery_path") == TWO_PASS_RERUN_SELECTED:
        if report.get("teacher_rerun_count") != report.get("selected_example_count"):
            blockers.append(
                "Path B teacher_rerun_count does not match selected examples"
            )
    for name in ("unselected_candidate_payloads", "temporary_candidates"):
        if (artifact_dir / name).exists():
            blockers.append(
                "final artifact includes temporary unselected candidate payloads: "
                f"{name}"
            )
    if report.get("non_selected_exemplar_payload_retained") is False:
        retained_arrays = _retained_one_pass_candidate_payload_arrays(artifact_dir)
        if retained_arrays:
            blockers.append(
                "final artifact retains unselected one-pass candidate payload arrays: "
                + ", ".join(retained_arrays)
            )
    corridor_validation = validate_corridor_artifacts(
        artifact_dir,
        selected_records=selected,
        selected_payloads=payloads,
        expected_selected_count=expected_selected_count,
    )
    blockers.extend(corridor_validation.blockers)
    warnings.extend(corridor_validation.warnings)
    return blockers, warnings


def _validate_selected_ids_against_dataset(
    report: dict[str, Any],
    selected_ids: set[str],
    blockers: list[str],
) -> None:
    dataset_path_value = report.get("dataset_path")
    if not dataset_path_value:
        return
    dataset_path = Path(str(dataset_path_value))
    if not dataset_path.is_file():
        blockers.append("delivery_report.json dataset_path is missing")
        return
    try:
        dataset_ids = {
            example.example_id
            for example in _load_examples(
                dataset_path,
                max_examples=max(len(selected_ids), 1_000_000_000),
            )
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(f"delivery_report.json dataset_path invalid: {exc}")
        return
    missing = selected_ids.difference(dataset_ids)
    if missing:
        blockers.append(
            "selected_exemplars.json references examples not present in dataset: "
            + ", ".join(sorted(missing))
        )


def _validate_selected_record_payload_linkage(
    artifact_dir: Path,
    *,
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    if not selected_records or not selected_payloads:
        return
    if len(selected_records) != len(selected_payloads):
        return
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    try:
        store = TeacherTargetStore.open(artifact_dir)
    except (OSError, ValueError):
        blockers.append(SELECTED_LINKAGE_MISMATCH)
        return
    for record, payload in zip(selected_records, selected_payloads, strict=True):
        if _record_payload_tuple_mismatch(record, payload):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        try:
            source_shard_id = int(record["source_shard_id"])
            source_row = int(record["source_row"])
        except (KeyError, TypeError, ValueError):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        try:
            shard = shard_cache.setdefault(
                source_shard_id,
                store.read_shard(source_shard_id),
            )
        except (OSError, ValueError, KeyError):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        if _source_coordinate_linkage_mismatch(
            shard,
            row=source_row,
            record=record,
            payload=payload,
        ):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return


def _record_payload_tuple_mismatch(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    fields = (
        "selected_example_id",
        "selected_position",
        "source_shard_id",
        "source_row",
        "source_position",
        "source_top_token_id",
        "source_score_policy",
        "corridor_mode_id",
        "corridor_assignment_status",
    )
    if any(record.get(field) != payload.get(field) for field in fields):
        return True
    return not (
        _close_float(record.get("selected_score"), payload.get("selected_score"))
        and _close_float(record.get("source_score"), payload.get("source_score"))
    )


def _source_coordinate_linkage_mismatch(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    try:
        selected_position = int(record["selected_position"])
        source_position = int(record["source_position"])
        source_score = float(record["source_score"])
        source_top_token_id = int(record["source_top_token_id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return True
    if selected_position != source_position:
        return True
    if not _close_float(record.get("selected_score"), source_score):
        return True
    if not _close_float(payload.get("selected_score"), source_score):
        return True
    if not _close_float(payload.get("source_score"), source_score):
        return True
    if not _close_float(payload.get("teacher_entropy"), source_score):
        return True
    if int(payload.get("source_top_token_id", -1)) != source_top_token_id:
        return True
    if not _path_b_score_pass_aliases_match(record, payload, shard, row=row):
        return True
    top_token_ids = payload.get("top_token_ids")
    if not isinstance(top_token_ids, list) or not top_token_ids:
        return True
    if int(top_token_ids[0]) != source_top_token_id:
        return True
    entropy_key = (
        "corridor_entropy"
        if "corridor_entropy" in shard
        else "corridor_teacher_entropy"
    )
    try:
        corridor_entropy = float(np.asarray(shard[entropy_key])[row, source_position])
        corridor_top_token_id = int(
            np.asarray(shard["corridor_top_token_ids"])[row, source_position]
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return True
    return not (
        _close_float(corridor_entropy, source_score)
        and corridor_top_token_id == source_top_token_id
    )


def _path_b_score_pass_aliases_match(
    record: dict[str, Any],
    payload: dict[str, Any],
    shard: dict[str, np.ndarray],
    *,
    row: int,
) -> bool:
    payload_ref = record.get("payload_ref", {})
    if not isinstance(payload_ref, dict):
        payload_ref = {}
    if payload_ref.get("kind") != "corridor_exemplar_score_pass_v1":
        return True
    try:
        shard_position = int(np.asarray(shard["score_selected_position"])[row])
        shard_score = float(np.asarray(shard["score_selected_position_entropy"])[row])
        shard_top_token_id = int(np.asarray(shard["score_top_token_id"])[row])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return (
        int(record.get("source_position", -1)) == shard_position
        and _close_float(record.get("source_score"), shard_score)
        and int(record.get("source_top_token_id", -1)) == shard_top_token_id
        and _close_float(record.get("score_selected_position_entropy"), shard_score)
        and int(record.get("score_top_token_id", -1)) == shard_top_token_id
        and _close_float(payload.get("score_selected_position_entropy"), shard_score)
        and int(payload.get("score_top_token_id", -1)) == shard_top_token_id
    )


def _close_float(left: Any, right: Any, *, atol: float = 1e-4) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=atol))
    except (TypeError, ValueError):
        return False


def compare_exemplar_delivery_artifacts(
    path_a: Path,
    path_b: Path,
    *,
    output: Path | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    left = _artifact_selection(path_a)
    right = _artifact_selection(path_b)
    blockers: list[str] = []
    warnings: list[str] = []
    if left["ids"] != right["ids"]:
        blockers.append("selected example IDs differ")
    if left["positions"] != right["positions"]:
        blockers.append("selected positions differ")
    if left["ranks"] != right["ranks"]:
        blockers.append("selected score ranks differ")
    for index, (left_score, right_score) in enumerate(
        zip(left["scores"], right["scores"], strict=False)
    ):
        if abs(left_score - right_score) > atol:
            blockers.append(f"selected score differs at rank {index + 1}")
            break
    if left["mode_keys"] != right["mode_keys"]:
        blockers.append("selected mode keys differ")
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
        "selected_example_ids_match": left["ids"] == right["ids"],
        "selected_positions_match": left["positions"] == right["positions"],
        "selected_score_ranks_match": left["ranks"] == right["ranks"],
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


def _validate_delivery_config(config: ExemplarDeliveryConfig) -> None:
    if not config.selection_enabled:
        raise ValueError("selected exemplar delivery requires selection_enabled=True")
    if config.delivery_path not in {ONE_PASS_PRUNED_CANDIDATE, TWO_PASS_RERUN_SELECTED}:
        raise ValueError("unsupported exemplar delivery path")
    if config.score_policy != EXEMPLAR_SCORE_POLICY:
        raise ValueError("exemplar_score_policy must be 'entropy_top_n_v1'")
    if config.leaderboard_capacity < 1:
        raise ValueError("exemplar_leaderboard_capacity must be positive")
    if (
        config.selected_exemplar_budget is not None
        and config.selected_exemplar_budget < 1
    ):
        raise ValueError("selected_exemplar_budget must be positive")
    if config.selected_exemplar_fraction is not None and not (
        0.0 < config.selected_exemplar_fraction <= 1.0
    ):
        raise ValueError("selected_exemplar_fraction must be in (0, 1]")


def _load_examples(path: Path, *, max_examples: int) -> tuple[TinyTextExample, ...]:
    examples: list[TinyTextExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(examples) >= max_examples:
                break
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"dataset line {line_number} must be an object")
            text = str(payload.get("text", ""))
            example_id = str(payload.get("example_id") or f"row-{line_number:06d}")
            examples.append(TinyTextExample(example_id=example_id, text=text))
    return tuple(examples)


def _batch_size_from_store(store: TeacherTargetStore) -> int:
    if store.metadata.shard_count <= 1:
        return max(1, store.metadata.num_examples)
    first = store.read_shard(0)
    return int(first["input_ids"].shape[0])


def _capture_mode_for_delivery(delivery_path: str) -> str:
    if delivery_path == TWO_PASS_RERUN_SELECTED:
        return "two_pass_sparse_exemplar"
    return "one_pass_candidate"


def _flatten_selected_records(
    manifest: dict[str, Any],
    *,
    delivery_path: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in manifest.get("selected_examples", ()):
        if not isinstance(example, dict):
            continue
        for position_record in example.get("selected_position_records", ()):
            if not isinstance(position_record, dict):
                continue
            scores = position_record.get("scores_by_board", {})
            if not isinstance(scores, dict):
                scores = {}
            assigned_board = str(position_record.get("assigned_board", "entropy"))
            selected_score = float(position_record["selected_score"])
            score_entropy = float(position_record["score_selected_position_entropy"])
            if not np.isclose(selected_score, score_entropy, rtol=1e-5, atol=1e-5):
                raise ValueError(SELECTED_LINKAGE_MISMATCH)
            score_top_token_id = position_record.get("score_top_token_id")
            payload_ref = position_record.get("payload_ref", {})
            if not isinstance(payload_ref, dict):
                payload_ref = {}
            source_shard_id = int(
                position_record.get(
                    "source_shard_id",
                    payload_ref.get("source_shard_id", -1),
                )
            )
            source_row = int(
                position_record.get("source_row", payload_ref.get("source_row", -1))
            )
            source_position = int(position_record.get("source_position", -1))
            source_score = float(position_record.get("source_score", selected_score))
            source_top_token_id = position_record.get("source_top_token_id")
            if source_top_token_id is None:
                source_top_token_id = score_top_token_id
            source_score_policy = str(
                position_record.get("source_score_policy", EXEMPLAR_SCORE_POLICY)
            )
            if source_position < 0:
                source_position = int(position_record.get("selected_position", 0))
            records.append(
                {
                    "rank": len(records) + 1,
                    "selected_example_id": str(example.get("example_id")),
                    "selected_position": source_position,
                    "selected_score": source_score,
                    "score_selected_position_entropy": source_score,
                    "score_top_token_id": (
                        None if score_top_token_id is None else int(score_top_token_id)
                    ),
                    "source_shard_id": source_shard_id,
                    "source_row": source_row,
                    "source_position": source_position,
                    "source_score": source_score,
                    "source_top_token_id": (
                        None
                        if source_top_token_id is None
                        else int(source_top_token_id)
                    ),
                    "source_score_policy": source_score_policy,
                    "selected_policy": EXEMPLAR_SCORE_POLICY,
                    "source_delivery_path": delivery_path,
                    "mode_key": assigned_board,
                    "payload_ref": payload_ref,
                    "rank_by_board": position_record.get("rank_by_board", {}),
                    "scores_by_board": scores,
                }
            )
    records.sort(
        key=lambda item: (
            -float(item["selected_score"]),
            str(item["selected_example_id"]),
            int(item["selected_position"]),
        )
    )
    for rank, item in enumerate(records, start=1):
        item["rank"] = rank
    return records


def _selected_payloads(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        return _selected_payloads_from_backend(
            selected_records,
            examples=examples,
            config=config,
        )
    return _selected_payloads_from_one_pass_capture(
        selected_records,
        store=store,
        config=config,
    )


def _selected_payloads_from_one_pass_capture(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    for record in selected_records:
        payload_ref = record.get("payload_ref", {})
        if not isinstance(payload_ref, dict):
            raise ValueError("selected record missing one-pass payload_ref")
        source_shard_id = int(payload_ref.get("source_shard_id", -1))
        source_row = int(payload_ref.get("source_row", -1))
        if source_shard_id < 0 or source_row < 0:
            raise ValueError("selected record has invalid one-pass payload_ref")
        shard = shard_cache.setdefault(
            source_shard_id, store.read_shard(source_shard_id)
        )
        payloads.append(
            _selected_payload_from_one_pass_shard(
                record,
                shard=shard,
                row=source_row,
                config=config,
            )
        )
    return payloads


def _selected_payload_from_one_pass_shard(
    record: dict[str, Any],
    *,
    shard: dict[str, np.ndarray],
    row: int,
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    position = int(record["source_position"])
    payload_position = _one_pass_payload_position_index(
        shard,
        row=row,
        source_position=position,
        payload_ref=record.get("payload_ref", {}),
    )
    top_selection_mask = _payload_slice(
        shard,
        "exemplar_source_top_selection_mask",
        row,
        payload_position,
    )
    effective_top_k = int(
        _payload_scalar(shard, "exemplar_source_effective_top_k", row, payload_position)
    )
    return {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["source_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "top_token_ids": _payload_slice(
            shard,
            "exemplar_source_top_token_ids",
            row,
            payload_position,
        ),
        "top_log_probs": _payload_slice(
            shard,
            "exemplar_source_top_log_probs",
            row,
            payload_position,
        ),
        "top_probs": _payload_slice(
            shard,
            "exemplar_source_top_probs",
            row,
            payload_position,
        ),
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": _payload_scalar(
            shard,
            "exemplar_source_top_mass",
            row,
            payload_position,
        ),
        "tail_mass": _payload_scalar(
            shard,
            "exemplar_source_tail_mass",
            row,
            payload_position,
        ),
        "bucket_masses": _payload_slice(
            shard,
            "exemplar_source_bucket_masses",
            row,
            payload_position,
        ),
        "teacher_entropy": _payload_scalar(
            shard,
            "corridor_teacher_entropy",
            row,
            position,
        ),
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "num_buckets": config.num_buckets,
        "dynamic_top_k": _dynamic_top_k_metadata(
            config,
            effective_top_k=effective_top_k,
            source_payload="one_pass_candidate_shard",
        ),
    }


def _one_pass_payload_position_index(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    source_position: int,
    payload_ref: object,
) -> int:
    positions = np.asarray(shard.get("exemplar_positions", ()))
    candidate_rank = None
    if isinstance(payload_ref, dict):
        raw_rank = payload_ref.get("candidate_rank", payload_ref.get("position_index"))
        if raw_rank is not None:
            candidate_rank = int(raw_rank)
    if positions.ndim == 2 and candidate_rank is not None:
        if int(positions[row, candidate_rank]) != source_position:
            raise ValueError(SELECTED_LINKAGE_MISMATCH)
    source = np.asarray(shard["exemplar_source_top_token_ids"])
    if source.ndim >= 3:
        sequence_length = int(np.asarray(shard["corridor_teacher_entropy"]).shape[1])
        if int(source.shape[1]) == sequence_length:
            return source_position
    if candidate_rank is None:
        raise ValueError(SELECTED_LINKAGE_MISMATCH)
    return candidate_rank


def _selected_payloads_from_backend(
    selected_records: list[dict[str, Any]],
    *,
    examples: tuple[TinyTextExample, ...],
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    if not selected_records:
        return []
    if config.backend_config is None:
        raise ValueError("selected exemplar delivery requires backend_config")
    selected_example_ids = _unique_selected_example_ids(selected_records)
    examples_by_id = {example.example_id: example for example in examples}
    missing = [
        example_id
        for example_id in selected_example_ids
        if example_id not in examples_by_id
    ]
    if missing:
        raise ValueError(
            "selected examples are missing from dataset: " + ", ".join(missing)
        )
    selected_examples = tuple(
        examples_by_id[example_id] for example_id in selected_example_ids
    )
    batch_size = max(1, config.selected_rerun_batch_size)
    backend_config = replace(
        config.backend_config,
        target_policy="dynamic_cascaded_soft_labels_v1",  # type: ignore[arg-type]
        exemplar_source_policy="dynamic_cascaded_soft_labels_v1",
        batch_size=batch_size,
    )
    backend = create_backend(backend_config)
    payloads_by_record: dict[int, dict[str, Any]] = {}
    try:
        for start in range(0, len(selected_examples), batch_size):
            chunk = selected_examples[start : start + batch_size]
            result = backend.emit_batch(
                TeacherBatchInput(
                    example_ids=tuple(example.example_id for example in chunk),
                    texts=tuple(example.text for example in chunk),
                )
            )
            row_by_example_id = {
                example.example_id: row for row, example in enumerate(chunk)
            }
            for record_index, record in enumerate(selected_records):
                example_id = str(record["selected_example_id"])
                if example_id not in row_by_example_id:
                    continue
                payloads_by_record[record_index] = _selected_payload_from_emission(
                    record,
                    payload=result.payload,
                    row=row_by_example_id[example_id],
                    config=config,
                )
            _notify_delivery_progress(
                config,
                phase="selected_rerun",
                event="progress",
                selected_examples_processed=start + len(chunk),
                selected_examples_total=len(selected_examples),
            )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return [payloads_by_record[index] for index in range(len(selected_records))]


def _notify_delivery_progress(
    config: ExemplarDeliveryConfig,
    **payload: Any,
) -> None:
    if config.progress_callback is not None:
        config.progress_callback(payload)


def _prune_path_a_candidate_payload_arrays(
    store: TeacherTargetStore,
    *,
    retain: bool,
    enabled: bool,
) -> int:
    if not enabled or retain:
        return 0
    removed_bytes = 0
    for shard_id in range(store.metadata.shard_count):
        path = store.root / "shards" / f"shard-{shard_id:05d}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        remove_keys = [
            key for key in _ONE_PASS_CANDIDATE_PAYLOAD_ARRAYS if key in arrays
        ]
        if not remove_keys:
            continue
        removed_bytes += sum(int(arrays[key].nbytes) for key in remove_keys)
        pruned = {key: value for key, value in arrays.items() if key not in remove_keys}
        np.savez(path, **pruned)
    return removed_bytes


def _retained_one_pass_candidate_payload_arrays(artifact_dir: Path) -> list[str]:
    retained: set[str] = set()
    shards_dir = artifact_dir / "shards"
    if not shards_dir.is_dir():
        return []
    for path in sorted(shards_dir.glob("shard-*.npz")):
        with np.load(path, allow_pickle=False) as loaded:
            retained.update(
                key for key in loaded.files if key.startswith("exemplar_source_")
            )
    return sorted(retained)


def _selected_payload_from_emission(
    record: dict[str, Any],
    *,
    payload: Any,
    row: int,
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    position = int(record["source_position"])
    top_selection_mask = _payload_slice(payload, "top_selection_mask", row, position)
    effective_top_k = int(_payload_scalar(payload, "effective_top_k", row, position))
    top_token_ids = _payload_slice(payload, "top_token_ids", row, position)
    if top_token_ids and int(top_token_ids[0]) != int(record["source_top_token_id"]):
        raise ValueError(SELECTED_LINKAGE_MISMATCH)
    return {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["source_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "top_token_ids": top_token_ids,
        "top_log_probs": _payload_slice(payload, "top_log_probs", row, position),
        "top_probs": _payload_slice(payload, "top_probs", row, position),
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": _payload_scalar(payload, "top_mass", row, position),
        "tail_mass": _payload_scalar(payload, "tail_mass", row, position),
        "bucket_masses": _payload_slice(payload, "bucket_masses", row, position),
        "teacher_entropy": _payload_scalar(payload, "teacher_entropy", row, position),
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "num_buckets": config.num_buckets,
        "dynamic_top_k": _dynamic_top_k_metadata(
            config,
            effective_top_k=effective_top_k,
            source_payload="backend_emit_batch",
        ),
    }


def _dynamic_top_k_metadata(
    config: ExemplarDeliveryConfig,
    *,
    effective_top_k: int,
    source_payload: str,
) -> dict[str, Any]:
    return {
        "policy": config.backend_config.dynamic_top_k_policy
        if config.backend_config is not None
        else "mass_threshold_v1",
        "requested_top_k": config.top_k,
        "effective_top_k": effective_top_k,
        "score_policy": config.score_policy,
        "source_payload": source_payload,
    }


def _payload_slice(payload: Any, key: str, row: int, position: int) -> list[Any]:
    if key not in payload:
        raise ValueError(f"selected backend payload missing {key}")
    value = np.asarray(payload[key])[row, position]
    if value.dtype == np.bool_:
        return [bool(item) for item in value.tolist()]
    if np.issubdtype(value.dtype, np.integer):
        return [int(item) for item in value.tolist()]
    return [float(item) for item in value.tolist()]


def _payload_scalar(payload: Any, key: str, row: int, position: int) -> int | float:
    if key not in payload:
        raise ValueError(f"selected backend payload missing {key}")
    value = np.asarray(payload[key])[row, position].item()
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _unique_selected_example_ids(selected_records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for record in sorted(selected_records, key=_selected_record_source_key):
        example_id = str(record["selected_example_id"])
        if example_id in seen:
            continue
        seen.add(example_id)
        ids.append(example_id)
    return ids


def _selected_record_source_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(record.get("source_shard_id", 999_999_999)),
        int(record.get("source_row", 999_999_999)),
        str(record.get("selected_example_id", "")),
    )


def _materialize_path_a_temp_cache(
    output: Path,
    *,
    selected_payloads: list[dict[str, Any]],
    retain: bool,
    enabled: bool,
) -> int:
    if not enabled:
        return 0
    temp_dir = output / "temporary_candidates"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "candidate-cache.json"
    write_json(
        temp_path,
        {
            "schema_version": "temporary_candidate_cache_v1",
            "candidate_count": len(selected_payloads),
            "selected_candidate_preview": selected_payloads[:1],
        },
    )
    byte_count = _tree_bytes(temp_dir)
    if retain:
        retained = output / "unselected_candidate_payloads"
        if retained.exists():
            shutil.rmtree(retained)
        temp_dir.rename(retained)
    else:
        shutil.rmtree(temp_dir)
    return byte_count


def _leaderboard_report(
    manifest: dict[str, Any],
    *,
    selected_records: list[dict[str, Any]],
    config: ExemplarDeliveryConfig,
    created_at: str,
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
        "boards": manifest.get("boards", []),
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
    for path in sorted(selected_dir.glob("*.json")):
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


def _metadata_int(artifact_dir: Path, key: str) -> int | None:
    try:
        metadata = read_json_object(artifact_dir / "metadata.json")
    except (OSError, ValueError):
        return None
    value = metadata.get(key)
    return int(value) if value is not None else None


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
    }


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


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _elapsed(started_at: float) -> float:
    return max(0.0, perf_counter() - started_at)


def _rate(count: int, wall_seconds: float) -> float | None:
    if wall_seconds <= 0:
        return None
    return float(count) / wall_seconds


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
