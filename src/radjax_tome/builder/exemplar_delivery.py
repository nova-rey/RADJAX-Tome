from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
from collections.abc import Callable, Mapping
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
    CorridorArtifactBuildResult,
    build_corridor_artifacts,
    validate_corridor_artifacts,
)
from radjax_tome.builder.exemplar_selection import (
    PATH_A_FULFILLMENT_POLICY,
    PATH_B_FULFILLMENT_POLICY,
    build_exemplar_selection_manifest,
)
from radjax_tome.builder.long_tail import (
    DEFAULT_LONG_TAIL_WARNING_K,
    DEFAULT_PERVERSE_TAIL_WARNING_K,
    DEFAULT_VERY_LONG_TAIL_WARNING_K,
    LONG_TAIL_UNCERTAINTY_BOARD,
    PERVERSE_TAIL_DIAGNOSTIC_BOARD,
    PRIMARY_SELECTED_BOARD,
    LongTailPolicy,
    is_perverse_long_tail,
    long_tail_diagnostics,
    long_tail_summary,
    selected_board_for_long_tail,
    semantic_tail_tag,
)
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.quantization import (
    ENTROPY_PARITY_QUANTIZATION_STEP,
    entropy_absolute_delta,
    entropy_parity_close,
)
from radjax_tome.targets.store import TeacherTargetStore

EXEMPLAR_DELIVERY_REPORT_FILENAME = "delivery_report.json"
EXEMPLAR_DELIVERY_REPORT_SCHEMA = "selected_exemplar_delivery_report_v1"
EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA = "exemplar_delivery_parity_report_v1"
LEADERBOARD_REPORT_FILENAME = "leaderboard_report.json"
SELECTED_EXEMPLARS_FILENAME = "selected_exemplars.json"
CURRICULUM_ROUTES_FILENAME = "selected_routes.json"
CURRICULUM_ROUTES_SCHEMA = "selected_exemplar_curriculum_routes_v1"
_SIDE_SELECTED_BOARD_IDS = (
    LONG_TAIL_UNCERTAINTY_BOARD,
    PERVERSE_TAIL_DIAGNOSTIC_BOARD,
)
SELECTED_LINKAGE_MISMATCH = (
    "selected exemplar linkage mismatch: selected record/payload does not match "
    "source candidate coordinate"
)

ONE_PASS_PRUNED_CANDIDATE = "one_pass_pruned_candidate"
TWO_PASS_RERUN_SELECTED = "two_pass_rerun_selected"
NATIVE_C6_PATH_B_EXECUTION = "native_c6_path_b_v1"
EXEMPLAR_SCORE_POLICY = "entropy_top_n_v1"
DeliveryProgressCallback = Callable[[dict[str, Any]], None]


class SelectedExemplarDeliveryError(ValueError):
    """Preserves a machine-readable coordinate trace for delivery failures."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{SELECTED_LINKAGE_MISMATCH}: {json.dumps(diagnostic, sort_keys=True)}"
        )


class SelectedRerunCudaOOMError(RuntimeError):
    """Native selected rerun exhausted the final microbatch size."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            "selected rerun CUDA OOM at batch size 1: "
            + json.dumps(diagnostic, sort_keys=True)
        )


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
    "payload_ref",
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
    "dynamic_mass_threshold",
    "dynamic_top_k_max",
    "top_k_saturated",
    "long_tail_class",
    "long_tail_warnings",
    "effective_top_k_fraction_of_vocab",
    "semantic_tail_tag",
    "selected_board",
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
    long_tail_warning_k: int = DEFAULT_LONG_TAIL_WARNING_K
    very_long_tail_warning_k: int = DEFAULT_VERY_LONG_TAIL_WARNING_K
    perverse_tail_warning_k: int = DEFAULT_PERVERSE_TAIL_WARNING_K
    reject_perverse_exemplars: bool = False
    primary_selected_exemplar_budget: int | None = None
    long_tail_side_board_cap: int = 128
    perverse_tail_side_board_cap: int = 32
    include_long_tail_in_primary: bool = False
    include_perverse_tail_in_primary: bool = False
    include_perverse_tail_in_student: bool = False
    progress_callback: DeliveryProgressCallback | None = None
    authoritative_selection: bool = False
    authoritative_records: tuple[dict[str, Any], ...] | None = None
    execution_mode: str = "legacy_delivery_v1"
    rerun_metrics: dict[str, Any] | None = None
    delivery_authority_hash: str | None = None


@dataclass(frozen=True)
class PreparedSelectedDelivery:
    """In-memory handoff between the native selected-delivery phases."""

    config: ExemplarDeliveryConfig
    created_at: str
    delivery_started: float
    store: TeacherTargetStore
    examples: tuple[TinyTextExample, ...]
    manifest: dict[str, Any]
    selected_records: list[dict[str, Any]]
    selected_payloads: list[dict[str, Any]]
    rerun_selected_example_count: int
    rerun_selected_example_ids: list[str]
    selection_wall_seconds: float
    payload_wall_seconds: float
    selected_example_count: int
    tail_summary: dict[str, Any]
    selected_board_summary: dict[str, Any]
    selected_records_by_board: dict[str, list[dict[str, Any]]]
    corridors_dir: Path
    leaderboards_dir: Path
    selected_dir: Path
    curriculum_dir: Path
    corridor_result: CorridorArtifactBuildResult | None = None


def materialize_selected_exemplar_delivery(
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    """Materialize selected delivery through its canonical three-phase order."""

    prepared = run_selected_delivery_rerun(config)
    finalized = finalize_selected_delivery_corridor(prepared)
    return assemble_selected_delivery_artifacts(finalized)


def run_selected_delivery_rerun(
    config: ExemplarDeliveryConfig,
) -> PreparedSelectedDelivery:
    """Select and materialize rerun payloads before final corridor export."""

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
    long_tail_policy = _long_tail_policy(config)
    selection_started = perf_counter()
    candidate_filter = None
    if config.reject_perverse_exemplars:

        def candidate_filter(candidate: Any) -> bool:
            return not _candidate_is_perverse(candidate, config=config)

    if config.authoritative_records is not None:
        manifest = {
            "selection_policy": "radjax.multi_role_selected_exemplar.v1",
            "fulfillment_policy": fulfillment_policy,
            "num_candidates_seen": len(config.authoritative_records),
            "num_board_winners": len(config.authoritative_records),
            "boards": [],
        }
        selected_records = [dict(item) for item in config.authoritative_records]
    else:
        manifest = build_exemplar_selection_manifest(
            store,
            examples=examples,
            batch_size=_batch_size_from_store(store),
            capture_mode=_capture_mode_for_delivery(config.delivery_path),
            fulfillment_policy=fulfillment_policy,
            board_capacity=config.leaderboard_capacity,
            created_at=created_at,
            budget_examples=None,
            budget_fraction=None,
            canonical_score_fields_only=True,
            use_score_pass_fields=config.delivery_path == TWO_PASS_RERUN_SELECTED,
            candidate_filter=candidate_filter,
            candidate_filter_name=(
                "reject_perverse_dynamic_top_k"
                if config.reject_perverse_exemplars
                else None
            ),
        )
        selected_records = _route_records_for_delivery(
            _flatten_selected_records(manifest, delivery_path=config.delivery_path),
            config=config,
        )
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        _validate_path_b_score_pass_records(
            selected_records,
            store=store,
            require_score_pass_tuple=not config.authoritative_selection,
        )
    rerun_selected_example_count = len(
        {record["selected_example_id"] for record in selected_records}
    )
    rerun_selected_example_ids = _unique_selected_example_ids(selected_records)
    selection_wall_seconds = _elapsed(selection_started)
    payload_started = perf_counter()
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        _notify_delivery_progress(
            config,
            phase="selected_rerun",
            event="started",
            selected_examples_processed=0,
            selected_examples_total=rerun_selected_example_count,
            selected_coordinates_committed=0,
            selected_coordinates_total=len(selected_records),
        )
    output = config.artifact_dir
    corridors_dir = output / "corridors"
    leaderboards_dir = output / "leaderboards"
    selected_dir = output / "selected_exemplars"
    curriculum_dir = output / "curriculum"
    leaderboards_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    staged_payload_summaries: dict[int, dict[str, Any]] = {}
    if _native_streamed_payloads(config):
        staged_payload_summaries = _prepare_native_payload_staging(
            config,
            selected_records=selected_records,
        )
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    selected_payloads = _selected_payloads(
        selected_records,
        store=store,
        examples=examples,
        config=config,
        completed_record_indices=set(staged_payload_summaries),
        existing_payload_summaries=staged_payload_summaries,
    )
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        _notify_delivery_progress(
            config,
            phase="selected_rerun",
            event="complete",
            selected_examples_processed=rerun_selected_example_count,
            selected_examples_total=rerun_selected_example_count,
            selected_coordinates_committed=len(selected_payloads),
            selected_coordinates_total=len(selected_records),
        )
    if not _native_streamed_payloads(config):
        _attach_long_tail_diagnostics(
            selected_records,
            selected_payloads,
            config=config,
            policy=long_tail_policy,
        )
    selected_records, selected_payloads = _route_materialized_selected_exemplars(
        selected_records,
        selected_payloads,
        config=config,
    )
    selected_example_count = len(
        {record["selected_example_id"] for record in selected_records}
    )
    tail_summary = long_tail_summary(selected_payloads)
    selected_board_summary = _selected_board_summary(
        selected_payloads,
        selected_records,
    )
    selected_records_by_board = _records_by_selected_board(selected_records)
    payload_wall_seconds = _elapsed(payload_started)
    return PreparedSelectedDelivery(
        config=config,
        created_at=created_at,
        delivery_started=delivery_started,
        store=store,
        examples=examples,
        manifest=manifest,
        selected_records=selected_records,
        selected_payloads=selected_payloads,
        rerun_selected_example_count=rerun_selected_example_count,
        rerun_selected_example_ids=rerun_selected_example_ids,
        selection_wall_seconds=selection_wall_seconds,
        payload_wall_seconds=payload_wall_seconds,
        selected_example_count=selected_example_count,
        tail_summary=tail_summary,
        selected_board_summary=selected_board_summary,
        selected_records_by_board=selected_records_by_board,
        corridors_dir=corridors_dir,
        leaderboards_dir=leaderboards_dir,
        selected_dir=selected_dir,
        curriculum_dir=curriculum_dir,
    )


def finalize_selected_delivery_corridor(
    prepared: PreparedSelectedDelivery,
) -> PreparedSelectedDelivery:
    """Write the final selected-linked public corridor surface."""

    config = prepared.config
    corridor_result = build_corridor_artifacts(
        output_dir=config.artifact_dir,
        examples=prepared.examples,
        selected_records=prepared.selected_records,
        selected_payloads=prepared.selected_payloads,
        delivery_path=config.delivery_path,
        non_selected_exemplar_payload_retained=(
            config.retain_unselected_exemplar_payloads
        ),
        progress_callback=config.progress_callback,
    )
    return replace(prepared, corridor_result=corridor_result)


def assemble_selected_delivery_artifacts(
    prepared: PreparedSelectedDelivery,
) -> dict[str, Any]:
    """Promote native payloads and write the legacy delivery artifact surface."""

    if prepared.corridor_result is None:
        raise ValueError(
            "selected delivery artifact assembly requires finalized corridor artifacts"
        )
    config = prepared.config
    output = config.artifact_dir
    store = prepared.store
    selected_records = prepared.selected_records
    selected_payloads = prepared.selected_payloads
    corridors_dir = prepared.corridors_dir
    leaderboards_dir = prepared.leaderboards_dir
    selected_dir = prepared.selected_dir
    curriculum_dir = prepared.curriculum_dir
    corridor_result = prepared.corridor_result
    if _native_streamed_payloads(config):
        native_payload_hashes = _synchronize_native_payload_shards(
            _native_payload_stage_dir(config),
            selected_records=selected_records,
        )
        _promote_native_payload_shards(config)
        for summary in selected_payloads:
            record_index = int(summary.get("_record_index", -1))
            if record_index in native_payload_hashes:
                summary["payload_hash"] = native_payload_hashes[record_index]
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
        prepared.manifest,
        selected_records=selected_records,
        config=config,
        created_at=prepared.created_at,
        long_tail_summary=prepared.tail_summary,
        selected_board_summary=prepared.selected_board_summary,
    )
    selected_exemplars = {
        "schema_version": "selected_exemplars_v1",
        "created_at": prepared.created_at,
        "delivery_path": config.delivery_path,
        "score_policy": config.score_policy,
        "selected_exemplars": selected_records,
        "long_tail_summary": prepared.tail_summary,
        "selected_board_summary": prepared.selected_board_summary,
        "selected_exemplar_boards": prepared.selected_records_by_board,
    }
    write_json(leaderboards_dir / LEADERBOARD_REPORT_FILENAME, leaderboard_report)
    write_json(leaderboards_dir / SELECTED_EXEMPLARS_FILENAME, selected_exemplars)
    if _native_streamed_payloads(config):
        _write_json_atomic(
            selected_dir / "payload_index.json",
            {
                "schema_version": "selected_exemplar_payload_index_v1",
                "delivery_path": config.delivery_path,
                "storage_kind": "one_record_json_shards_v1",
                "selected_exemplars": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "_record_index"
                    }
                    for item in selected_payloads
                ],
            },
        )
    curriculum_summary = _write_curriculum_routes(
        curriculum_dir / CURRICULUM_ROUTES_FILENAME,
        selected_records,
    )
    for board_id in _SIDE_SELECTED_BOARD_IDS:
        write_json(
            leaderboards_dir / f"{board_id}.json",
            {
                "schema_version": "selected_exemplar_side_board_v1",
                "delivery_path": config.delivery_path,
                "selected_board": board_id,
                "selected_exemplars": prepared.selected_records_by_board[board_id],
                "selected_board_summary": prepared.selected_board_summary,
            },
        )
    if not _native_streamed_payloads(config):
        write_json(
            selected_dir / "selected-exemplars-00000.json",
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "delivery_path": config.delivery_path,
                "long_tail_summary": prepared.tail_summary,
                "selected_board_summary": prepared.selected_board_summary,
                "selected_exemplars": selected_payloads,
            },
        )

    retained_bytes = _tree_bytes(corridors_dir) + _tree_bytes(leaderboards_dir)
    retained_bytes += _tree_bytes(selected_dir)
    rerun_metrics = dict(config.rerun_metrics or {})
    report = {
        "schema_version": EXEMPLAR_DELIVERY_REPORT_SCHEMA,
        "status": "pass",
        "blockers": [],
        "warnings": [],
        "long_tail_observations": _long_tail_observations(prepared.tail_summary),
        "created_at": prepared.created_at,
        "completed_at": _now(),
        "selection_enabled": config.selection_enabled,
        "delivery_path": config.delivery_path,
        "execution_mode": config.execution_mode,
        "delivery_authority_hash": config.delivery_authority_hash,
        "dataset_path": str(config.dataset_path),
        "score_policy": config.score_policy,
        "entropy_quantization_step": ENTROPY_PARITY_QUANTIZATION_STEP,
        "entropy_parity_tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
        "num_examples_scored": store.metadata.num_examples,
        "num_positions_scored": store.metadata.num_examples
        * store.metadata.sequence_length,
        "num_selected_exemplars": len(selected_payloads),
        "selected_board_summary": prepared.selected_board_summary,
        "primary_selected_exemplar_budget": _primary_budget(config),
        "long_tail_side_board_cap": config.long_tail_side_board_cap,
        "perverse_tail_side_board_cap": config.perverse_tail_side_board_cap,
        "include_long_tail_in_primary": config.include_long_tail_in_primary,
        "include_perverse_tail_in_primary": config.include_perverse_tail_in_primary,
        "include_perverse_tail_in_student": config.include_perverse_tail_in_student,
        "long_tail_summary": prepared.tail_summary,
        "selected_example_count": prepared.selected_example_count,
        "selected_rerun_example_ids": (
            prepared.rerun_selected_example_ids
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else []
        ),
        "selected_rerun_example_count": (
            prepared.rerun_selected_example_count
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else 0
        ),
        "selected_exemplar_payload_retained": bool(selected_payloads),
        "non_selected_exemplar_payload_retained": (
            config.retain_unselected_exemplar_payloads
        ),
        "teacher_rerun_count": (
            prepared.rerun_selected_example_count
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else 0
        ),
        "selected_rerun_batch_size": (
            rerun_metrics.get(
                "selected_rerun_batch_size", config.selected_rerun_batch_size
            )
            if config.delivery_path == TWO_PASS_RERUN_SELECTED
            else None
        ),
        "selected_rerun_batch_count": rerun_metrics.get(
            "selected_rerun_batch_count", 0
        ),
        "selected_rerun_examples": rerun_metrics.get(
            "selected_rerun_examples", prepared.rerun_selected_example_count
        ),
        "selected_rerun_examples_per_second": rerun_metrics.get(
            "selected_rerun_examples_per_second"
        ),
        "selected_rerun_teacher_seconds": rerun_metrics.get(
            "selected_rerun_teacher_seconds"
        ),
        "selected_rerun_compression_seconds": rerun_metrics.get(
            "selected_rerun_compression_seconds"
        ),
        "selected_rerun_io_seconds": rerun_metrics.get("selected_rerun_io_seconds"),
        "selected_rerun_peak_host_memory_bytes": rerun_metrics.get(
            "selected_rerun_peak_host_memory_bytes"
        ),
        "selected_rerun_peak_device_memory_bytes": rerun_metrics.get(
            "selected_rerun_peak_device_memory_bytes"
        ),
        "selected_rerun_requested_batch_size": rerun_metrics.get(
            "selected_rerun_requested_batch_size"
        ),
        "selected_rerun_effective_batch_sizes": rerun_metrics.get(
            "selected_rerun_effective_batch_sizes"
        ),
        "selected_source_example_count": rerun_metrics.get(
            "selected_source_example_count"
        ),
        "selected_coordinate_count": rerun_metrics.get(
            "selected_coordinate_count", len(selected_records)
        ),
        "requested_source_batch_size": rerun_metrics.get("requested_source_batch_size"),
        "effective_source_batch_sizes": rerun_metrics.get(
            "effective_source_batch_sizes"
        ),
        "source_batch_count": rerun_metrics.get("source_batch_count"),
        "coordinate_compression_batch_count": rerun_metrics.get(
            "coordinate_compression_batch_count"
        ),
        "selected_row_gather_seconds": rerun_metrics.get("selected_row_gather_seconds"),
        "payload_write_seconds": rerun_metrics.get("payload_write_seconds"),
        "staging_directory": rerun_metrics.get("staging_directory"),
        "staging_preserved": rerun_metrics.get("staging_preserved", False),
        "staging_payload_count": rerun_metrics.get("staging_payload_count", 0),
        "staging_quarantined_count": rerun_metrics.get("staging_quarantined_count", 0),
        "staging_quarantine_directory": rerun_metrics.get(
            "staging_quarantine_directory"
        ),
        "cuda_oom_retry_count": rerun_metrics.get("cuda_oom_retry_count", 0),
        "cuda_oom_retry_batch_transitions": rerun_metrics.get(
            "cuda_oom_retry_batch_transitions", []
        ),
        "cuda_oom_failure_stage_counts": rerun_metrics.get(
            "cuda_oom_failure_stage_counts", {}
        ),
        "coordinates_committed_before_each_retry": rerun_metrics.get(
            "coordinates_committed_before_each_retry", []
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
        "curriculum_routes_path": str(curriculum_dir / CURRICULUM_ROUTES_FILENAME),
        "curriculum_route_count": curriculum_summary["route_count"],
        "curriculum_unique_coordinate_count": curriculum_summary[
            "unique_coordinate_count"
        ],
        "selected_payload_shard_count": (
            int((config.rerun_metrics or {}).get("selected_payload_shard_count", 1))
        ),
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
        delivery_wall_seconds = _elapsed(prepared.delivery_started)
        report.update(
            _delivery_timing_fields(
                config,
                num_examples=store.metadata.num_examples,
                num_selected_payloads=len(selected_payloads),
                selected_example_count=prepared.selected_example_count,
                delivery_wall_seconds=delivery_wall_seconds,
                selection_wall_seconds=prepared.selection_wall_seconds,
                payload_wall_seconds=prepared.payload_wall_seconds,
                pruning_wall_seconds=pruning_wall_seconds,
            )
        )
    _write_json_atomic(output / EXEMPLAR_DELIVERY_REPORT_FILENAME, report)
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
    payloads = _read_selected_payload_summaries(selected_dir, blockers)
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
        if report.get("teacher_rerun_count") != report.get(
            "selected_rerun_example_count",
            report.get("selected_example_count"),
        ):
            blockers.append(
                "Path B teacher_rerun_count does not match selected rerun examples"
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
    if record.get("source_delivery_path") == ONE_PASS_PRUNED_CANDIDATE and (
        not isinstance(record.get("payload_ref"), dict)
        or not record.get("payload_ref")
        or not isinstance(payload.get("payload_ref"), dict)
        or not payload.get("payload_ref")
    ):
        return True
    fields = [
        "selected_example_id",
        "selected_position",
        "score_top_token_id",
        "source_shard_id",
        "source_row",
        "source_position",
        "source_top_token_id",
        "source_score_policy",
        "payload_ref",
    ]
    if not bool(record.get("c5_authoritative_coordinate")):
        fields.extend(("corridor_mode_id", "corridor_assignment_status"))
    if any(record.get(field) != payload.get(field) for field in fields):
        return True
    if "selected_board" in record and record.get("selected_board") != payload.get(
        "selected_board"
    ):
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
    if not _entropy_parity_close(payload.get("teacher_entropy"), source_score):
        return True
    if int(payload.get("source_top_token_id", -1)) != source_top_token_id:
        return True
    top_token_ids = payload.get("top_token_ids")
    if not isinstance(top_token_ids, list) or not top_token_ids:
        return True
    if int(top_token_ids[0]) != source_top_token_id:
        return True
    source_delivery_path = record.get("source_delivery_path")
    if source_delivery_path == ONE_PASS_PRUNED_CANDIDATE:
        if _path_a_source_payload_token_mismatch(shard, row=row, record=record):
            return True
    elif source_delivery_path == TWO_PASS_RERUN_SELECTED:
        if not bool(
            record.get("c5_authoritative_coordinate")
        ) and not _path_b_score_pass_aliases_match(
            record,
            payload,
            shard,
            row=row,
        ):
            return True
    else:
        return True
    entropy_key = (
        "corridor_entropy"
        if "corridor_entropy" in shard
        else "corridor_teacher_entropy"
    )
    try:
        corridor_entropy = float(np.asarray(shard[entropy_key])[row, source_position])
    except (IndexError, KeyError, TypeError, ValueError):
        return True
    if not _close_float(corridor_entropy, source_score):
        return True
    if source_delivery_path == TWO_PASS_RERUN_SELECTED and not bool(
        record.get("c5_authoritative_coordinate")
    ):
        try:
            corridor_top_token_id = int(
                np.asarray(shard["corridor_top_token_ids"])[row, source_position]
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return True
        return corridor_top_token_id != source_top_token_id
    return False


def _path_a_source_payload_token_mismatch(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
) -> bool:
    if "exemplar_source_top_token_ids" not in shard:
        # Candidate arrays are deliberately pruned after payload materialization.
        return False
    try:
        payload_position = _one_pass_payload_position_index(
            shard,
            row=row,
            record=record,
        )
        source_top_token_id = _source_top_token_at(
            np.asarray(shard["exemplar_source_top_token_ids"]),
            row=row,
            position=payload_position,
        )
        return source_top_token_id != int(record["source_top_token_id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return True


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
        shard_score = float(np.asarray(shard["score_selected_position_entropy"])[row])
        shard_top_token_id = int(np.asarray(shard["score_top_token_id"])[row])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return (
        _path_b_score_pass_record_matches(record, shard, row=row)
        and _close_float(payload.get("score_selected_position_entropy"), shard_score)
        and int(payload.get("score_top_token_id", -1)) == shard_top_token_id
    )


def _path_b_score_pass_record_matches(
    record: dict[str, Any],
    shard: dict[str, np.ndarray],
    *,
    row: int,
) -> bool:
    payload_ref = record.get("payload_ref", {})
    if not isinstance(payload_ref, dict):
        return False
    try:
        shard_position = int(np.asarray(shard["score_selected_position"])[row])
        shard_score = float(np.asarray(shard["score_selected_position_entropy"])[row])
        shard_top_token_id = int(np.asarray(shard["score_top_token_id"])[row])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return (
        payload_ref.get("kind") == "corridor_exemplar_score_pass_v1"
        and int(record.get("selected_position", -1)) == shard_position
        and int(record.get("source_position", -1)) == shard_position
        and _close_float(record.get("selected_score"), shard_score)
        and _close_float(record.get("source_score"), shard_score)
        and int(record.get("source_top_token_id", -1)) == shard_top_token_id
        and _close_float(record.get("score_selected_position_entropy"), shard_score)
        and int(record.get("score_top_token_id", -1)) == shard_top_token_id
        and _int_or_none(payload_ref.get("source_shard_id"))
        == _int_or_none(record.get("source_shard_id"))
        and _int_or_none(payload_ref.get("source_row"))
        == _int_or_none(record.get("source_row"))
        and _int_or_none(payload_ref.get("source_position")) == shard_position
        and _close_float(payload_ref.get("source_score"), shard_score)
        and _int_or_none(payload_ref.get("source_top_token_id")) == shard_top_token_id
    )


def _validate_path_b_score_pass_records(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    require_score_pass_tuple: bool = True,
) -> None:
    selected_record_order = [
        str(record.get("selected_example_id", "")) for record in selected_records
    ]
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    for record in selected_records:
        try:
            source_shard_id = int(record["source_shard_id"])
            source_row = int(record["source_row"])
            shard = shard_cache.setdefault(
                source_shard_id,
                store.read_shard(source_shard_id),
            )
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            raise _path_b_delivery_error(
                record,
                store=store,
                failure_reason=(
                    f"selected record cannot be resolved to a score-pass shard: {exc}"
                ),
                selected_record_order=selected_record_order,
            ) from exc
        if require_score_pass_tuple and not _path_b_score_pass_record_matches(
            record,
            shard,
            row=source_row,
        ):
            raise _path_b_delivery_error(
                record,
                store=store,
                failure_reason=(
                    "selected record does not match its score-pass shard tuple"
                ),
                selected_record_order=selected_record_order,
            )


def _path_b_rerun_payload_mismatch(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    mismatch_fields: list[str] = []
    top_token_ids = payload.get("top_token_ids")
    if not isinstance(top_token_ids, list) or not top_token_ids:
        mismatch_fields.append("top_token_ids")
    elif int(top_token_ids[0]) != int(record["source_top_token_id"]):
        mismatch_fields.append("top_token_ids[0]")
    if not _entropy_parity_close(
        payload.get("teacher_entropy"), record.get("source_score")
    ):
        mismatch_fields.append("teacher_entropy")
    if _record_payload_tuple_mismatch(record, payload):
        mismatch_fields.append("record_payload_tuple")
    return mismatch_fields


def _path_b_delivery_error(
    record: dict[str, Any],
    *,
    store: TeacherTargetStore,
    failure_reason: str,
    selected_record_order: list[str],
    rerun_input_order: list[str] | None = None,
    rerun_row_index: int | None = None,
    rerun_payload: dict[str, Any] | None = None,
    mismatch_fields: list[str] | None = None,
) -> SelectedExemplarDeliveryError:
    source_shard_id = _int_or_none(record.get("source_shard_id"))
    source_row = _int_or_none(record.get("source_row"))
    source_position = _int_or_none(record.get("source_position"))
    shard: dict[str, np.ndarray] | None = None
    if source_shard_id is not None:
        try:
            shard = store.read_shard(source_shard_id)
        except (OSError, ValueError, KeyError):
            shard = None
    evidence_row = _resolve_score_pass_evidence_row(shard, record)
    score_fields = _path_b_shard_diagnostic_fields(
        shard,
        row=evidence_row if evidence_row is not None else source_row,
        position=source_position,
    )
    record_matches_score_pass = (
        shard is not None
        and source_row is not None
        and evidence_row == source_row
        and _path_b_score_pass_record_matches(record, shard, row=source_row)
    )
    diagnostic = {
        "failure_stage": "selected_exemplar_delivery",
        "delivery_path": TWO_PASS_RERUN_SELECTED,
        "failure_reason": failure_reason,
        "selected_example_id": record.get("selected_example_id"),
        "rank": record.get("rank"),
        "source_shard_id": source_shard_id,
        "source_row": source_row,
        "source_position": source_position,
        "source_score": record.get("source_score"),
        "source_top_token_id": record.get("source_top_token_id"),
        "payload_ref": record.get("payload_ref"),
        "selected_record": record,
        "record_matches_score_pass_tuple": record_matches_score_pass,
        "score_pass_evidence_row": evidence_row,
        "score_pass_evidence_coordinate_match": evidence_row == source_row,
        "selected_record_order": selected_record_order,
        "rerun_input_order": rerun_input_order,
        "rerun_row_index": rerun_row_index,
        "rerun_payload_top_token_id": _first_payload_token_id(rerun_payload or {}),
        "rerun_payload_teacher_entropy": (
            None if rerun_payload is None else rerun_payload.get("teacher_entropy")
        ),
        "entropy_absolute_delta": _entropy_absolute_delta(
            None if rerun_payload is None else rerun_payload.get("teacher_entropy"),
            record.get("source_score"),
        ),
        "entropy_allowed_tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
        "entropy_parity_status": (
            "pass"
            if rerun_payload is not None
            and _entropy_parity_close(
                rerun_payload.get("teacher_entropy"), record.get("source_score")
            )
            else "fail"
        ),
        "mismatch_fields": mismatch_fields or [],
        **score_fields,
    }
    return SelectedExemplarDeliveryError(diagnostic)


def _path_b_shard_diagnostic_fields(
    shard: dict[str, np.ndarray] | None,
    *,
    row: int | None,
    position: int | None,
) -> dict[str, Any]:
    if shard is None or row is None or position is None:
        return {
            "score_selected_position": None,
            "score_selected_position_entropy": None,
            "score_top_token_id": None,
            "corridor_entropy_at_source_position": None,
            "corridor_top_token_id_at_source_position": None,
        }
    return {
        "score_selected_position": _array_scalar_or_none(
            shard,
            "score_selected_position",
            row,
        ),
        "score_selected_position_entropy": _array_scalar_or_none(
            shard,
            "score_selected_position_entropy",
            row,
        ),
        "score_top_token_id": _array_scalar_or_none(
            shard,
            "score_top_token_id",
            row,
        ),
        "corridor_entropy_at_source_position": _array_scalar_or_none(
            shard,
            "corridor_entropy"
            if "corridor_entropy" in shard
            else "corridor_teacher_entropy",
            row,
            position,
        ),
        "corridor_top_token_id_at_source_position": _array_scalar_or_none(
            shard,
            "corridor_top_token_ids",
            row,
            position,
        ),
    }


def _resolve_score_pass_evidence_row(
    shard: dict[str, np.ndarray] | None,
    record: Mapping[str, Any],
) -> int | None:
    """Resolve score evidence by exact ID/position, then verify the passport row."""

    if shard is None:
        return None
    example_id = str(record.get("selected_example_id", ""))
    source_position = _int_or_none(record.get("source_position"))
    source_row = _int_or_none(record.get("source_row"))
    if source_position is None:
        return None
    try:
        example_ids = np.asarray(shard["score_example_ids"]).reshape(-1)
        positions = np.asarray(shard["score_selected_position"]).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return source_row
    if np.issubdtype(example_ids.dtype, np.integer):
        matches = [
            index
            for index, candidate_position in enumerate(positions.tolist())
            if index == source_row and int(candidate_position) == source_position
        ]
    else:
        matches = [
            index
            for index, (candidate_id, candidate_position) in enumerate(
                zip(example_ids.tolist(), positions.tolist(), strict=False)
            )
            if _score_pass_example_id(candidate_id) == example_id
            and int(candidate_position) == source_position
        ]
    if len(matches) != 1:
        return source_row
    return matches[0]


def _score_pass_example_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _array_scalar_or_none(
    shard: dict[str, np.ndarray],
    key: str,
    row: int,
    position: int | None = None,
) -> int | float | None:
    try:
        value = np.asarray(shard[key])[row]
        if position is not None:
            value = value[position]
        scalar = value.item()
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return float(scalar)


def _close_float(left: Any, right: Any, *, atol: float = 1e-4) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=atol))
    except (TypeError, ValueError):
        return False


def _entropy_absolute_delta(left: Any, right: Any) -> float | None:
    return entropy_absolute_delta(left, right)


def _entropy_parity_close(left: Any, right: Any) -> bool:
    return entropy_parity_close(left, right)


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
    if (
        config.primary_selected_exemplar_budget is not None
        and config.primary_selected_exemplar_budget < 1
    ):
        raise ValueError("primary_selected_exemplar_budget must be positive")
    if config.long_tail_side_board_cap < 1:
        raise ValueError("long_tail_side_board_cap must be positive")
    if config.perverse_tail_side_board_cap < 1:
        raise ValueError("perverse_tail_side_board_cap must be positive")
    if config.execution_mode == NATIVE_C6_PATH_B_EXECUTION:
        if config.delivery_path != TWO_PASS_RERUN_SELECTED:
            raise ValueError("native C6 execution requires two_pass_rerun_selected")
        if not config.authoritative_selection or not config.authoritative_records:
            raise ValueError(
                "native C6 execution requires frozen authoritative C5 records"
            )
    elif config.execution_mode != "legacy_delivery_v1":
        raise ValueError("unsupported selected exemplar delivery execution_mode")
    _long_tail_policy(config)


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
            if delivery_path == ONE_PASS_PRUNED_CANDIDATE and not payload_ref:
                raise ValueError(SELECTED_LINKAGE_MISMATCH)
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
                    "diagnostic_effective_top_k": position_record.get(
                        "diagnostic_effective_top_k",
                        1,
                    ),
                    "diagnostic_top_mass": position_record.get(
                        "diagnostic_top_mass",
                        0.0,
                    ),
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
    completed_record_indices: set[int] | None = None,
    existing_payload_summaries: Mapping[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        return _selected_payloads_from_backend(
            selected_records,
            store=store,
            examples=examples,
            config=config,
            completed_record_indices=completed_record_indices,
            existing_payload_summaries=existing_payload_summaries,
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
        if not isinstance(payload_ref, dict) or not payload_ref:
            raise ValueError("selected record missing one-pass payload_ref")
        source_shard_id = int(record.get("source_shard_id", -1))
        source_row = int(record.get("source_row", -1))
        if source_shard_id < 0 or source_row < 0:
            raise ValueError("selected record has invalid one-pass payload_ref")
        shard = shard_cache.setdefault(
            source_shard_id, store.read_shard(source_shard_id)
        )
        payload_ref_mismatch = _one_pass_payload_ref_mismatch(record, payload_ref)
        if payload_ref_mismatch:
            raise _one_pass_linkage_error(
                record=record,
                shard=shard,
                row=source_row,
                failure_reason=(
                    "one-pass payload reference does not match selected record "
                    "source coordinate"
                ),
                mismatch_fields=payload_ref_mismatch,
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
        record=record,
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
    payload = {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["score_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "payload_ref": record["payload_ref"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "delivery_authority_hash": getattr(config, "delivery_authority_hash", None),
        "selected_board": record.get("selected_board", PRIMARY_SELECTED_BOARD),
        "mode_key": record.get("mode_key"),
        "rank_by_board": record.get("rank_by_board", {}),
        "scores_by_board": record.get("scores_by_board", {}),
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
    mismatch = _path_a_selected_payload_mismatch(payload, record)
    if mismatch:
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason="materialized payload does not match its source coordinate",
            payload_position=payload_position,
            payload_top_token_id=_first_payload_token_id(payload),
            payload_teacher_entropy=payload.get("teacher_entropy"),
            mismatch_fields=mismatch,
        )
    return payload


def _one_pass_payload_position_index(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
) -> int:
    positions = np.asarray(shard.get("exemplar_positions", ()))
    source = np.asarray(shard["exemplar_source_top_token_ids"])
    source_position = int(record["source_position"])
    source_top_token_id = int(record["source_top_token_id"])
    payload_ref = record.get("payload_ref", {})
    candidate_rank = None
    if isinstance(payload_ref, dict):
        raw_rank = payload_ref.get("candidate_rank", payload_ref.get("position_index"))
        if raw_rank is not None:
            try:
                candidate_rank = int(raw_rank)
            except (TypeError, ValueError):
                candidate_rank = None
    storage_kind = _one_pass_payload_storage_kind(shard, source)
    if storage_kind == "full_sequence":
        full_sequence_top_token_id = _source_top_token_at(
            source,
            row=row,
            position=source_position,
        )
        if full_sequence_top_token_id == source_top_token_id:
            return source_position
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason=(
                "full-sequence source payload top token does not match record"
            ),
            candidate_rank=candidate_rank,
            full_sequence_top_token_id=full_sequence_top_token_id,
        )
    rank = _find_matching_candidate_rank(
        source,
        positions=positions,
        row=row,
        source_position=source_position,
        source_top_token_id=source_top_token_id,
    )
    if rank is None:
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason=(
                "no compact candidate payload slot matches source coordinate"
            ),
            candidate_rank=candidate_rank,
        )
    return rank


def _one_pass_payload_storage_kind(
    shard: dict[str, np.ndarray],
    source_top_token_ids: np.ndarray,
) -> str:
    entropy = np.asarray(shard.get("corridor_teacher_entropy", ()))
    if (
        source_top_token_ids.ndim >= 3
        and entropy.ndim >= 2
        and int(source_top_token_ids.shape[1]) == int(entropy.shape[1])
    ):
        return "full_sequence"
    return "compact_candidate_rank"


def _source_top_token_at(
    source_top_token_ids: np.ndarray,
    *,
    row: int,
    position: int,
) -> int | None:
    try:
        return int(source_top_token_ids[row, position, 0])
    except (IndexError, TypeError, ValueError):
        return None


def _candidate_payload_slot_matches(
    source_top_token_ids: np.ndarray,
    *,
    positions: np.ndarray,
    row: int,
    candidate_rank: int,
    source_position: int,
    source_top_token_id: int,
) -> bool:
    try:
        if (
            positions.ndim == 2
            and int(positions[row, candidate_rank]) != source_position
        ):
            return False
        return int(source_top_token_ids[row, candidate_rank, 0]) == source_top_token_id
    except (IndexError, TypeError, ValueError):
        return False


def _find_matching_candidate_rank(
    source_top_token_ids: np.ndarray,
    *,
    positions: np.ndarray,
    row: int,
    source_position: int,
    source_top_token_id: int,
) -> int | None:
    if positions.ndim != 2:
        return None
    for candidate_rank, candidate_position in enumerate(positions[row].tolist()):
        if int(candidate_position) != source_position:
            continue
        if _candidate_payload_slot_matches(
            source_top_token_ids,
            positions=positions,
            row=row,
            candidate_rank=candidate_rank,
            source_position=source_position,
            source_top_token_id=source_top_token_id,
        ):
            return candidate_rank
    return None


def _path_a_selected_payload_mismatch(
    payload: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    top_token_ids = payload.get("top_token_ids")
    mismatch_fields: list[str] = []
    if not isinstance(top_token_ids, list) or not top_token_ids:
        mismatch_fields.append("top_token_ids")
    elif int(top_token_ids[0]) != int(record["source_top_token_id"]):
        mismatch_fields.append("top_token_ids[0]")
    if not _close_float(payload.get("teacher_entropy"), record["source_score"]):
        mismatch_fields.append("teacher_entropy")
    if int(payload.get("selected_position", -1)) != int(record["source_position"]):
        mismatch_fields.append("selected_position")
    if not _close_float(payload.get("selected_score"), record["source_score"]):
        mismatch_fields.append("selected_score")
    return mismatch_fields


def _first_payload_token_id(payload: dict[str, Any]) -> int | None:
    top_token_ids = payload.get("top_token_ids")
    if not isinstance(top_token_ids, list) or not top_token_ids:
        return None
    try:
        return int(top_token_ids[0])
    except (TypeError, ValueError):
        return None


def _one_pass_linkage_error(
    *,
    record: dict[str, Any],
    shard: dict[str, np.ndarray],
    row: int,
    failure_reason: str,
    candidate_rank: int | None = None,
    full_sequence_top_token_id: int | None = None,
    payload_position: int | None = None,
    payload_top_token_id: int | None = None,
    payload_teacher_entropy: Any = None,
    mismatch_fields: list[str] | None = None,
) -> SelectedExemplarDeliveryError:
    positions = np.asarray(shard.get("exemplar_positions", ()))
    source_top_token_ids = np.asarray(shard.get("exemplar_source_top_token_ids", ()))
    storage_kind = _one_pass_payload_storage_kind(shard, source_top_token_ids)
    source_position = _int_or_none(record.get("source_position"))
    source_top_token_id = _int_or_none(record.get("source_top_token_id"))
    payload_ref = record.get("payload_ref")
    searched_ranks = (
        []
        if storage_kind == "full_sequence"
        else _candidate_rank_diagnostics(
            positions=positions,
            source_top_token_ids=source_top_token_ids,
            row=row,
            source_position=source_position,
            source_top_token_id=source_top_token_id,
        )
    )
    diagnostic = {
        "failure_stage": "selected_exemplar_delivery",
        "failure_reason": failure_reason,
        "delivery_path": ONE_PASS_PRUNED_CANDIDATE,
        "selected_example_id": record.get("selected_example_id"),
        "rank": record.get("rank"),
        "source_shard_id": record.get("source_shard_id"),
        "source_row": record.get("source_row"),
        "resolved_source_row": row,
        "source_position": source_position,
        "source_score": record.get("source_score"),
        "source_top_token_id": source_top_token_id,
        "payload_ref": payload_ref,
        "candidate_rank": candidate_rank,
        "exemplar_source_top_token_ids_shape": list(source_top_token_ids.shape),
        "exemplar_positions_shape": list(positions.shape),
        "payload_array_storage_kind": storage_kind,
        "candidate_ranks_searched": searched_ranks,
        "full_sequence_considered": storage_kind == "full_sequence",
        "full_sequence_source_position": source_position,
        "full_sequence_top_token_id": full_sequence_top_token_id,
        "full_sequence_top_match": (
            full_sequence_top_token_id == source_top_token_id
            if full_sequence_top_token_id is not None
            and source_top_token_id is not None
            else None
        ),
        "payload_position": payload_position,
        "payload_top_token_id": payload_top_token_id,
        "payload_teacher_entropy": payload_teacher_entropy,
        "mismatch_fields": mismatch_fields or [],
    }
    return SelectedExemplarDeliveryError(diagnostic)


def _candidate_rank_diagnostics(
    *,
    positions: np.ndarray,
    source_top_token_ids: np.ndarray,
    row: int,
    source_position: int | None,
    source_top_token_id: int | None,
) -> list[dict[str, Any]]:
    if positions.ndim != 2 or not 0 <= row < positions.shape[0]:
        return []
    diagnostics: list[dict[str, Any]] = []
    for candidate_rank, candidate_position in enumerate(positions[row].tolist()):
        candidate_top_token_id = _source_top_token_at(
            source_top_token_ids,
            row=row,
            position=candidate_rank,
        )
        diagnostics.append(
            {
                "candidate_rank": candidate_rank,
                "exemplar_position": int(candidate_position),
                "exemplar_source_top_token_id": candidate_top_token_id,
                "position_match": int(candidate_position) == source_position,
                "top_token_match": candidate_top_token_id == source_top_token_id,
            }
        )
    return diagnostics


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _one_pass_payload_ref_mismatch(
    record: dict[str, Any],
    payload_ref: dict[str, Any],
) -> list[str]:
    mismatch_fields: list[str] = []
    for field in (
        "source_shard_id",
        "source_row",
        "source_position",
        "source_top_token_id",
    ):
        if _int_or_none(record.get(field)) != _int_or_none(payload_ref.get(field)):
            mismatch_fields.append(f"payload_ref.{field}")
    if not _close_float(record.get("source_score"), payload_ref.get("source_score")):
        mismatch_fields.append("payload_ref.source_score")
    return mismatch_fields


def _selected_payloads_from_backend(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
    config: ExemplarDeliveryConfig,
    completed_record_indices: set[int] | None = None,
    existing_payload_summaries: Mapping[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not selected_records:
        return []
    if config.backend_config is None:
        raise ValueError("selected exemplar delivery requires backend_config")
    completed_record_indices = completed_record_indices or set()
    existing_payload_summaries = existing_payload_summaries or {}
    pending_records = [
        (record_index, record)
        for record_index, record in enumerate(selected_records)
        if record_index not in completed_record_indices
    ]
    selected_example_ids = _unique_selected_example_ids(
        [record for _, record in pending_records]
    )
    all_selected_example_ids = _unique_selected_example_ids(selected_records)
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
    selected_record_order = [
        str(record["selected_example_id"]) for record in selected_records
    ]
    batch_size = max(1, config.selected_rerun_batch_size)
    backend_config = replace(
        config.backend_config,
        target_policy="dynamic_cascaded_soft_labels_v1",  # type: ignore[arg-type]
        exemplar_source_policy="dynamic_cascaded_soft_labels_v1",
        batch_size=batch_size,
    )
    backend = create_backend(backend_config)
    payloads_by_record: dict[int, dict[str, Any]] = {
        int(record_index): dict(summary)
        for record_index, summary in existing_payload_summaries.items()
    }
    native_streaming = _native_streamed_payloads(config)
    payload_summaries: list[dict[str, Any]] = [
        dict(existing_payload_summaries[index])
        for index in sorted(existing_payload_summaries)
    ]
    records_by_example_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for record_index, record in pending_records:
        records_by_example_id.setdefault(str(record["selected_example_id"]), []).append(
            (record_index, record)
        )
    positions_by_example_id = {
        example_id: tuple(
            dict.fromkeys(int(record["source_position"]) for _, record in records)
        )
        for example_id, records in records_by_example_id.items()
    }
    selected_row_by_record: dict[int, int] = {}
    selected_row_offset = 0
    for example_id in selected_example_ids:
        for record_index, _ in records_by_example_id[example_id]:
            selected_row_by_record[record_index] = selected_row_offset
            selected_row_offset += 1
    teacher_seconds = 0.0
    compression_seconds = 0.0
    peak_host_memory_bytes = _host_rss_bytes()
    batch_count = 0
    requested_batch_size = batch_size
    effective_batch_sizes: list[int] = []
    cuda_oom_retry_count = 0
    cuda_oom_retry_transitions: list[dict[str, int]] = []
    cuda_oom_failure_stage_counts: dict[str, int] = {}
    coordinates_committed = len(completed_record_indices)
    committed_before_retries: list[int] = []
    start = 0
    try:
        while start < len(selected_examples):
            chunk = selected_examples[start : start + batch_size]
            batch_selected_row_offset = sum(
                len(positions_by_example_id[example_id])
                for example_id in selected_example_ids[:start]
            )
            teacher_started = perf_counter()
            try:
                result = backend.emit_batch(
                    TeacherBatchInput(
                        example_ids=tuple(example.example_id for example in chunk),
                        texts=tuple(example.text for example in chunk),
                        selected_positions_by_example=(
                            tuple(
                                positions_by_example_id[example.example_id]
                                for example in chunk
                            )
                            if native_streaming
                            else None
                        ),
                    )
                )
            except RuntimeError as exc:
                if not _is_recoverable_cuda_oom(exc):
                    raise
                cuda_oom_failure_stage_counts["teacher_or_selected_reduction"] = (
                    cuda_oom_failure_stage_counts.get(
                        "teacher_or_selected_reduction", 0
                    )
                    + 1
                )
                next_batch_size = _next_rerun_batch_size(batch_size)
                if next_batch_size is None:
                    raise SelectedRerunCudaOOMError(
                        {
                            "failure_stage": "selected_rerun",
                            "requested_batch_size": requested_batch_size,
                            "failed_batch_size": batch_size,
                            "coordinates_committed": coordinates_committed,
                            "coordinates_total": len(selected_records),
                            "cuda_oom_retry_count": cuda_oom_retry_count,
                        }
                    ) from exc
                cuda_oom_retry_count += 1
                committed_before_retries.append(coordinates_committed)
                cuda_oom_retry_transitions.append(
                    {"from": batch_size, "to": next_batch_size}
                )
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
                batch_size = next_batch_size
                backend = create_backend(replace(backend_config, batch_size=batch_size))
                continue
            effective_batch_sizes.append(batch_size)
            teacher_seconds += _elapsed(teacher_started)
            compression_started = perf_counter()
            row_by_example_id = {
                example.example_id: row for row, example in enumerate(chunk)
            }
            for example_id, rerun_row in row_by_example_id.items():
                for record_index, record in records_by_example_id[example_id]:
                    try:
                        selected_payload = _selected_payload_from_emission(
                            record,
                            payload=result.payload,
                            row=0 if native_streaming else rerun_row,
                            config=config,
                            position_index=(
                                selected_row_by_record[record_index]
                                - batch_selected_row_offset
                                if native_streaming
                                else None
                            ),
                        )
                    except (IndexError, KeyError, TypeError, ValueError) as exc:
                        raise _path_b_delivery_error(
                            record,
                            store=store,
                            failure_reason=(
                                "selected rerun payload could not be materialized: "
                                f"{exc}"
                            ),
                            selected_record_order=selected_record_order,
                            rerun_input_order=[example.example_id for example in chunk],
                            rerun_row_index=rerun_row,
                        ) from exc
                    mismatch_fields = _path_b_rerun_payload_mismatch(
                        record,
                        selected_payload,
                    )
                    if mismatch_fields:
                        raise _path_b_delivery_error(
                            record,
                            store=store,
                            failure_reason=(
                                "selected rerun payload does not match score-pass "
                                "source tuple"
                            ),
                            selected_record_order=selected_record_order,
                            rerun_input_order=[example.example_id for example in chunk],
                            rerun_row_index=rerun_row,
                            rerun_payload=selected_payload,
                            mismatch_fields=mismatch_fields,
                        )
                    if native_streaming:
                        _attach_long_tail_diagnostics(
                            [record],
                            [selected_payload],
                            config=config,
                            policy=_long_tail_policy(config),
                        )
                        selected_board = selected_board_for_long_tail(
                            str(selected_payload["long_tail_class"]),
                            include_long_tail_in_primary=config.include_long_tail_in_primary,
                            include_perverse_tail_in_primary=(
                                config.include_perverse_tail_in_primary
                            ),
                        )
                        record["selected_board"] = selected_board
                        selected_payload["selected_board"] = selected_board
                        payload_hash = _write_native_payload_shard(
                            _native_payload_stage_dir(config),
                            record_index=record_index,
                            payload=selected_payload,
                            delivery_path=config.delivery_path,
                        )
                        payload_summary = _payload_scalar_summary(
                            selected_payload,
                            record_index=record_index,
                        )
                        payload_summary["payload_hash"] = payload_hash
                        payload_summaries.append(payload_summary)
                        coordinates_committed += 1
                        del selected_payload
                    else:
                        payloads_by_record[record_index] = selected_payload
            compression_seconds += _elapsed(compression_started)
            batch_count += 1
            peak_host_memory_bytes = max(peak_host_memory_bytes, _host_rss_bytes())
            _notify_delivery_progress(
                config,
                phase="selected_rerun",
                event="progress",
                selected_examples_processed=start + len(chunk),
                selected_examples_total=len(all_selected_example_ids),
                selected_coordinates_committed=coordinates_committed,
                selected_coordinates_total=len(selected_records),
            )
            del result
            start += len(chunk)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    if config.rerun_metrics is not None:
        config.rerun_metrics.update(
            {
                "selected_rerun_examples": len(all_selected_example_ids),
                "selected_source_example_count": len(all_selected_example_ids),
                "selected_coordinate_count": len(selected_records),
                "requested_source_batch_size": requested_batch_size,
                "effective_source_batch_sizes": effective_batch_sizes,
                "source_batch_count": batch_count,
                "coordinate_compression_batch_count": batch_count,
                "selected_row_gather_seconds": None,
                "payload_write_seconds": 0.0,
                "selected_rerun_batch_size": batch_size,
                "selected_rerun_batch_count": batch_count,
                "selected_rerun_teacher_seconds": teacher_seconds,
                "selected_rerun_compression_seconds": compression_seconds,
                "selected_rerun_io_seconds": 0.0,
                "selected_rerun_examples_per_second": _rate(
                    len(selected_examples), teacher_seconds
                ),
                "selected_rerun_peak_host_memory_bytes": peak_host_memory_bytes,
                "selected_rerun_peak_device_memory_bytes": _device_peak_memory_bytes(),
                "selected_payload_shard_count": (
                    len(payload_summaries) if native_streaming else 1
                ),
                "selected_rerun_requested_batch_size": requested_batch_size,
                "selected_rerun_effective_batch_sizes": effective_batch_sizes,
                "cuda_oom_retry_count": cuda_oom_retry_count,
                "cuda_oom_retry_batch_transitions": cuda_oom_retry_transitions,
                "cuda_oom_failure_stage_counts": cuda_oom_failure_stage_counts,
                "coordinates_committed_before_each_retry": committed_before_retries,
            }
        )
    if native_streaming:
        return sorted(payload_summaries, key=lambda item: int(item["_record_index"]))
    return [payloads_by_record[index] for index in range(len(selected_records))]


def _native_streamed_payloads(config: ExemplarDeliveryConfig) -> bool:
    return config.execution_mode == NATIVE_C6_PATH_B_EXECUTION


def _is_recoverable_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "nv_err_no_memory" in text


def _next_rerun_batch_size(batch_size: int) -> int | None:
    if batch_size <= 1:
        return None
    return max(1, batch_size // 2)


def _write_native_payload_shard(
    selected_dir: Path,
    *,
    record_index: int,
    payload: dict[str, Any],
    delivery_path: str,
) -> str:
    selected_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "schema_version": "selected_exemplar_payload_shard_v1",
        "delivery_path": delivery_path,
        "delivery_authority_hash": payload.get("delivery_authority_hash"),
        "record_index": record_index,
        "selected_exemplars": [payload],
    }
    shard["payload_hash"] = _native_payload_hash(shard)
    _write_json_atomic(
        selected_dir / f"selected-exemplars-{record_index:05d}.json", shard
    )
    return str(shard["payload_hash"])


def _payload_scalar_summary(
    payload: dict[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "top_token_ids",
            "top_log_probs",
            "top_probs",
            "top_selection_mask",
            "bucket_masses",
        }
    }
    top_token_ids = payload.get("top_token_ids")
    if isinstance(top_token_ids, list) and top_token_ids:
        summary["payload_top_token_id"] = top_token_ids[0]
    summary["_record_index"] = record_index
    return summary


def _synchronize_native_payload_shards(
    selected_dir: Path,
    *,
    selected_records: list[dict[str, Any]],
) -> dict[int, str]:
    linkage = {
        (str(record["selected_example_id"]), int(record["selected_position"])): record
        for record in selected_records
    }
    hashes: dict[int, str] = {}
    for path in sorted(selected_dir.glob("selected-exemplars-*.json")):
        payload = read_json_object(path)
        records = payload.get("selected_exemplars")
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError(f"native payload shard is invalid: {path.name}")
        item = records[0]
        if not isinstance(item, dict):
            raise ValueError(f"native payload shard record is invalid: {path.name}")
        record = linkage.get(
            (str(item["selected_example_id"]), int(item["selected_position"]))
        )
        if record is None:
            raise ValueError(f"native payload shard is not selected: {path.name}")
        for key in (
            "corridor_fingerprint_id",
            "corridor_mode_id",
            "corridor_assignment_status",
            "selected_board",
            "semantic_tail_tag",
        ):
            item[key] = record.get(key)
        payload["payload_hash"] = _native_payload_hash(payload)
        _write_json_atomic(path, payload)
        hashes[int(payload["record_index"])] = str(payload["payload_hash"])
    return hashes


def _native_payload_stage_dir(config: ExemplarDeliveryConfig) -> Path:
    authority = (config.delivery_authority_hash or "unbound").replace(":", "-")
    return config.artifact_dir / ".staging-native-c6" / authority


def _prepare_native_payload_staging(
    config: ExemplarDeliveryConfig,
    *,
    selected_records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    stage = _native_payload_stage_dir(config)
    stage.mkdir(parents=True, exist_ok=True)
    public = config.artifact_dir / "selected_exemplars"
    for path in public.glob("selected-exemplars-*.json"):
        path.unlink()
    (public / "payload_index.json").unlink(missing_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    quarantined = 0
    quarantine_dir: Path | None = None
    for path in sorted(stage.glob("selected-exemplars-*.json")):
        try:
            record_index = int(path.stem.rsplit("-", 1)[1])
            payload = read_json_object(path)
            item = _validate_native_staged_payload(
                payload,
                path=path,
                record_index=record_index,
                selected_records=selected_records,
                expected_authority=config.delivery_authority_hash,
            )
        except (OSError, TypeError, ValueError, KeyError):
            if quarantine_dir is None:
                quarantine_dir = stage.parent / f"quarantine-{_now().replace(':', '-')}"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
            os.replace(path, quarantine_dir / path.name)
            quarantined += 1
            continue
        summary = _payload_scalar_summary(
            item,
            record_index=record_index,
        )
        summary["payload_hash"] = payload["payload_hash"]
        completed[record_index] = summary
    if config.rerun_metrics is not None:
        config.rerun_metrics.update(
            {
                "staging_directory": str(stage),
                "staging_preserved": bool(completed),
                "staging_payload_count": len(completed),
                "staging_quarantined_count": quarantined,
                "staging_quarantine_directory": (
                    str(quarantine_dir) if quarantine_dir is not None else None
                ),
            }
        )
    return completed


def selected_delivery_staging_diagnostic(
    artifact_dir: Path,
    *,
    delivery_authority_hash: str | None,
) -> dict[str, Any]:
    stage = (
        artifact_dir
        / ".staging-native-c6"
        / ((delivery_authority_hash or "unbound").replace(":", "-"))
    )
    files = sorted(stage.glob("selected-exemplars-*.json"))
    quarantine_dirs = sorted(stage.parent.glob("quarantine-*"))
    quarantined_files = sorted(
        path
        for directory in quarantine_dirs
        for path in directory.glob("selected-exemplars-*.json")
    )
    return {
        "staging_directory": str(stage),
        "staging_authority_hash": delivery_authority_hash,
        "staging_payload_count": len(files),
        "staging_payload_files": [path.name for path in files],
        "staging_preserved": bool(files),
        "staging_quarantine_directories": [str(path) for path in quarantine_dirs],
        "staging_quarantined_payload_count": len(quarantined_files),
    }


def _promote_native_payload_shards(config: ExemplarDeliveryConfig) -> None:
    stage = _native_payload_stage_dir(config)
    public = config.artifact_dir / "selected_exemplars"
    staged = sorted(stage.glob("selected-exemplars-*.json"))
    expected = len(config.authoritative_records or ())
    if len(staged) != expected:
        raise ValueError(
            "native selected payload transaction incomplete: "
            f"expected={expected} staged={len(staged)}"
        )
    staged_indices: list[int] = []
    for path in staged:
        payload = read_json_object(path)
        if payload.get("delivery_authority_hash") != config.delivery_authority_hash:
            raise ValueError(f"native payload authority hash mismatch: {path.name}")
        record_index = int(payload.get("record_index", -1))
        staged_indices.append(record_index)
        _validate_native_staged_payload(
            payload,
            path=path,
            record_index=record_index,
            selected_records=list(config.authoritative_records or ()),
            expected_authority=config.delivery_authority_hash,
        )
    if sorted(staged_indices) != list(range(expected)):
        raise ValueError(
            "native selected payload transaction has invalid record indices"
        )
    for path in staged:
        os.replace(path, public / path.name)
    try:
        stage.rmdir()
        stage.parent.rmdir()
    except OSError:
        pass


def _validate_native_staged_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    record_index: int,
    selected_records: list[dict[str, Any]],
    expected_authority: str | None,
) -> dict[str, Any]:
    if payload.get("schema_version") != "selected_exemplar_payload_shard_v1":
        raise ValueError(f"native payload schema mismatch: {path.name}")
    if payload.get("delivery_authority_hash") != expected_authority:
        raise ValueError(f"native payload authority hash mismatch: {path.name}")
    if payload.get("record_index") != record_index:
        raise ValueError(f"native payload record index mismatch: {path.name}")
    if not 0 <= record_index < len(selected_records):
        raise ValueError(f"native payload record index out of range: {path.name}")
    records = payload.get("selected_exemplars")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"native payload record count mismatch: {path.name}")
    item = records[0]
    if not isinstance(item, dict):
        raise ValueError(f"native payload record is invalid: {path.name}")
    expected = selected_records[record_index]
    if str(item.get("selected_example_id")) != str(
        expected.get("selected_example_id")
    ) or int(item.get("selected_position", -1)) != int(
        expected.get("selected_position", -1)
    ):
        raise ValueError(f"native payload coordinate mismatch: {path.name}")
    if payload.get("payload_hash") != _native_payload_hash(payload):
        raise ValueError(f"native payload hash mismatch: {path.name}")
    for field in _REQUIRED_SELECTED_PAYLOAD_FIELDS:
        if field not in item:
            raise ValueError(f"native payload missing {field}: {path.name}")
    return item


def _native_payload_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _host_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _device_peak_memory_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return None


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
    position_index: int | None = None,
) -> dict[str, Any]:
    position = int(record["source_position"])
    payload_position = position if position_index is None else position_index
    top_selection_mask = _payload_slice(
        payload, "top_selection_mask", row, payload_position
    )
    effective_top_k = int(
        _payload_scalar(payload, "effective_top_k", row, payload_position)
    )
    top_token_ids = _payload_slice(payload, "top_token_ids", row, payload_position)
    return {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["score_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "payload_ref": record["payload_ref"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "delivery_authority_hash": config.delivery_authority_hash,
        "selected_board": record.get("selected_board", PRIMARY_SELECTED_BOARD),
        "mode_key": record.get("mode_key"),
        "rank_by_board": record.get("rank_by_board", {}),
        "scores_by_board": record.get("scores_by_board", {}),
        "top_token_ids": top_token_ids,
        "top_log_probs": _payload_slice(
            payload, "top_log_probs", row, payload_position
        ),
        "top_probs": _payload_slice(payload, "top_probs", row, payload_position),
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": _payload_scalar(payload, "top_mass", row, payload_position),
        "tail_mass": _payload_scalar(payload, "tail_mass", row, payload_position),
        "bucket_masses": _payload_slice(
            payload, "bucket_masses", row, payload_position
        ),
        "teacher_entropy": _payload_scalar(
            payload, "teacher_entropy", row, payload_position
        ),
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
        "dynamic_mass_threshold": _dynamic_mass_threshold(config),
        "dynamic_top_k_max": _dynamic_top_k_max(config),
        "score_policy": config.score_policy,
        "source_payload": source_payload,
    }


def _long_tail_policy(config: ExemplarDeliveryConfig) -> LongTailPolicy:
    return LongTailPolicy(
        long_tail_warning_k=config.long_tail_warning_k,
        very_long_tail_warning_k=config.very_long_tail_warning_k,
        perverse_tail_warning_k=config.perverse_tail_warning_k,
        reject_perverse_exemplars=config.reject_perverse_exemplars,
    )


def _dynamic_top_k_max(config: ExemplarDeliveryConfig) -> int:
    if config.backend_config is None:
        return config.top_k
    return int(config.backend_config.dynamic_top_k_max)


def _dynamic_mass_threshold(config: ExemplarDeliveryConfig) -> float:
    if config.backend_config is None:
        return 0.95
    return float(config.backend_config.dynamic_mass_threshold)


def _candidate_is_perverse(
    candidate: Any,
    *,
    config: ExemplarDeliveryConfig,
) -> bool:
    diagnostic = _candidate_long_tail_diagnostic(candidate, config=config)
    return is_perverse_long_tail(diagnostic)


def _candidate_long_tail_diagnostic(
    candidate: Any,
    *,
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    effective_top_k = int(
        candidate.score_fields.get(
            "diagnostic_effective_top_k",
            candidate.score_fields.get("effective_top_k", 1),
        )
        or 1
    )
    diagnostic = long_tail_diagnostics(
        effective_top_k=effective_top_k,
        top_mass=float(
            candidate.score_fields.get(
                "diagnostic_top_mass",
                candidate.score_fields.get("top_mass", 0.0),
            )
            or 0.0
        ),
        vocab_size=config.vocab_size,
        dynamic_mass_threshold=_dynamic_mass_threshold(config),
        dynamic_top_k_max=_dynamic_top_k_max(config),
        policy=_long_tail_policy(config),
    )
    return diagnostic


def _primary_budget(config: ExemplarDeliveryConfig) -> int | None:
    return (
        config.primary_selected_exemplar_budget
        if config.primary_selected_exemplar_budget is not None
        else config.selected_exemplar_budget
    )


def _records_by_selected_board(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        board_id: [
            record
            for record in records
            if str(record.get("selected_board") or PRIMARY_SELECTED_BOARD) == board_id
        ]
        for board_id in (
            PRIMARY_SELECTED_BOARD,
            LONG_TAIL_UNCERTAINTY_BOARD,
            PERVERSE_TAIL_DIAGNOSTIC_BOARD,
        )
    }


def _route_records_for_delivery(
    records: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    for record in records:
        diagnostic = long_tail_diagnostics(
            effective_top_k=max(1, int(record.get("diagnostic_effective_top_k") or 1)),
            top_mass=float(record.get("diagnostic_top_mass") or 0.0),
            vocab_size=config.vocab_size,
            dynamic_mass_threshold=_dynamic_mass_threshold(config),
            dynamic_top_k_max=_dynamic_top_k_max(config),
            policy=_long_tail_policy(config),
        )
        record["selected_board"] = selected_board_for_long_tail(
            str(diagnostic["long_tail_class"]),
            include_long_tail_in_primary=config.include_long_tail_in_primary,
            include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        )
    return _cap_curriculum_records(records, config=config)


def _route_materialized_selected_exemplars(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, payload in zip(records, payloads, strict=True):
        board = selected_board_for_long_tail(
            str(payload["long_tail_class"]),
            include_long_tail_in_primary=config.include_long_tail_in_primary,
            include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        )
        record["selected_board"] = board
        payload["selected_board"] = board
        routed.append((record, payload))
    if config.authoritative_selection:
        selected_pairs = routed
    else:
        selected_records = _cap_curriculum_records(
            [record for record, _ in routed],
            config=config,
        )
        selected_ids = {id(record) for record in selected_records}
        selected_pairs = [pair for pair in routed if id(pair[0]) in selected_ids]
        selected_pairs.sort(
            key=lambda pair: (
                -float(pair[0]["selected_score"]),
                str(pair[0]["selected_example_id"]),
                int(pair[0]["selected_position"]),
            )
        )
    for rank, (record, _) in enumerate(selected_pairs, start=1):
        record["rank"] = rank
    return (
        [record for record, _ in selected_pairs],
        [payload for _, payload in selected_pairs],
    )


def _cap_curriculum_records(
    records: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    grouped = _records_by_selected_board(records)
    primary_limit = _primary_budget(config)
    if config.selected_exemplar_fraction is not None:
        fraction_limit = max(
            1,
            int(
                np.ceil(
                    len(grouped[PRIMARY_SELECTED_BOARD])
                    * config.selected_exemplar_fraction
                )
            ),
        )
        primary_limit = (
            fraction_limit
            if primary_limit is None
            else min(primary_limit, fraction_limit)
        )
    limits = {
        PRIMARY_SELECTED_BOARD: primary_limit,
        LONG_TAIL_UNCERTAINTY_BOARD: config.long_tail_side_board_cap,
        PERVERSE_TAIL_DIAGNOSTIC_BOARD: config.perverse_tail_side_board_cap,
    }
    return [
        record
        for board_id in (
            PRIMARY_SELECTED_BOARD,
            LONG_TAIL_UNCERTAINTY_BOARD,
            PERVERSE_TAIL_DIAGNOSTIC_BOARD,
        )
        for record in grouped[board_id][: limits[board_id]]
    ]


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


def _write_curriculum_routes(
    path: Path,
    selected_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Persist the current consumption-board routes independently of selection."""

    routes = [
        {
            "selected_example_id": record["selected_example_id"],
            "selected_position": record["selected_position"],
            "payload_key": (record.get("payload_identity") or {}).get("payload_key"),
            "curriculum_board": record.get("selected_board", PRIMARY_SELECTED_BOARD),
            "selection_roles": list(record.get("selection_roles") or ()),
        }
        for record in selected_records
    ]
    unique = {
        (str(route["selected_example_id"]), int(route["selected_position"]))
        for route in routes
    }
    write_json(
        path,
        {
            "schema_version": CURRICULUM_ROUTES_SCHEMA,
            "route_policy": "selected_board_consumption_v1",
            "routes": routes,
            "route_count": len(routes),
            "unique_coordinate_count": len(unique),
        },
    )
    return {"route_count": len(routes), "unique_coordinate_count": len(unique)}


def _attach_long_tail_diagnostics(
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
    policy: LongTailPolicy,
) -> None:
    if len(selected_records) != len(selected_payloads):
        raise ValueError("selected record/payload count mismatch")
    for record, payload in zip(selected_records, selected_payloads, strict=True):
        diagnostic = long_tail_diagnostics(
            effective_top_k=int(payload["effective_top_k"]),
            top_mass=float(payload["top_mass"]),
            vocab_size=int(payload["vocab_size"]),
            dynamic_mass_threshold=_dynamic_mass_threshold(config),
            dynamic_top_k_max=_dynamic_top_k_max(config),
            policy=policy,
        )
        payload.update(diagnostic)
        payload["dynamic_top_k"].update(
            {
                "dynamic_mass_threshold": diagnostic["dynamic_mass_threshold"],
                "dynamic_top_k_max": diagnostic["dynamic_top_k_max"],
                "top_k_saturated": diagnostic["top_k_saturated"],
            }
        )
        record.update(diagnostic)
        record["dynamic_top_k"] = dict(payload["dynamic_top_k"])
        selected_board = str(record.get("selected_board") or PRIMARY_SELECTED_BOARD)
        record["selected_board"] = selected_board
        payload["selected_board"] = selected_board
        tag = semantic_tail_tag(long_tail_class=str(diagnostic["long_tail_class"]))
        record["semantic_tail_tag"] = tag
        payload["semantic_tail_tag"] = tag


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
