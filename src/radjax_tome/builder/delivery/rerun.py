"""rerun ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .measurement import SelectedPassMeasurementControl
from .payloads import (
    _attach_long_tail_diagnostics,
    _candidate_is_perverse,
    _long_tail_policy,
    _records_by_selected_board,
    _route_materialized_selected_exemplars,
    _route_records_for_delivery,
    _selected_payloads_from_one_pass_capture,
)
from .reporting import _elapsed, _now, _selected_board_summary
from .staging import (
    _native_streamed_payloads,
    _notify_delivery_progress,
    _prepare_native_payload_staging,
    _selected_payloads_from_backend,
)
from .validation import _validate_path_b_score_pass_records


def _materialize_selected_payloads(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
    config: ExemplarDeliveryConfig,
    completed_record_indices: set[int] | None = None,
    existing_payload_summaries: Mapping[int, dict[str, Any]] | None = None,
    _measurement_control: SelectedPassMeasurementControl | None = None,
    full_payloads: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Choose the Path-A or Path-B payload mechanism from the owner layer."""
    if config.delivery_path == TWO_PASS_RERUN_SELECTED:
        return _selected_payloads_from_backend(
            selected_records,
            store=store,
            examples=examples,
            config=config,
            completed_record_indices=completed_record_indices,
            existing_payload_summaries=existing_payload_summaries,
            _measurement_control=_measurement_control,
            full_payloads=full_payloads,
        )
    return _selected_payloads_from_one_pass_capture(
        selected_records,
        store=store,
        config=config,
    )


def run_selected_delivery_rerun(
    config: ExemplarDeliveryConfig,
    *,
    _measurement_control: SelectedPassMeasurementControl | None = None,
) -> PreparedSelectedDelivery:
    """Select and materialize rerun payloads before final corridor export."""

    _validate_delivery_config(config)
    if _measurement_control is not None:
        _measurement_control.validate_for_output(config.artifact_dir)
        if config.authoritative_records is None or not config.authoritative_selection:
            raise ValueError(
                "selected-pass measurement requires frozen authoritative C5 records"
            )
        if config.rerun_metrics is None:
            raise ValueError(
                "selected-pass measurement requires an evidence metrics sink"
            )
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
    full_payloads = [] if config.retain_full_payloads_for_publication else None
    selected_payloads = _materialize_selected_payloads(
        selected_records,
        store=store,
        examples=examples,
        config=config,
        completed_record_indices=set(staged_payload_summaries),
        existing_payload_summaries=staged_payload_summaries,
        _measurement_control=_measurement_control,
        full_payloads=full_payloads,
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
    publication_payloads = None
    if config.retain_full_payloads_for_publication:
        full_by_index = {
            int(item["_record_index"]): item
            for item in (full_payloads or [])
        }
        publication_payloads = tuple(
            dict(full_by_index[int(summary["_record_index"])])
            for summary in selected_payloads
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
        publication_payloads=publication_payloads,
    )


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


__all__ = [name for name in globals() if not name.startswith("__")]
