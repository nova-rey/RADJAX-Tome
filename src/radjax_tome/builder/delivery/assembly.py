"""assembly ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .modes import (
    COMPACT_K_IMMUTABLE_BODY,
    COMPACT_K_MONOLITHIC,
    compact_body_from_logical_payload,
    compact_payload_for_storage,
)
from .payloads import _primary_budget
from .reporting import (
    _delivery_timing_fields,
    _elapsed,
    _leaderboard_report,
    _long_tail_observations,
    _now,
)
from .simple_compact_body import write_compact_body_store_pipelined_from_compact
from .staging import (
    _native_payload_stage_dir,
    _native_streamed_payloads,
    _promote_native_payload_shards,
    _synchronize_native_payload_shards,
    _write_json_atomic,
)


def _publish_immutable_bodies(prepared: PreparedSelectedDelivery) -> None:
    """Publish compact bodies through the accepted transaction for mode 3."""

    from radjax_contract.tome.m8g import (
        _m8g_fv3,
        body_raw_digest,
        encode_compact_body,
        manifest_semantic_id,
    )

    from .immutable_body import ImmutableBodyTransaction

    config = prepared.config
    transaction = ImmutableBodyTransaction(
        config.artifact_dir / "m8g_immutable",
        profile="producer_evidence",
    )
    authority = bytes.fromhex(
        (config.delivery_authority_hash or "sha256:" + "00" * 32).split(":", 1)[-1]
    )
    for payload in prepared.selected_payloads:
        compact = compact_payload_for_storage(payload)
        body = compact_body_from_logical_payload(compact, profile="producer_evidence")
        body_bytes = encode_compact_body(body)
        manifest = {
            "schema_version": "selected_exemplar_manifest_v1",
            "profile": "producer_evidence",
            "selected_example_id": str(payload["selected_example_id"]),
            "selected_position": int(payload["selected_position"]),
            "source_passport_id": (
                f"{payload['selected_example_id']}:{payload['selected_position']}"
            ),
            "corridor_mode_id": str(payload.get("mode_key"))
            if payload.get("mode_key") is not None
            else None,
            "corridor_fingerprint_id": None,
            "selection_obligation_count": 0,
            "selection_obligations": [],
            "body_semantic_id": body.semantic_id,
            "body_raw_digest": body_raw_digest(body_bytes),
            "authority_id": authority,
            "selection_authority_id": authority,
            "package_role": "producer_evidence",
        }
        manifest["manifest_semantic_id"] = manifest_semantic_id(manifest)
        transaction.commit(
            body,
            manifest,
            canonical_manifest_bytes=_m8g_fv3(manifest),
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
            _measurement_metrics=(
                config.rerun_metrics.get("selected_pass_execution_v1")
                if config.rerun_metrics is not None
                else None
            ),
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
    if _native_streamed_payloads(config) and config.representation_mode == COMPACT_K_MONOLITHIC:
        if prepared.publication_payloads is None:
            raise ValueError("compact C6 requires the canonical payload handoff")
        store_dir = selected_dir / "compact_body_store"
        write_compact_body_store_pipelined_from_compact(
            store_dir, prepared.publication_payloads, profile="compact_k_monolithic"
        )
    if not _native_streamed_payloads(config):
        if config.representation_mode == COMPACT_K_MONOLITHIC:
            store_dir = selected_dir / "compact_body_store"
            write_compact_body_store_pipelined_from_compact(
                store_dir, selected_payloads, profile="compact_k_monolithic"
            )
            metadata = [
                json.loads(line)
                for line in (store_dir / "metadata.jsonl").read_text().splitlines()
                if line
            ]
            serialized_payloads = metadata
        elif config.representation_mode == COMPACT_K_IMMUTABLE_BODY:
            serialized_payloads = [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "top_token_ids",
                        "top_probs",
                        "top_log_probs",
                        "top_selection_mask",
                    }
                }
                for item in selected_payloads
            ]
        else:
            serialized_payloads = selected_payloads
        write_json(
            selected_dir / "selected-exemplars-00000.json",
            {
                "schema_version": (
                    "selected_exemplar_compact_monolithic_v1"
                    if config.representation_mode == COMPACT_K_MONOLITHIC
                    else "selected_exemplar_payload_shard_v1"
                ),
                "delivery_path": config.delivery_path,
                "long_tail_summary": prepared.tail_summary,
                "selected_board_summary": prepared.selected_board_summary,
                "selected_exemplars": serialized_payloads,
            },
        )
    if config.representation_mode == COMPACT_K_IMMUTABLE_BODY:
        _publish_immutable_bodies(prepared)

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
        "representation_mode_requested": config.representation_mode,
        "representation_mode_executed": config.representation_mode,
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
        "materialization_counters": rerun_metrics.get("materialization_counters", {}),
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


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


__all__ = [name for name in globals() if not name.startswith("__")]
