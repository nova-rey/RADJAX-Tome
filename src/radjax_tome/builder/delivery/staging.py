"""staging ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .measurement import (
    SelectedPassExecutionDiagnostics,
    SelectedPassMeasurementControl,
)
from .payloads import (
    _attach_long_tail_diagnostics,
    _long_tail_policy,
    _payload_scalar_summary,
    _selected_payload_from_emission,
)
from .reporting import _elapsed, _now, _rate
from .validation import _path_b_delivery_error, _path_b_rerun_payload_mismatch


def _selected_payloads_from_backend(
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
    requested_batch_size = max(1, config.selected_rerun_batch_size)
    batch_size = (
        min(requested_batch_size, _measurement_control.effective_execution_cap)
        if _measurement_control is not None
        else requested_batch_size
    )
    backend_config = replace(
        config.backend_config,
        target_policy="dynamic_cascaded_soft_labels_v1",  # type: ignore[arg-type]
        exemplar_source_policy="dynamic_cascaded_soft_labels_v1",
        batch_size=batch_size,
    )
    diagnostics = (
        SelectedPassExecutionDiagnostics(
            control=_measurement_control,
            requested_batch_size=requested_batch_size,
        )
        if _measurement_control is not None
        else None
    )
    construction_started = perf_counter()
    backend = create_backend(backend_config)
    backend_observer_attached = _attach_measurement_observer(backend, diagnostics)
    if diagnostics is not None:
        diagnostics.add("backend_construction", _elapsed(construction_started))
    payloads_by_record: dict[int, dict[str, Any]] = {
        int(record_index): dict(summary)
        for record_index, summary in existing_payload_summaries.items()
    }
    native_streaming = _native_streamed_payloads(config)
    payload_summaries: list[dict[str, Any]] = [
        dict(existing_payload_summaries[index])
        for index in sorted(existing_payload_summaries)
    ]
    position_preparation_started = perf_counter() if diagnostics is not None else None
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
    if diagnostics is not None and position_preparation_started is not None:
        diagnostics.add(
            "selected_position_index_preparation",
            _elapsed(position_preparation_started),
        )
    teacher_seconds = 0.0
    compression_seconds = 0.0
    peak_host_memory_bytes = _host_rss_bytes()
    batch_count = 0
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
            position_started = perf_counter()
            batch_selected_row_offset = sum(
                len(positions_by_example_id[example_id])
                for example_id in selected_example_ids[:start]
            )
            if diagnostics is not None:
                diagnostics.add(
                    "selected_position_index_preparation",
                    _elapsed(position_started),
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
                retry_started = perf_counter()
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
                batch_size = next_batch_size
                backend = create_backend(replace(backend_config, batch_size=batch_size))
                backend_observer_attached = _attach_measurement_observer(
                    backend, diagnostics
                )
                if diagnostics is not None:
                    diagnostics.add("retry_reload", _elapsed(retry_started))
                    diagnostics.oom_events.append(
                        {
                            "from": cuda_oom_retry_transitions[-1]["from"],
                            "to": next_batch_size,
                        }
                    )
                continue
            effective_batch_sizes.append(batch_size)
            teacher_seconds += _elapsed(teacher_started)
            if diagnostics is not None:
                if not backend_observer_attached:
                    diagnostics.add_backend_diagnostics(result.metadata)
                diagnostics.record_batch(
                    source_count=len(chunk),
                    coordinate_count=sum(
                        len(positions_by_example_id[example.example_id])
                        for example in chunk
                    ),
                    selected_positions_per_source=(
                        len(positions_by_example_id[example.example_id])
                        for example in chunk
                    ),
                    result=result,
                    effective_size=batch_size,
                )
            compression_started = perf_counter()
            payload_write_seconds = 0.0
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
                        write_started = perf_counter()
                        payload_hash = _write_native_payload_shard(
                            _native_payload_stage_dir(config),
                            record_index=record_index,
                            payload=selected_payload,
                            delivery_path=config.delivery_path,
                            _measurement_diagnostics=diagnostics,
                        )
                        payload_summary = _payload_scalar_summary(
                            selected_payload,
                            record_index=record_index,
                        )
                        payload_summary["payload_hash"] = payload_hash
                        payload_summaries.append(payload_summary)
                        if full_payloads is not None:
                            full_payloads.append(
                                {
                                    **selected_payload,
                                    "_record_index": record_index,
                                }
                            )
                        coordinates_committed += 1
                        if diagnostics is not None:
                            write_seconds = _elapsed(write_started)
                            if not diagnostics.staging_subphases_observed:
                                diagnostics.add(
                                    "hashing_json_atomic_write_fsync",
                                    write_seconds,
                                )
                            payload_write_seconds += write_seconds
                        del selected_payload
                    else:
                        payloads_by_record[record_index] = selected_payload
            compression_elapsed = _elapsed(compression_started)
            compression_seconds += compression_elapsed
            if diagnostics is not None:
                diagnostics.add(
                    "payload_conversion_linkage_validation",
                    max(0.0, compression_elapsed - payload_write_seconds),
                )
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
        cleanup_started = perf_counter()
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        if diagnostics is not None:
            diagnostics.add("backend_close_cleanup", _elapsed(cleanup_started))
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
        if diagnostics is not None:
            config.rerun_metrics["selected_pass_execution_v1"] = diagnostics.finish()
    if native_streaming:
        return sorted(payload_summaries, key=lambda item: int(item["_record_index"]))
    return [payloads_by_record[index] for index in range(len(selected_records))]


def _native_streamed_payloads(config: ExemplarDeliveryConfig) -> bool:
    return config.execution_mode == NATIVE_C6_PATH_B_EXECUTION


def _attach_measurement_observer(
    backend: Any, diagnostics: SelectedPassExecutionDiagnostics | None
) -> bool:
    if diagnostics is None:
        return False
    attach = getattr(backend, "_attach_selected_pass_measurement_observer", None)
    if not callable(attach):
        return False
    attach(diagnostics)
    return True


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
    _measurement_diagnostics: SelectedPassExecutionDiagnostics | None = None,
) -> str:
    selected_dir.mkdir(parents=True, exist_ok=True)
    if _measurement_diagnostics is not None:
        _measurement_diagnostics.record_payload_anatomy(
            payload,
            stage="initial_staging",
            record_index=record_index,
        )
    shard = {
        "schema_version": "selected_exemplar_payload_shard_v1",
        "delivery_path": delivery_path,
        "delivery_authority_hash": payload.get("delivery_authority_hash"),
        "record_index": record_index,
        "selected_exemplars": [payload],
    }
    hash_started = perf_counter() if _measurement_diagnostics is not None else None
    shard["payload_hash"] = _native_payload_hash(
        shard, _measurement_diagnostics=_measurement_diagnostics
    )
    if _measurement_diagnostics is not None and hash_started is not None:
        _measurement_diagnostics.add_staging_phase(
            "canonical_body_encoding_hash", _elapsed(hash_started)
        )
    _write_json_atomic(
        selected_dir / f"selected-exemplars-{record_index:05d}.json",
        shard,
        _measurement_diagnostics=_measurement_diagnostics,
    )
    return str(shard["payload_hash"])


def _native_payload_hash(
    payload: dict[str, Any],
    *,
    _measurement_diagnostics: SelectedPassExecutionDiagnostics | None = None,
) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if _measurement_diagnostics is not None:
        _measurement_diagnostics.count_operations(
            "canonical_body_encoding_hash",
            bytes_read=len(encoded),
            records=1,
            hashes=1,
        )
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    _measurement_diagnostics: SelectedPassExecutionDiagnostics | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encode_started = perf_counter() if _measurement_diagnostics is not None else None
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if _measurement_diagnostics is not None and encode_started is not None:
        _measurement_diagnostics.add_staging_phase(
            "staging_json_encoding", _elapsed(encode_started)
        )
        _measurement_diagnostics.count_operations(
            "staging_json_encoding",
            bytes_written=len(encoded.encode("utf-8")),
            records=1,
        )
    handle = temporary.open("w", encoding="utf-8")
    try:
        write_started = perf_counter() if _measurement_diagnostics is not None else None
        handle.write(encoded)
        if _measurement_diagnostics is not None and write_started is not None:
            _measurement_diagnostics.add_staging_phase(
                "temporary_file_write", _elapsed(write_started)
            )
            _measurement_diagnostics.count_operations(
                "temporary_file_write",
                bytes_written=len(encoded.encode("utf-8")),
                files_opened=1,
            )
    finally:
        close_started = perf_counter() if _measurement_diagnostics is not None else None
        handle.close()
        if _measurement_diagnostics is not None and close_started is not None:
            _measurement_diagnostics.add_staging_phase(
                "temporary_file_close", _elapsed(close_started)
            )
    replace_started = perf_counter() if _measurement_diagnostics is not None else None
    os.replace(temporary, path)
    if _measurement_diagnostics is not None and replace_started is not None:
        _measurement_diagnostics.add_staging_phase(
            "atomic_replacement", _elapsed(replace_started)
        )
        _measurement_diagnostics.count_operations(
            "atomic_replacement", files_replaced=1, files_created=1
        )


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


def _synchronize_native_payload_shards(
    selected_dir: Path,
    *,
    selected_records: list[dict[str, Any]],
    _measurement_metrics: dict[str, Any] | None = None,
) -> dict[int, str]:
    linkage = {
        (str(record["selected_example_id"]), int(record["selected_position"])): record
        for record in selected_records
    }
    hashes: dict[int, str] = {}
    for path in sorted(selected_dir.glob("selected-exemplars-*.json")):
        synchronization_started = perf_counter()
        input_bytes = path.stat().st_size
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
        if _measurement_metrics is not None:
            anatomy = _measurement_metrics.get("payload_anatomy")
            if isinstance(anatomy, dict):
                # The post-linkage body is measured after the linkage fields
                # are present and before the atomic rewrite.
                effective = anatomy.setdefault("effective_top_k", [])
                if isinstance(effective, list) and isinstance(
                    item.get("effective_top_k"), (int, float)
                ):
                    effective.append(int(item["effective_top_k"]))
                totals = anatomy.setdefault("stage_totals", {})
                if isinstance(totals, dict):
                    stage_total = totals.setdefault(
                        "post_linkage",
                        {
                            "count": 0,
                            "canonical_bytes": 0,
                            "pretty_bytes": 0,
                            "bytes_read": 0,
                            "bytes_rewritten": 0,
                        },
                    )
                    canonical = _native_payload_hash(payload)
                    stage_total["count"] += 1
                    stage_total["canonical_bytes"] += len(
                        json.dumps(
                            {k: v for k, v in payload.items() if k != "payload_hash"},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    stage_total["bytes_read"] += input_bytes
                    stage_total["canonical_payload_hashes"] = (
                        stage_total.get("canonical_payload_hashes", 0) + 1
                    )
                    del canonical
        payload["payload_hash"] = _native_payload_hash(payload)
        _write_json_atomic(path, payload)
        output_bytes = path.stat().st_size
        if _measurement_metrics is not None:
            phases = _measurement_metrics.setdefault("phases", {})
            phase = phases.setdefault(
                "corridor_synchronization_rewrite",
                {"seconds": 0.0, "status": "measured_host_wall"},
            )
            phase["seconds"] += _elapsed(synchronization_started)
            phase["status"] = "measured_host_wall"
            counts = _measurement_metrics.setdefault("operation_counts", {})
            count = counts.setdefault(
                "corridor_synchronization_rewrite",
                {
                    "bytes_read": 0,
                    "bytes_written": 0,
                    "records": 0,
                    "files_opened": 0,
                    "files_created": 0,
                    "files_replaced": 0,
                    "files_removed": 0,
                    "hashes": 0,
                    "validations": 0,
                },
            )
            count["bytes_read"] += input_bytes
            count["bytes_written"] += output_bytes
            count["records"] += 1
            count["files_opened"] += 1
            count["files_replaced"] += 1
            count["hashes"] += 1
            count["validations"] += 1
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


@dataclass(frozen=True)
class V4SealedShard:
    """One verified, immutable shard in a resumable v4 delivery transaction.

    This is deliberately a delivery primitive rather than a second production
    state machine.  The Path-B orchestrator decides when to emit and promote;
    this helper only makes a contiguous sequence of JSONL shard files safe to
    reuse after an interrupted delivery.
    """

    shard_id: int
    first_selection_index: int
    record_count: int
    sha256: str
    path: str


@dataclass(frozen=True)
class V4ShardResumeState:
    """The verified contiguous prefix available to a resumed v4 delivery."""

    stage: Path
    config_sha256: str
    payload_records_per_shard: int
    sealed_shards: tuple[V4SealedShard, ...]

    @property
    def completed_record_count(self) -> int:
        return sum(item.record_count for item in self.sealed_shards)

    @property
    def next_shard_id(self) -> int:
        return len(self.sealed_shards)


_V4_STAGING_RECEIPT = "v4-shard-receipt.json"
_V4_STAGING_COMPLETE = "v4-shard-transaction-complete.json"


def prepare_v4_shard_staging(
    stage: Path,
    *,
    config: Mapping[str, Any],
    payload_records_per_shard: int,
) -> V4ShardResumeState:
    """Open and verify a resumable v4 shard transaction.

    Only a contiguous prefix whose receipt, filenames, counts, and raw digests
    all agree is reusable.  Interrupted temporary files are never receipts and
    are discarded before the caller receives the resume state.
    """
    if (
        not isinstance(payload_records_per_shard, int)
        or isinstance(payload_records_per_shard, bool)
        or payload_records_per_shard < 1
    ):
        raise ValueError("payload_records_per_shard must be a positive integer")
    stage.mkdir(parents=True, exist_ok=True)
    shards = stage / "shards"
    shards.mkdir(exist_ok=True)
    for temporary in shards.glob("*.tmp"):
        temporary.unlink()
    config_sha256 = _v4_staging_digest(config)
    receipt_path = stage / _V4_STAGING_RECEIPT
    if not receipt_path.exists():
        _write_json_atomic(
            receipt_path,
            {
                "schema_version": "radjax_tome_v4_shard_staging_receipt_v1",
                "config_sha256": config_sha256,
                "payload_records_per_shard": payload_records_per_shard,
                "sealed_shards": [],
            },
        )
        return V4ShardResumeState(stage, config_sha256, payload_records_per_shard, ())
    receipt = read_json_object(receipt_path)
    if receipt.get("schema_version") != "radjax_tome_v4_shard_staging_receipt_v1":
        raise ValueError("v4 staging receipt schema mismatch")
    if receipt.get("config_sha256") != config_sha256:
        raise ValueError("v4 staging configuration mismatch; refusing resume")
    if receipt.get("payload_records_per_shard") != payload_records_per_shard:
        raise ValueError("v4 staging shard capacity mismatch; refusing resume")
    raw_sealed = receipt.get("sealed_shards")
    if not isinstance(raw_sealed, list):
        raise ValueError("v4 staging receipt sealed_shards is invalid")
    sealed: list[V4SealedShard] = []
    expected_first = 0
    for expected_id, raw in enumerate(raw_sealed):
        if not isinstance(raw, dict):
            raise ValueError("v4 staging receipt shard is invalid")
        item = _v4_sealed_shard_from_dict(raw)
        if item.shard_id != expected_id:
            raise ValueError("v4 staging receipt has a shard gap or reorder")
        if item.first_selection_index != expected_first:
            raise ValueError("v4 staging receipt has overlapping shard ranges")
        if not 1 <= item.record_count <= payload_records_per_shard:
            raise ValueError("v4 staging receipt shard record count is invalid")
        shard_path = stage / item.path
        if shard_path != shards / f"shard-{item.shard_id:05d}.jsonl":
            raise ValueError("v4 staging receipt shard path is invalid")
        if not shard_path.is_file() or _v4_file_digest(shard_path) != item.sha256:
            raise ValueError("v4 staging sealed shard digest mismatch")
        if _v4_jsonl_record_count(shard_path) != item.record_count:
            raise ValueError("v4 staging sealed shard row count mismatch")
        sealed.append(item)
        expected_first += item.record_count
    expected_paths = {item.path for item in sealed}
    actual_paths = {
        path.relative_to(stage).as_posix() for path in shards.glob("shard-*.jsonl")
    }
    if actual_paths != expected_paths:
        raise ValueError("v4 staging contains unreceipted or missing sealed shards")
    return V4ShardResumeState(
        stage, config_sha256, payload_records_per_shard, tuple(sealed)
    )


def seal_v4_shard(
    state: V4ShardResumeState,
    records: Iterable[bytes],
) -> V4ShardResumeState:
    """Atomically seal the next count-bounded JSONL shard and its receipt."""
    if (state.stage / _V4_STAGING_COMPLETE).exists():
        raise ValueError("v4 staging transaction is already complete")
    shard_id = state.next_shard_id
    relative = f"shards/shard-{shard_id:05d}.jsonl"
    final = state.stage / relative
    if final.exists():
        raise ValueError("v4 staging shard already exists without a valid receipt")
    temporary = final.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temporary.open("wb") as handle:
            for record in records:
                if not isinstance(record, bytes) or not record or b"\n" in record:
                    raise ValueError("v4 shard records must be nonempty JSONL lines")
                if count == state.payload_records_per_shard:
                    raise ValueError("v4 shard exceeds payload_records_per_shard")
                line = record + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
        if count == 0:
            raise ValueError("v4 shard must contain at least one record")
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    item = V4SealedShard(
        shard_id=shard_id,
        first_selection_index=state.completed_record_count,
        record_count=count,
        sha256="sha256:" + digest.hexdigest(),
        path=relative,
    )
    _write_v4_staging_receipt(state, (*state.sealed_shards, item))
    return V4ShardResumeState(
        state.stage,
        state.config_sha256,
        state.payload_records_per_shard,
        (*state.sealed_shards, item),
    )


def complete_v4_shard_staging(
    state: V4ShardResumeState,
    *,
    expected_record_count: int,
) -> None:
    """Mark a verified complete transaction ready for later final promotion."""
    if state.completed_record_count != expected_record_count:
        raise ValueError(
            "v4 staging transaction incomplete: "
            f"expected={expected_record_count} staged={state.completed_record_count}"
        )
    _write_json_atomic(
        state.stage / _V4_STAGING_COMPLETE,
        {
            "schema_version": "radjax_tome_v4_shard_staging_complete_v1",
            "config_sha256": state.config_sha256,
            "record_count": expected_record_count,
            "shard_count": len(state.sealed_shards),
        },
    )


def _write_v4_staging_receipt(
    state: V4ShardResumeState, sealed: tuple[V4SealedShard, ...]
) -> None:
    _write_json_atomic(
        state.stage / _V4_STAGING_RECEIPT,
        {
            "schema_version": "radjax_tome_v4_shard_staging_receipt_v1",
            "config_sha256": state.config_sha256,
            "payload_records_per_shard": state.payload_records_per_shard,
            "sealed_shards": [
                {
                    "shard_id": item.shard_id,
                    "first_selection_index": item.first_selection_index,
                    "record_count": item.record_count,
                    "sha256": item.sha256,
                    "path": item.path,
                }
                for item in sealed
            ],
        },
    )


def _v4_sealed_shard_from_dict(raw: dict[str, Any]) -> V4SealedShard:
    required = {
        "shard_id",
        "first_selection_index",
        "record_count",
        "sha256",
        "path",
    }
    if set(raw) != required:
        raise ValueError("v4 staging receipt shard keys are invalid")
    numeric_keys = ("shard_id", "first_selection_index", "record_count")
    if any(isinstance(raw[key], bool) for key in numeric_keys):
        raise ValueError("v4 staging receipt shard numeric field is invalid")
    if not all(isinstance(raw[key], int) for key in numeric_keys):
        raise ValueError("v4 staging receipt shard numeric field is invalid")
    if not isinstance(raw["path"], str) or not isinstance(raw["sha256"], str):
        raise ValueError("v4 staging receipt shard digest or path is invalid")
    digest = raw["sha256"]
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError("v4 staging receipt shard digest is invalid")
    try:
        int(digest.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError("v4 staging receipt shard digest is invalid") from exc
    return V4SealedShard(
        shard_id=raw["shard_id"],
        first_selection_index=raw["first_selection_index"],
        record_count=raw["record_count"],
        sha256=digest,
        path=raw["path"],
    )


def _v4_staging_digest(config: Mapping[str, Any]) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    )


def _v4_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _v4_jsonl_record_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n") or not line[:-1]:
                raise ValueError("v4 staging sealed shard JSONL is invalid")
            count += 1
    return count


__all__ = [name for name in globals() if not name.startswith("__")]
