"""staging ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
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
    from .rerun import _unique_selected_example_ids

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


__all__ = [name for name in globals() if not name.startswith("__")]
