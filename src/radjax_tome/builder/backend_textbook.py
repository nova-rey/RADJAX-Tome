from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.backends import (
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
)
from radjax_tome.builder.exemplar_selection import (
    EXEMPLAR_SELECTION_MANIFEST_FILENAME,
    MULTI_LEADERBOARD_SELECTOR_POLICY,
    PATH_A_FULFILLMENT_POLICY,
    PATH_B_FULFILLMENT_POLICY,
    build_exemplar_selection_manifest,
    write_exemplar_selection_manifest,
)
from radjax_tome.builder.teacher_textbook import (
    DEFAULT_FAKE_TEACHER_MODEL_ID,
    DEFAULT_FAKE_TOKENIZER_ID,
    DEFAULT_FAKE_VOCAB_SIZE,
    TEACHER_TEXTBOOK_VERSION,
    TinyTextExample,
    load_text_examples,
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from radjax_tome.corpora import stringify_corpus_provenance
from radjax_tome.io.json import write_json
from radjax_tome.provenance import (
    teacher_model_provenance_summary,
    teacher_model_target_params,
)
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)
from radjax_tome.targets.store import TeacherTargetStore, target_store_shard_path
from radjax_tome.tome import write_cover_page

STREAMING_RUN_MANIFEST_FILENAME = "run_manifest.json"
STREAMING_PROGRESS_LOG_FILENAME = "progress_log.jsonl"
STREAMING_FAILURE_REPORT_FILENAME = "failure_report.json"
STREAMING_RUN_MANIFEST_SCHEMA = "streaming_run_manifest_v1"
STREAMING_FAILURE_REPORT_SCHEMA = "streaming_failure_report_v1"
STREAMING_BUILDER_VERSION = "spec_4_6_streaming_backend_builder_v1"


@dataclass(frozen=True)
class BackendTeacherTextbookBuildConfig:
    output_dir: Path
    dataset_path: Path | None = None
    teacher_backend: str = "cpu_reference"
    runtime_mode: str = "cpu"
    target_policy: str = "dense_logits"
    teacher_model_id: str = DEFAULT_FAKE_TEACHER_MODEL_ID
    tokenizer_id: str = DEFAULT_FAKE_TOKENIZER_ID
    sequence_length: int = 16
    batch_size: int = 2
    max_examples: int = 4
    vocab_size: int = DEFAULT_FAKE_VOCAB_SIZE
    top_k: int = 8
    num_buckets: int = 4
    exemplar_source_policy: str = "dynamic_cascaded_soft_labels_v1"
    exemplar_capture_mode: str = "one_pass_candidate"
    exemplar_second_pass_source_policy: str = "dynamic_cascaded_soft_labels_v1"
    gpu_batch_size_mode: str = "preset"
    gpu_batch_size_preset: int = 8
    gpu_batch_size_custom: int | None = None
    gpu_batch_size_auto_min: int = 1
    gpu_batch_size_auto_max: int = 64
    fallback_policy: str = "error"
    exemplar_selector_policy: str = MULTI_LEADERBOARD_SELECTOR_POLICY
    exemplar_selection_enabled: bool = False
    exemplar_selection_board_capacity: int = 16
    exemplar_selection_budget_examples: int | None = None
    exemplar_selection_budget_fraction: float | None = None
    exemplar_fulfillment_policy: str = "auto"
    local_files_only: bool = True
    allow_downloads: bool = False
    overwrite: bool = False
    corpus_manifest_path: Path | None = None
    teacher_model_provenance_path: Path | None = None
    streaming: bool = False
    resume: bool = False
    shard_size_examples: int | None = None
    progress_log_path: Path | None = None
    run_manifest_path: Path | None = None
    fail_fast: bool = True


def build_backend_teacher_textbook(
    config: BackendTeacherTextbookBuildConfig,
) -> Any:
    _validate_backend_build_config(config)
    if config.streaming:
        return build_streaming_backend_teacher_textbook(config)
    examples = load_text_examples(config.dataset_path, max_examples=config.max_examples)
    if config.output_dir.exists():
        if not config.overwrite:
            raise ValueError(
                f"TeacherTextbook output already exists: {config.output_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(config.output_dir)

    backend_config = teacher_backend_config_from_build_config(config)
    backend = create_backend(backend_config)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    shard_count = (len(examples) + config.batch_size - 1) // config.batch_size
    first_metadata: dict[str, object] | None = None
    target_type = _artifact_target_type(config.target_policy, first_metadata)
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id=config.teacher_model_id,
        model_family=config.teacher_backend,
        tokenizer_id=config.tokenizer_id,
        tokenizer_hash=None,
        vocab_size=config.vocab_size,
        target_type=target_type,
        dtype="float32",
        sequence_length=config.sequence_length,
        num_examples=len(examples),
        shard_count=shard_count,
        created_by="radjax_tome.builder.backend_textbook",
        created_at=created_at,
        source=_metadata_source(config),
        provenance={
            "phase": "radjax-tome-spec-3.3F10",
            "teacher_backend": config.teacher_backend,
        },
        target_params=_target_params(config, {}),
    )
    store = TeacherTargetStore.create(config.output_dir, metadata, overwrite=True)
    try:
        for shard_id, start in enumerate(range(0, len(examples), config.batch_size)):
            batch_examples = examples[start : start + config.batch_size]
            batch = TeacherBatchInput(
                example_ids=tuple(example.example_id for example in batch_examples),
                texts=tuple(example.text for example in batch_examples),
            )
            try:
                result = backend.emit_batch(batch)
            except Exception as exc:
                raise RuntimeError(
                    "backend-routed TeacherTextbook build failed for "
                    f"teacher_backend={config.teacher_backend!r}, "
                    f"runtime_mode={config.runtime_mode!r}, "
                    f"target_policy={config.target_policy!r}; no CPU fallback, "
                    "silent fallback, or backend swap was performed"
                ) from exc
            if first_metadata is None:
                first_metadata = dict(result.metadata)
                target_type = _artifact_target_type(
                    config.target_policy, first_metadata
                )
                _rewrite_metadata_with_backend_params(
                    store,
                    config=config,
                    target_type=target_type,
                    backend_metadata=first_metadata,
                )
            store.write_shard(
                shard_id,
                _arrays_for_store(
                    result.input_ids,
                    result.attention_mask,
                    result.payload,
                ),
            )
    finally:
        backend.close()

    selection_manifest: dict[str, object] | None = None
    if config.exemplar_selection_enabled:
        capture_mode = _effective_capture_mode(config, first_metadata or {})
        fulfillment_policy = _effective_fulfillment_policy(config, capture_mode)
        selection_manifest = build_exemplar_selection_manifest(
            store,
            examples=examples,
            batch_size=config.batch_size,
            capture_mode=capture_mode,
            fulfillment_policy=fulfillment_policy,
            board_capacity=config.exemplar_selection_board_capacity,
            budget_examples=config.exemplar_selection_budget_examples,
            budget_fraction=config.exemplar_selection_budget_fraction,
            created_at=created_at,
        )
        manifest_path = write_exemplar_selection_manifest(
            config.output_dir,
            selection_manifest,
        )
        _rewrite_metadata_with_selection_params(
            store,
            selection_manifest=selection_manifest,
            manifest_path=manifest_path,
        )

    _write_backend_sidecars(
        config,
        examples,
        created_at,
        backend_metadata=first_metadata or {},
        selection_manifest=selection_manifest,
        target_type=target_type,
        shard_count=shard_count,
    )
    report = validate_teacher_textbook(config.output_dir)
    write_teacher_textbook_validation_report(
        report,
        config.output_dir / "validation_report.json",
    )
    if report.status != "pass":
        raise ValueError(
            "built backend TeacherTextbook failed validation: "
            + "; ".join(report.blockers)
        )
    write_cover_page(config.output_dir)
    return validate_teacher_textbook(config.output_dir)


def build_streaming_backend_teacher_textbook(
    config: BackendTeacherTextbookBuildConfig,
) -> Any:
    _validate_streaming_build_config(config)
    output_dir = config.output_dir
    run_manifest_path = _run_manifest_path(config)
    progress_log_path = _progress_log_path(config)
    failure_report_path = output_dir / STREAMING_FAILURE_REPORT_FILENAME
    if output_dir.exists():
        if config.overwrite:
            shutil.rmtree(output_dir)
        elif not config.resume:
            raise ValueError(
                f"TeacherTextbook output already exists: {output_dir}. "
                "Pass overwrite=True to replace it or resume=True to resume."
            )
        elif not run_manifest_path.is_file():
            raise ValueError("cannot resume streaming build without run_manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_tmp_shards(output_dir)

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    dataset_hash = _sha256_file(config.dataset_path)
    corpus_hash = _corpus_hash_for_streaming(config)
    teacher_hashes = _teacher_model_hashes(config)
    num_examples_planned = _count_streaming_examples(
        config.dataset_path,
        max_examples=config.max_examples,
    )
    shard_size = int(config.shard_size_examples or config.batch_size)
    shard_count = _ceil_div(num_examples_planned, shard_size)
    resume_payload = _resume_config_payload(
        config,
        dataset_hash=dataset_hash,
        corpus_hash=corpus_hash,
        teacher_model_hashes=teacher_hashes,
        shard_size_examples=shard_size,
    )
    resume_hash = _hash_json_payload(resume_payload)
    emission_hash = _hash_json_payload(
        {key: resume_payload[key] for key in sorted(resume_payload)}
    )
    start_index = 0
    completed_shards: list[dict[str, object]] = []
    if config.resume and run_manifest_path.is_file():
        manifest = _read_streaming_manifest(run_manifest_path)
        if manifest.get("status") == "complete":
            report = validate_teacher_textbook(output_dir)
            if report.status == "pass":
                _append_progress(
                    progress_log_path,
                    "run_resumed",
                    message="streaming build already complete",
                )
                return report
            raise ValueError("cannot resume complete run because validation failed")
        if manifest.get("resume_config_hash") != resume_hash:
            _write_failure_report(
                failure_report_path,
                failure_stage="resume_config",
                failure_reason="resume_config_hash mismatch; config drift detected",
                exception_type="ConfigDriftError",
                shard_id=None,
                example_start_index=None,
                example_end_index_exclusive=None,
                num_examples_completed=int(manifest.get("num_examples_completed", 0)),
                num_shards_completed=int(manifest.get("num_shards_completed", 0)),
                resume_available=False,
                recommended_action=(
                    "Config changed; rerun without --resume or use --overwrite."
                ),
            )
            raise ValueError("resume_config_hash mismatch; refusing to resume")
        completed_shards = _verified_completed_shards(output_dir, manifest)
        start_index = _next_example_index(completed_shards)
        _append_progress(
            progress_log_path,
            "run_resumed",
            message=f"resuming at example_index={start_index}",
        )
    else:
        _append_progress(progress_log_path, "run_started")

    backend_config = teacher_backend_config_from_build_config(config)
    backend = create_backend(backend_config)
    store: TeacherTargetStore | None = (
        TeacherTargetStore.open(output_dir)
        if (output_dir / "metadata.json").is_file()
        else None
    )
    first_metadata: dict[str, object] | None = None
    target_type = config.target_policy
    manifest = _streaming_manifest(
        config,
        created_at=created_at,
        updated_at=created_at,
        status="running",
        dataset_hash=dataset_hash,
        corpus_hash=corpus_hash,
        teacher_model_hashes=teacher_hashes,
        emission_config_hash=emission_hash,
        resume_config_hash=resume_hash,
        shard_size_examples=shard_size,
        num_examples_planned=num_examples_planned,
        shard_count=shard_count,
        completed_shards=completed_shards,
        failure_report_path=None,
    )
    _write_streaming_manifest(run_manifest_path, manifest)
    try:
        for shard_id, shard_examples in _iter_streaming_shards(
            config.dataset_path,
            shard_size_examples=shard_size,
            max_examples=config.max_examples,
            start_index=start_index,
        ):
            if any(int(item["shard_id"]) == shard_id for item in completed_shards):
                _append_progress(
                    progress_log_path,
                    "shard_skipped_existing",
                    shard_id=shard_id,
                    example_start_index=shard_id * shard_size,
                    example_end_index_exclusive=shard_id * shard_size
                    + len(shard_examples),
                    num_examples=len(shard_examples),
                )
                continue
            shard_start = shard_id * shard_size
            shard_end = shard_start + len(shard_examples)
            _append_progress(
                progress_log_path,
                "shard_started",
                shard_id=shard_id,
                example_start_index=shard_start,
                example_end_index_exclusive=shard_end,
                num_examples=len(shard_examples),
            )
            arrays, metadata = _emit_streaming_shard(
                backend,
                config,
                shard_examples,
            )
            if first_metadata is None:
                first_metadata = metadata
            if store is None:
                target_type = _artifact_target_type(config.target_policy, metadata)
                store = _create_streaming_store(
                    config,
                    created_at=created_at,
                    target_type=target_type,
                    backend_metadata=metadata,
                    num_examples=num_examples_planned,
                    shard_count=shard_count,
                    resume_config_hash=resume_hash,
                    shard_size_examples=shard_size,
                )
            final_path = _write_shard_atomic(output_dir, shard_id, arrays)
            record = {
                "shard_id": shard_id,
                "path": final_path.relative_to(output_dir).as_posix(),
                "example_start_index": shard_start,
                "example_end_index_exclusive": shard_end,
                "num_examples": len(shard_examples),
                "sha256": _sha256_file(final_path),
                "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            }
            completed_shards.append(record)
            manifest = _streaming_manifest(
                config,
                created_at=created_at,
                updated_at=record["completed_at"],
                status="running",
                dataset_hash=dataset_hash,
                corpus_hash=corpus_hash,
                teacher_model_hashes=teacher_hashes,
                emission_config_hash=emission_hash,
                resume_config_hash=resume_hash,
                shard_size_examples=shard_size,
                num_examples_planned=num_examples_planned,
                shard_count=shard_count,
                completed_shards=completed_shards,
                failure_report_path=None,
            )
            _write_streaming_manifest(run_manifest_path, manifest)
            _append_progress(
                progress_log_path,
                "shard_completed",
                shard_id=shard_id,
                example_start_index=shard_start,
                example_end_index_exclusive=shard_end,
                num_examples=len(shard_examples),
            )
    except Exception as exc:
        failed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        _write_failure_report(
            failure_report_path,
            failure_stage="shard_emission",
            failure_reason=str(exc),
            exception_type=type(exc).__name__,
            shard_id=None,
            example_start_index=_next_example_index(completed_shards),
            example_end_index_exclusive=None,
            num_examples_completed=_completed_example_count(completed_shards),
            num_shards_completed=len(completed_shards),
            resume_available=True,
            recommended_action="Fix the failure and rerun with --resume.",
        )
        manifest = _streaming_manifest(
            config,
            created_at=created_at,
            updated_at=failed_at,
            status="failed",
            dataset_hash=dataset_hash,
            corpus_hash=corpus_hash,
            teacher_model_hashes=teacher_hashes,
            emission_config_hash=emission_hash,
            resume_config_hash=resume_hash,
            shard_size_examples=shard_size,
            num_examples_planned=num_examples_planned,
            shard_count=shard_count,
            completed_shards=completed_shards,
            failure_report_path=STREAMING_FAILURE_REPORT_FILENAME,
        )
        _write_streaming_manifest(run_manifest_path, manifest)
        _append_progress(progress_log_path, "run_failed", message=str(exc))
        raise
    finally:
        backend.close()

    if store is None:
        raise ValueError("streaming build produced no examples")
    _rewrite_metadata_with_streaming_completion(
        store,
        resume_config_hash=resume_hash,
        shard_size_examples=shard_size,
        num_examples_completed=_completed_example_count(completed_shards),
    )
    _write_streaming_backend_sidecars(
        config,
        created_at,
        backend_metadata=first_metadata or {},
        target_type=target_type,
        shard_count=shard_count,
        num_examples=num_examples_planned,
        resume_config_hash=resume_hash,
        shard_size_examples=shard_size,
    )
    report = validate_teacher_textbook(output_dir)
    write_teacher_textbook_validation_report(
        report,
        output_dir / "validation_report.json",
    )
    if report.status != "pass":
        failed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        _write_failure_report(
            failure_report_path,
            failure_stage="validation",
            failure_reason="; ".join(report.blockers),
            exception_type="ValidationError",
            shard_id=None,
            example_start_index=None,
            example_end_index_exclusive=None,
            num_examples_completed=_completed_example_count(completed_shards),
            num_shards_completed=len(completed_shards),
            resume_available=True,
            recommended_action="Fix the failure and rerun with --resume.",
        )
        manifest["status"] = "failed"
        manifest["updated_at"] = failed_at
        manifest["failure_report_path"] = STREAMING_FAILURE_REPORT_FILENAME
        _write_streaming_manifest(run_manifest_path, manifest)
        raise ValueError(
            "built streaming backend TeacherTextbook failed validation: "
            + "; ".join(report.blockers)
        )
    write_cover_page(output_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest = _streaming_manifest(
        config,
        created_at=created_at,
        updated_at=completed_at,
        status="complete",
        dataset_hash=dataset_hash,
        corpus_hash=corpus_hash,
        teacher_model_hashes=teacher_hashes,
        emission_config_hash=emission_hash,
        resume_config_hash=resume_hash,
        shard_size_examples=shard_size,
        num_examples_planned=num_examples_planned,
        shard_count=shard_count,
        completed_shards=completed_shards,
        failure_report_path=None,
    )
    _write_streaming_manifest(run_manifest_path, manifest)
    _append_progress(progress_log_path, "run_completed")
    return validate_teacher_textbook(output_dir)


def teacher_backend_config_from_build_config(
    config: BackendTeacherTextbookBuildConfig,
) -> TeacherBackendConfig:
    return TeacherBackendConfig(
        backend_id=config.teacher_backend,
        runtime_mode=config.runtime_mode,  # type: ignore[arg-type]
        target_policy=config.target_policy,  # type: ignore[arg-type]
        model_id=config.teacher_model_id,
        tokenizer_id=config.tokenizer_id,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        exemplar_source_policy=config.exemplar_source_policy,
        exemplar_capture_mode=config.exemplar_capture_mode,
        exemplar_second_pass_source_policy=config.exemplar_second_pass_source_policy,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        fallback_policy=config.fallback_policy,  # type: ignore[arg-type]
        local_files_only=config.local_files_only,
        allow_downloads=config.allow_downloads,
    )


def _validate_backend_build_config(config: BackendTeacherTextbookBuildConfig) -> None:
    if config.teacher_backend == "gpu_torch" and config.runtime_mode != "cpu_gpu":
        raise ValueError("gpu_torch builder routing requires runtime_mode='cpu_gpu'")
    if config.teacher_backend != "gpu_torch" and config.runtime_mode == "cpu_gpu":
        raise ValueError("runtime_mode='cpu_gpu' requires teacher_backend='gpu_torch'")
    if (
        config.exemplar_selector_policy != MULTI_LEADERBOARD_SELECTOR_POLICY
        and config.exemplar_selection_enabled
    ):
        raise ValueError(
            "only multi_leaderboard_exemplar_selector_v1 is supported for "
            "exemplar selection"
        )
    if (
        config.exemplar_selection_enabled
        and config.target_policy != "corridor_exemplar_v1"
    ):
        raise ValueError(
            "exemplar selection requires target_policy='corridor_exemplar_v1'"
        )
    if config.exemplar_selection_board_capacity < 1:
        raise ValueError("exemplar_selection_board_capacity must be positive")
    if config.exemplar_fulfillment_policy not in {
        "auto",
        PATH_A_FULFILLMENT_POLICY,
        PATH_B_FULFILLMENT_POLICY,
    }:
        raise ValueError("unsupported exemplar fulfillment policy")


def _validate_streaming_build_config(
    config: BackendTeacherTextbookBuildConfig,
) -> None:
    if config.dataset_path is None:
        raise ValueError("streaming backend build requires dataset_path")
    if not config.dataset_path.is_file():
        raise ValueError(f"streaming dataset path is missing: {config.dataset_path}")
    if config.batch_size < 1:
        raise ValueError("streaming batch_size must be positive")
    if config.shard_size_examples is not None and config.shard_size_examples < 1:
        raise ValueError("shard_size_examples must be positive")
    if (config.shard_size_examples or config.batch_size) < config.batch_size:
        raise ValueError("shard_size_examples must be >= batch_size")
    if config.exemplar_selection_enabled:
        raise ValueError(
            "streaming backend build does not support corpus-global exemplar "
            "selection yet; rerun without --exemplar-selection-enabled"
        )


def _run_manifest_path(config: BackendTeacherTextbookBuildConfig) -> Path:
    return (
        config.run_manifest_path or config.output_dir / STREAMING_RUN_MANIFEST_FILENAME
    )


def _progress_log_path(config: BackendTeacherTextbookBuildConfig) -> Path:
    return (
        config.progress_log_path or config.output_dir / STREAMING_PROGRESS_LOG_FILENAME
    )


def _iter_corpus_jsonl_examples(
    path: Path,
    *,
    max_examples: int | None,
) -> Iterable[tuple[int, TinyTextExample]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if max_examples is not None and emitted >= max_examples:
                break
            if not line.strip():
                raise ValueError(f"corpus JSONL line {row_index + 1} is blank")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"corpus JSONL line {row_index + 1} must be an object")
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"corpus JSONL line {row_index + 1} must contain string field text"
                )
            example_id = str(payload.get("example_id", f"corpus_{row_index + 1:09d}"))
            yield row_index, TinyTextExample(example_id=example_id, text=text)
            emitted += 1


def _count_streaming_examples(path: Path | None, *, max_examples: int | None) -> int:
    if path is None:
        return 0
    count = 0
    for _row_index, _example in _iter_corpus_jsonl_examples(
        path,
        max_examples=max_examples,
    ):
        count += 1
    return count


def _iter_streaming_shards(
    path: Path | None,
    *,
    shard_size_examples: int,
    max_examples: int | None,
    start_index: int,
) -> Iterable[tuple[int, tuple[TinyTextExample, ...]]]:
    if path is None:
        return
    batch: list[TinyTextExample] = []
    shard_id: int | None = None
    for row_index, example in _iter_corpus_jsonl_examples(
        path,
        max_examples=max_examples,
    ):
        if row_index < start_index:
            continue
        current_shard_id = row_index // shard_size_examples
        if shard_id is None:
            shard_id = current_shard_id
        if current_shard_id != shard_id:
            yield shard_id, tuple(batch)
            batch = []
            shard_id = current_shard_id
        batch.append(example)
    if batch and shard_id is not None:
        yield shard_id, tuple(batch)


def _emit_streaming_shard(
    backend: Any,
    config: BackendTeacherTextbookBuildConfig,
    examples: tuple[TinyTextExample, ...],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    first_metadata: dict[str, object] | None = None
    for start in range(0, len(examples), config.batch_size):
        batch_examples = examples[start : start + config.batch_size]
        batch = TeacherBatchInput(
            example_ids=tuple(example.example_id for example in batch_examples),
            texts=tuple(example.text for example in batch_examples),
        )
        try:
            result = backend.emit_batch(batch)
        except Exception as exc:
            raise RuntimeError(
                "streaming backend TeacherTextbook build failed for "
                f"teacher_backend={config.teacher_backend!r}, "
                f"runtime_mode={config.runtime_mode!r}, "
                f"target_policy={config.target_policy!r}; completed shards remain "
                "available for --resume"
            ) from exc
        if first_metadata is None:
            first_metadata = dict(result.metadata)
        batch_arrays = _arrays_for_store(
            result.input_ids,
            result.attention_mask,
            result.payload,
        )
        for key, value in batch_arrays.items():
            arrays_by_key.setdefault(key, []).append(np.asarray(value))
    return {
        key: np.concatenate(values, axis=0) for key, values in arrays_by_key.items()
    }, first_metadata or {}


def _create_streaming_store(
    config: BackendTeacherTextbookBuildConfig,
    *,
    created_at: str,
    target_type: str,
    backend_metadata: dict[str, object],
    num_examples: int,
    shard_count: int,
    resume_config_hash: str,
    shard_size_examples: int,
) -> TeacherTargetStore:
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id=config.teacher_model_id,
        model_family=config.teacher_backend,
        tokenizer_id=config.tokenizer_id,
        tokenizer_hash=None,
        vocab_size=config.vocab_size,
        target_type=target_type,
        dtype="float32",
        sequence_length=config.sequence_length,
        num_examples=num_examples,
        shard_count=shard_count,
        created_by="radjax_tome.builder.backend_textbook.streaming",
        created_at=created_at,
        source=_metadata_source(config),
        provenance={
            "phase": "radjax-tome-spec-4.6",
            "teacher_backend": config.teacher_backend,
        },
        target_params={
            **_target_params(config, backend_metadata),
            **_streaming_target_params(
                resume_config_hash=resume_config_hash,
                shard_size_examples=shard_size_examples,
                num_examples_completed=0,
            ),
        },
    )
    return TeacherTargetStore.create(config.output_dir, metadata, overwrite=True)


def _write_shard_atomic(
    output_dir: Path,
    shard_id: int,
    arrays: dict[str, np.ndarray],
) -> Path:
    final_path = target_store_shard_path(output_dir, shard_id)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def _remove_stale_tmp_shards(output_dir: Path) -> None:
    shards_dir = output_dir / "shards"
    if not shards_dir.is_dir():
        return
    for tmp_path in shards_dir.glob("*.tmp"):
        tmp_path.unlink()


def _read_streaming_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run_manifest.json must contain an object")
    if payload.get("schema_version") != STREAMING_RUN_MANIFEST_SCHEMA:
        raise ValueError("run_manifest.json schema_version is unsupported")
    return payload


def _write_streaming_manifest(path: Path, manifest: dict[str, object]) -> None:
    write_json(path, manifest)


def _streaming_manifest(
    config: BackendTeacherTextbookBuildConfig,
    *,
    created_at: str,
    updated_at: str,
    status: str,
    dataset_hash: str,
    corpus_hash: str | None,
    teacher_model_hashes: dict[str, object],
    emission_config_hash: str,
    resume_config_hash: str,
    shard_size_examples: int,
    num_examples_planned: int,
    shard_count: int,
    completed_shards: list[dict[str, object]],
    failure_report_path: str | None,
) -> dict[str, object]:
    completed_count = _completed_example_count(completed_shards)
    return {
        "schema_version": STREAMING_RUN_MANIFEST_SCHEMA,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "builder_version": STREAMING_BUILDER_VERSION,
        "output_dir": str(config.output_dir),
        "dataset_path": str(config.dataset_path),
        "dataset_hash": dataset_hash,
        "corpus_manifest_path": _path_or_none(config.corpus_manifest_path),
        "corpus_hash": corpus_hash,
        "teacher_model_provenance_path": _path_or_none(
            config.teacher_model_provenance_path
        ),
        "teacher_model_hashes": teacher_model_hashes,
        "teacher_backend": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "emission_config_hash": emission_config_hash,
        "resume_config_hash": resume_config_hash,
        "shard_size_examples": shard_size_examples,
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "num_examples_planned": num_examples_planned,
        "num_examples_completed": completed_count,
        "num_shards_planned": shard_count,
        "num_shards_completed": len(completed_shards),
        "completed_shards": completed_shards,
        "incomplete_shards": [
            shard_id
            for shard_id in range(shard_count)
            if shard_id not in {int(item["shard_id"]) for item in completed_shards}
        ],
        "last_completed_example_index": _next_example_index(completed_shards) - 1,
        "failure_report_path": failure_report_path,
        "claims_not_made": {
            "no_one_command_production_pipeline": True,
            "no_model_download": True,
            "no_network_verification": True,
            "no_multidevice_scheduling": True,
            "no_tpu_jax": True,
        },
    }


def _append_progress(
    path: Path,
    event: str,
    *,
    shard_id: int | None = None,
    example_start_index: int | None = None,
    example_end_index_exclusive: int | None = None,
    num_examples: int | None = None,
    message: str | None = None,
) -> None:
    payload = {
        "event": event,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    for key, value in (
        ("shard_id", shard_id),
        ("example_start_index", example_start_index),
        ("example_end_index_exclusive", example_end_index_exclusive),
        ("num_examples", num_examples),
        ("message", message),
    ):
        if value is not None:
            payload[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_failure_report(
    path: Path,
    *,
    failure_stage: str,
    failure_reason: str,
    exception_type: str,
    shard_id: int | None,
    example_start_index: int | None,
    example_end_index_exclusive: int | None,
    num_examples_completed: int,
    num_shards_completed: int,
    resume_available: bool,
    recommended_action: str,
) -> None:
    write_json(
        path,
        {
            "schema_version": STREAMING_FAILURE_REPORT_SCHEMA,
            "status": "failed",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "exception_type": exception_type,
            "shard_id": shard_id,
            "example_start_index": example_start_index,
            "example_end_index_exclusive": example_end_index_exclusive,
            "num_examples_completed": num_examples_completed,
            "num_shards_completed": num_shards_completed,
            "resume_available": resume_available,
            "recommended_action": recommended_action,
        },
    )


def _verified_completed_shards(
    output_dir: Path,
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    for item in manifest.get("completed_shards", ()):
        if not isinstance(item, dict):
            raise ValueError("completed_shards entries must be objects")
        shard_path = output_dir / str(item["path"])
        if not shard_path.is_file():
            raise ValueError(f"completed shard is missing: {shard_path}")
        if item.get("sha256") != _sha256_file(shard_path):
            raise ValueError(f"completed shard hash mismatch: {shard_path}")
        completed.append(dict(item))
    return completed


def _next_example_index(completed_shards: list[dict[str, object]]) -> int:
    if not completed_shards:
        return 0
    return max(int(item["example_end_index_exclusive"]) for item in completed_shards)


def _completed_example_count(completed_shards: list[dict[str, object]]) -> int:
    return sum(int(item["num_examples"]) for item in completed_shards)


def _resume_config_payload(
    config: BackendTeacherTextbookBuildConfig,
    *,
    dataset_hash: str,
    corpus_hash: str | None,
    teacher_model_hashes: dict[str, object],
    shard_size_examples: int,
) -> dict[str, object]:
    return {
        "teacher_backend": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "teacher_model_id": config.teacher_model_id,
        "tokenizer_id": config.tokenizer_id,
        "teacher_model_hashes": teacher_model_hashes,
        "dataset_hash": dataset_hash,
        "corpus_hash": corpus_hash,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "batch_size": config.batch_size,
        "shard_size_examples": shard_size_examples,
        "exemplar_source_policy": config.exemplar_source_policy,
        "exemplar_capture_mode": config.exemplar_capture_mode,
        "exemplar_second_pass_source_policy": (
            config.exemplar_second_pass_source_policy
        ),
        "exemplar_selection_enabled": config.exemplar_selection_enabled,
        "exemplar_selector_policy": config.exemplar_selector_policy,
        "exemplar_fulfillment_policy": config.exemplar_fulfillment_policy,
        "gpu_batch_size_mode": config.gpu_batch_size_mode,
        "gpu_batch_size_preset": config.gpu_batch_size_preset,
        "gpu_batch_size_custom": config.gpu_batch_size_custom,
        "gpu_batch_size_auto_min": config.gpu_batch_size_auto_min,
        "gpu_batch_size_auto_max": config.gpu_batch_size_auto_max,
        "local_files_only": config.local_files_only,
        "allow_downloads": config.allow_downloads,
    }


def _hash_json_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path | None) -> str:
    if path is None:
        raise ValueError("cannot hash missing path")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _corpus_hash_for_streaming(
    config: BackendTeacherTextbookBuildConfig,
) -> str | None:
    if config.corpus_manifest_path is None:
        return None
    return _corpus_provenance(config).get("source_corpus_hash")


def _teacher_model_hashes(
    config: BackendTeacherTextbookBuildConfig,
) -> dict[str, object]:
    provenance = _teacher_model_provenance(config)
    return {
        key: provenance.get(key)
        for key in (
            "config_hash",
            "tokenizer_hash",
            "weights_hash",
            "model_directory_hash",
        )
        if provenance.get(key) is not None
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _streaming_target_params(
    *,
    resume_config_hash: str,
    shard_size_examples: int,
    num_examples_completed: int,
) -> dict[str, str]:
    return {
        "streaming_build": "true",
        "resume_supported": "true",
        "run_manifest_path": STREAMING_RUN_MANIFEST_FILENAME,
        "progress_log_path": STREAMING_PROGRESS_LOG_FILENAME,
        "shard_size_examples": str(shard_size_examples),
        "num_examples_completed": str(num_examples_completed),
        "resume_config_hash": resume_config_hash,
        "atomic_shard_write_policy": "tmp_file_fsync_rename_v1",
    }


def _write_streaming_backend_sidecars(
    config: BackendTeacherTextbookBuildConfig,
    created_at: str,
    *,
    backend_metadata: dict[str, object],
    target_type: str,
    shard_count: int,
    num_examples: int,
    resume_config_hash: str,
    shard_size_examples: int,
) -> None:
    corpus_provenance = _corpus_provenance(config)
    teacher_model_provenance = _teacher_model_provenance(config)
    write_json(
        config.output_dir / "vocab_contract.json",
        {
            "tokenizer_id": config.tokenizer_id,
            "tokenizer_hash": None,
            "vocab_size": config.vocab_size,
            "model_id": config.teacher_model_id,
            "model_family": config.teacher_backend,
        },
    )
    streaming_fields = {
        "streaming_build": True,
        "resume_supported": True,
        "run_manifest_path": STREAMING_RUN_MANIFEST_FILENAME,
        "progress_log_path": STREAMING_PROGRESS_LOG_FILENAME,
        "shard_size_examples": shard_size_examples,
        "resume_config_hash": resume_config_hash,
        "atomic_shard_write_policy": "tmp_file_fsync_rename_v1",
    }
    write_json(
        config.output_dir / "teacher_manifest.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": TEACHER_TEXTBOOK_VERSION,
            "teacher_model_id": config.teacher_model_id,
            "teacher_backend_type": config.teacher_backend,
            "teacher_revision_or_hash": None,
            "tokenizer_id": config.tokenizer_id,
            "vocab_size": config.vocab_size,
            "vocab_contract_path": "vocab_contract.json",
            "target_type": target_type,
            "dtype": "float32",
            "sequence_length": config.sequence_length,
            "num_examples": num_examples,
            "shard_count": shard_count,
            "created_at": created_at,
            "local_files_only": config.local_files_only,
            "allow_downloads": config.allow_downloads,
            "corpus_provenance": corpus_provenance or None,
            "teacher_model_provenance": teacher_model_provenance or None,
            "claims_not_made": (
                "no_silent_cpu_fallback",
                "no_historical_parity_claim",
                "no_production_global_two_pass_selector",
                "no_real_auto_batch_probing",
                "no_multidevice_scheduling",
                "no_tpu_jax",
            ),
            **streaming_fields,
        },
    )
    write_json(
        config.output_dir / "emission_config.json",
        {
            "dataset_source": _dataset_source(config.dataset_path),
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "logits_dtype": "float32",
            "include_hidden_states": False,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": config.top_k,
            "seed": 0,
            "teacher_mode": "backend_contract",
            "teacher_backend": config.teacher_backend,
            "runtime_mode": config.runtime_mode,
            "target_policy": config.target_policy,
            "backend_metadata": backend_metadata,
            "exemplar_selection_enabled": config.exemplar_selection_enabled,
            "exemplar_selector_policy": config.exemplar_selector_policy,
            "exemplar_selection_manifest": None,
            "corpus_provenance": corpus_provenance or None,
            "teacher_model_provenance": teacher_model_provenance or None,
            **streaming_fields,
        },
    )


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _arrays_for_store(
    input_ids: Any,
    attention_mask: Any,
    payload: Any,
) -> dict[str, np.ndarray]:
    arrays = {
        "input_ids": np.asarray(input_ids),
        "attention_mask": np.asarray(attention_mask),
    }
    for key, value in dict(payload).items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
    return arrays


def _artifact_target_type(
    target_policy: str,
    backend_metadata: dict[str, object] | None,
) -> str:
    if (
        target_policy == "corridor_exemplar_v1"
        and backend_metadata is not None
        and backend_metadata.get("exemplar_capture_stage") == "score_pass"
    ):
        return "corridor_exemplar_score_pass_v1"
    return target_policy


def _rewrite_metadata_with_backend_params(
    store: TeacherTargetStore,
    *,
    config: BackendTeacherTextbookBuildConfig,
    target_type: str,
    backend_metadata: dict[str, object],
) -> None:
    metadata = TargetStoreMetadata(
        **{
            **asdict(store.metadata),
            "target_type": target_type,
            "target_params": _target_params(config, backend_metadata),
        }
    )
    metadata_file = store.root / "metadata.json"
    metadata_file.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(store, "metadata", metadata)


def _rewrite_metadata_with_selection_params(
    store: TeacherTargetStore,
    *,
    selection_manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    metadata = TargetStoreMetadata(
        **{
            **asdict(store.metadata),
            "target_params": {
                **store.metadata.target_params,
                **_selection_target_params(selection_manifest, manifest_path),
            },
        }
    )
    metadata_file = store.root / "metadata.json"
    metadata_file.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(store, "metadata", metadata)


def _rewrite_metadata_with_streaming_completion(
    store: TeacherTargetStore,
    *,
    resume_config_hash: str,
    shard_size_examples: int,
    num_examples_completed: int,
) -> None:
    metadata = TargetStoreMetadata(
        **{
            **asdict(store.metadata),
            "target_params": {
                **store.metadata.target_params,
                **_streaming_target_params(
                    resume_config_hash=resume_config_hash,
                    shard_size_examples=shard_size_examples,
                    num_examples_completed=num_examples_completed,
                ),
            },
        }
    )
    metadata_file = store.root / "metadata.json"
    metadata_file.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(store, "metadata", metadata)


def _target_params(
    config: BackendTeacherTextbookBuildConfig,
    backend_metadata: dict[str, object],
) -> dict[str, str]:
    params: dict[str, str] = {
        "target_policy": config.target_policy,
        "backend_id": config.teacher_backend,
        "requested_backend_id": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "artifact_emission_path": "teacher_backend_contract",
        "student_consumption_ready": "false",
        "experimental_target_schema": "true",
        "production_global_selector": "false",
        **_corpus_provenance(config),
        **teacher_model_target_params(config.teacher_model_provenance_path),
    }
    for key, value in backend_metadata.items():
        params[_target_param_key(key)] = _stringify_metadata_value(value)
    return params


def _selection_target_params(
    manifest: dict[str, object],
    manifest_path: Path,
) -> dict[str, str]:
    params = {
        "exemplar_selector_policy": str(manifest["selection_policy"]),
        "exemplar_selection_enabled": "true",
        "exemplar_selection_manifest_path": manifest_path.name,
        "exemplar_selection_manifest_schema": str(manifest["schema_version"]),
        "exemplar_fulfillment_policy": str(manifest["fulfillment_policy"]),
        "selection_application": str(manifest["selection_application"]),
        "deduplication_policy": str(manifest["deduplication_policy"]),
        "duplicate_candidate_count": _stringify_metadata_value(
            manifest["duplicate_candidate_count"]
        ),
        "backfill_success_count": _stringify_metadata_value(
            manifest["backfill_success_count"]
        ),
        "score_aware_budget_trimming": _stringify_metadata_value(
            manifest["score_aware_budget_trimming"]
        ),
        "budget_trimming_policy": str(manifest["budget_trimming_policy"]),
        "budget_applied": _stringify_metadata_value(manifest["budget_applied"]),
        "num_candidates_seen": _stringify_metadata_value(
            manifest["num_candidates_seen"]
        ),
        "num_unique_examples_selected": _stringify_metadata_value(
            manifest["num_unique_examples_selected"]
        ),
        "num_unique_positions_selected": _stringify_metadata_value(
            manifest["num_unique_positions_selected"]
        ),
        "production_global_selector": _stringify_metadata_value(
            manifest["production_global_selector"]
        ),
        "semantic_diversity_used": _stringify_metadata_value(
            manifest["semantic_diversity_used"]
        ),
        "utility_calibrated": _stringify_metadata_value(manifest["utility_calibrated"]),
        "retention_policy": str(manifest["retention_policy"]),
    }
    if manifest["fulfillment_policy"] == PATH_B_FULFILLMENT_POLICY:
        params.update(
            {
                "rerun_manifest_ready": "true",
                "selected_pass_rerun_performed": "false",
            }
        )
    if manifest["fulfillment_policy"] == PATH_A_FULFILLMENT_POLICY:
        params["selected_from_existing_capture"] = "true"
    return params


def _target_param_key(key: str) -> str:
    if key == "backend_id":
        return "effective_backend_id"
    return key


def _stringify_metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def _write_backend_sidecars(
    config: BackendTeacherTextbookBuildConfig,
    examples: tuple[TinyTextExample, ...],
    created_at: str,
    *,
    backend_metadata: dict[str, object],
    selection_manifest: dict[str, object] | None,
    target_type: str,
    shard_count: int,
) -> None:
    corpus_provenance = _corpus_provenance(config)
    teacher_model_provenance = _teacher_model_provenance(config)
    write_json(
        config.output_dir / "vocab_contract.json",
        {
            "tokenizer_id": config.tokenizer_id,
            "tokenizer_hash": None,
            "vocab_size": config.vocab_size,
            "model_id": config.teacher_model_id,
            "model_family": config.teacher_backend,
        },
    )
    write_json(
        config.output_dir / "teacher_manifest.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": TEACHER_TEXTBOOK_VERSION,
            "teacher_model_id": config.teacher_model_id,
            "teacher_backend_type": config.teacher_backend,
            "teacher_revision_or_hash": None,
            "tokenizer_id": config.tokenizer_id,
            "vocab_size": config.vocab_size,
            "vocab_contract_path": "vocab_contract.json",
            "target_type": target_type,
            "dtype": "float32",
            "sequence_length": config.sequence_length,
            "num_examples": len(examples),
            "shard_count": shard_count,
            "created_at": created_at,
            "local_files_only": config.local_files_only,
            "allow_downloads": config.allow_downloads,
            "corpus_provenance": corpus_provenance or None,
            "teacher_model_provenance": teacher_model_provenance or None,
            "claims_not_made": (
                "no_silent_cpu_fallback",
                "no_historical_parity_claim",
                "no_production_global_two_pass_selector",
                "no_real_auto_batch_probing",
                "no_multidevice_scheduling",
                "no_tpu_jax",
            ),
        },
    )
    write_json(
        config.output_dir / "emission_config.json",
        {
            "dataset_source": _dataset_source(config.dataset_path),
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "logits_dtype": "float32",
            "include_hidden_states": False,
            "sampling_used": False,
            "temperature": None,
            "top_p": None,
            "top_k": config.top_k,
            "seed": 0,
            "teacher_mode": "backend_contract",
            "teacher_backend": config.teacher_backend,
            "runtime_mode": config.runtime_mode,
            "target_policy": config.target_policy,
            "backend_metadata": backend_metadata,
            "exemplar_selection_enabled": config.exemplar_selection_enabled,
            "exemplar_selector_policy": config.exemplar_selector_policy,
            "exemplar_selection_manifest": (
                EXEMPLAR_SELECTION_MANIFEST_FILENAME
                if selection_manifest is not None
                else None
            ),
            "corpus_provenance": corpus_provenance or None,
            "teacher_model_provenance": teacher_model_provenance or None,
        },
    )


def _dataset_source(path: Path | None) -> str:
    return "builtin_examples" if path is None else str(path)


def _metadata_source(config: BackendTeacherTextbookBuildConfig) -> dict[str, str]:
    return {"kind": _dataset_source(config.dataset_path), **_corpus_provenance(config)}


def _corpus_provenance(config: BackendTeacherTextbookBuildConfig) -> dict[str, str]:
    return stringify_corpus_provenance(config.corpus_manifest_path)


def _teacher_model_provenance(
    config: BackendTeacherTextbookBuildConfig,
) -> dict[str, object]:
    if config.teacher_model_provenance_path is None:
        return {}
    return teacher_model_provenance_summary(config.teacher_model_provenance_path)


def _effective_capture_mode(
    config: BackendTeacherTextbookBuildConfig,
    backend_metadata: dict[str, object],
) -> str:
    value = backend_metadata.get("exemplar_capture_mode_effective")
    if isinstance(value, str):
        return value
    if config.exemplar_capture_mode == "auto":
        return "two_pass_sparse_exemplar"
    return config.exemplar_capture_mode


def _effective_fulfillment_policy(
    config: BackendTeacherTextbookBuildConfig,
    capture_mode: str,
) -> str:
    if config.exemplar_fulfillment_policy != "auto":
        return config.exemplar_fulfillment_policy
    if capture_mode == "two_pass_sparse_exemplar":
        return PATH_B_FULFILLMENT_POLICY
    if capture_mode == "one_pass_candidate":
        return PATH_A_FULFILLMENT_POLICY
    if capture_mode == "auto":
        return PATH_B_FULFILLMENT_POLICY
    raise ValueError(f"unsupported exemplar capture mode: {capture_mode}")
