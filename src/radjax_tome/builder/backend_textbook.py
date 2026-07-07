from __future__ import annotations

import json
import shutil
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
from radjax_tome.targets.store import TeacherTargetStore
from radjax_tome.tome import write_cover_page


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


def build_backend_teacher_textbook(
    config: BackendTeacherTextbookBuildConfig,
) -> Any:
    _validate_backend_build_config(config)
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
