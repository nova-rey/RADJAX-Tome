from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.backends import GPUTorchTeacherEmissionBackend
from radjax_tome.builder import (
    BackendTeacherTextbookBuildConfig,
    build_backend_teacher_textbook,
    teacher_backend_config_from_build_config,
    validate_teacher_textbook,
)
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    validate_target_store_metadata,
)
from radjax_tome.targets.store import TeacherTargetStore


def _config(tmp_path: Path, **overrides: object) -> BackendTeacherTextbookBuildConfig:
    payload = {
        "output_dir": tmp_path / "backend_tome",
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dynamic_cascaded_soft_labels_v1",
        "sequence_length": 5,
        "batch_size": 2,
        "max_examples": 2,
        "vocab_size": 11,
        "top_k": 4,
        "num_buckets": 3,
        "overwrite": True,
    }
    payload.update(overrides)
    return BackendTeacherTextbookBuildConfig(**payload)


def test_builder_config_can_request_gpu_torch_runtime() -> None:
    config = BackendTeacherTextbookBuildConfig(
        output_dir=Path("unused"),
        teacher_backend="gpu_torch",
        runtime_mode="cpu_gpu",
        target_policy="dynamic_cascaded_soft_labels_v1",
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=8,
    )

    backend_config = teacher_backend_config_from_build_config(config)

    assert backend_config.backend_id == "gpu_torch"
    assert backend_config.runtime_mode == "cpu_gpu"
    assert backend_config.target_policy == "dynamic_cascaded_soft_labels_v1"
    assert backend_config.gpu_batch_size_mode == "preset"
    assert backend_config.gpu_batch_size_preset == 8


def test_gpu_torch_builder_routing_rejects_cpu_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpu_torch.*runtime_mode"):
        build_backend_teacher_textbook(
            _config(
                tmp_path,
                teacher_backend="gpu_torch",
                runtime_mode="cpu",
            )
        )


def test_gpu_torch_unavailable_fails_without_silent_cpu_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        build_backend_teacher_textbook(
            _config(
                tmp_path,
                teacher_backend="gpu_torch",
                runtime_mode="cpu_gpu",
                target_policy="dynamic_cascaded_soft_labels_v1",
                fallback_policy="auto",
            )
        )

    message = str(exc_info.value)
    assert "gpu_torch" in message
    assert "fallback" in message
    assert "no CPU fallback" in message


def test_unsupported_backend_target_combination_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hf_torch supports"):
        build_backend_teacher_textbook(
            _config(
                tmp_path,
                teacher_backend="hf_torch",
                runtime_mode="cpu",
                target_policy="dynamic_cascaded_soft_labels_v1",
            )
        )


@pytest.mark.parametrize(
    "target_type",
    (
        "dynamic_cascaded_soft_labels_v1",
        "corridor_exemplar_v1",
        "corridor_exemplar_score_pass_v1",
    ),
)
def test_target_store_schema_recognizes_backend_target_types(
    target_type: str,
) -> None:
    validate_target_store_metadata(
        TargetStoreMetadata(
            schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
            target_store_version=TEACHER_TARGET_STORE_VERSION,
            model_id="model",
            model_family="test",
            tokenizer_id="tok",
            tokenizer_hash=None,
            vocab_size=11,
            target_type=target_type,
            dtype="float32",
            sequence_length=5,
            num_examples=2,
            shard_count=1,
            created_by="test",
            created_at="2026-07-02T00:00:00+00:00",
        )
    )


def test_backend_builder_writes_dynamic_cascaded_artifact(tmp_path: Path) -> None:
    output = tmp_path / "dynamic"
    report = build_backend_teacher_textbook(_config(tmp_path, output_dir=output))

    assert report.status == "pass"
    store = TeacherTargetStore.open(output)
    metadata = store.metadata
    params = metadata.target_params
    cover_page = json.loads((output / "cover_page.json").read_text(encoding="utf-8"))

    assert metadata.target_type == "dynamic_cascaded_soft_labels_v1"
    assert validate_teacher_textbook(output).status == "pass"
    assert params["target_policy"] == "dynamic_cascaded_soft_labels_v1"
    assert params["requested_backend_id"] == "cpu_reference"
    assert params["effective_backend_id"] == "cpu_reference"
    assert params["requested_runtime_mode"] == "cpu"
    assert params["effective_runtime_mode"] == "cpu"
    assert params["fallback_policy"] == "error"
    assert params["backend_family"] == "cpu_reference"
    assert params["artifact_emission_path"] == "teacher_backend_contract"
    assert params["student_consumption_ready"] == "false"
    assert params["experimental_target_schema"] == "true"
    assert params["gpu_batch_size_policy"] == "gpu_batch_size_policy_v1"
    assert params["effective_gpu_batch_size"] == "8"
    assert params["measured_output_bytes_available"] == "true"
    assert params["optimized_path_used"] == "false"
    assert cover_page["targets"]["target_params"]["effective_backend_id"] == (
        "cpu_reference"
    )

    shard = store.read_shard(0)
    assert shard["input_ids"].shape == (2, 5)
    assert shard["top_token_ids"].shape[0] == 2
    assert shard["top_selection_mask"].shape[0] == 2


def test_backend_builder_writes_one_pass_corridor_artifact(tmp_path: Path) -> None:
    output = tmp_path / "corridor"
    report = build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="one_pass_candidate",
        )
    )

    assert report.status == "pass"
    store = TeacherTargetStore.open(output)
    params = store.metadata.target_params

    assert store.metadata.target_type == "corridor_exemplar_v1"
    assert params["exemplar_capture_mode_requested"] == "one_pass_candidate"
    assert params["exemplar_capture_mode_effective"] == "one_pass_candidate"
    assert params["requires_second_pass_for_final_exemplars"] == "false"
    assert params["production_global_selector"] == "false"
    shard = store.read_shard(0)
    assert shard["corridor_teacher_entropy"].shape == (2, 5)
    assert shard["exemplar_positions"].shape[0] == 2


def test_backend_builder_writes_two_pass_score_artifact_truthfully(
    tmp_path: Path,
) -> None:
    output = tmp_path / "score"
    report = build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="two_pass_sparse_exemplar",
        )
    )

    assert report.status == "pass"
    store = TeacherTargetStore.open(output)
    params = store.metadata.target_params

    assert store.metadata.target_type == "corridor_exemplar_score_pass_v1"
    assert params["exemplar_capture_stage"] == "score_pass"
    assert params["exemplar_candidate_scope"] == "batch_score_and_corridor_evidence"
    assert params["corpus_level_exemplar_finalization"] == "false"
    assert params["requires_second_pass_for_final_exemplars"] == "true"
    assert params["production_global_selector"] == "false"
    shard = store.read_shard(0)
    assert shard["score_max_entropy"].shape == (2,)
    assert shard["corridor_teacher_entropy"].shape == (2, 5)
    assert shard["corridor_top_token_ids"].shape == (2, 5)
    assert "exemplar_positions" not in shard


def test_backend_builder_preserves_batch_size_and_no_split_semantics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"
    build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            batch_size=3,
            max_examples=4,
            gpu_batch_size_mode="custom",
            gpu_batch_size_custom=3,
        )
    )
    store = TeacherTargetStore.open(output)
    first = store.read_shard(0)
    second = store.read_shard(1)
    params = store.metadata.target_params

    assert first["input_ids"].shape[0] == 3
    assert second["input_ids"].shape[0] == 1
    assert params["configured_batch_size"] == "3"
    assert params["actual_batch_size"] == "3"
    assert params["effective_gpu_batch_size"] == "3"


def test_gpu_torch_metadata_can_flow_through_fake_test_seam() -> None:
    backend = GPUTorchTeacherEmissionBackend(
        teacher_backend_config_from_build_config(
            BackendTeacherTextbookBuildConfig(
                output_dir=Path("unused"),
                teacher_backend="gpu_torch",
                runtime_mode="cpu_gpu",
                target_policy="corridor_exemplar_v1",
                exemplar_capture_mode="one_pass_candidate",
            )
        )
    )
    metadata = backend.metadata(actual_batch_size=2, effective_vocab_size=11)

    assert metadata["requested_runtime_mode"] == "cpu_gpu"
    assert metadata["backend_id"] == "gpu_torch"
    assert metadata["fallback_used"] is False
    assert metadata["fallback_handled_by"] == "none"
    assert metadata["gpu_batch_size_policy"] == "gpu_batch_size_policy_v1"
    assert metadata["exemplar_capture_mode_requested"] == "one_pass_candidate"
