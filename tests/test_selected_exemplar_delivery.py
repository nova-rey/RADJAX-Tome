from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import radjax_tome.builder.exemplar_delivery as exemplar_delivery
from radjax_tome.builder import (
    ProductionBuildConfig,
    build_production_gpu_tome,
    compare_exemplar_delivery_artifacts,
    validate_teacher_textbook,
)
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import write_json
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_model_dir(root: Path) -> Path:
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/local-test", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    return model


def _production_inputs(
    tmp_path: Path,
    *,
    example_count: int = 5,
) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(example_count):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"selected delivery example {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus_out"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = _fake_model_dir(tmp_path)
    provenance_path = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/local-test"),
        provenance_path,
    )
    return (
        corpus_dir / "corpus.jsonl",
        corpus_dir / "corpus_manifest.json",
        provenance_path,
        model,
    )


def _config(
    tmp_path: Path,
    *,
    output_name: str,
    delivery_path: str,
    retain_unselected: bool = False,
    example_count: int = 5,
    max_examples: int | None = 5,
    gpu_batch_size_preset: int = 2,
    shard_size_examples: int = 2,
    track_delivery_timing: bool = False,
) -> ProductionBuildConfig:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(
        tmp_path,
        example_count=example_count,
    )
    return ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id=str(model),
        dataset_path=dataset,
        corpus_manifest_path=corpus_manifest,
        teacher_model_provenance_path=provenance_path,
        output_dir=tmp_path / output_name,
        teacher_backend="cpu_reference",
        runtime_mode="cpu",
        target_policy="corridor_exemplar_v1",
        sequence_length=5,
        vocab_size=13,
        top_k=4,
        num_buckets=3,
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=gpu_batch_size_preset,
        shard_size_examples=shard_size_examples,
        max_examples=max_examples,
        exemplar_selection_enabled=True,
        exemplar_delivery_path=delivery_path,
        selected_exemplar_budget=2,
        retain_unselected_exemplar_payloads=retain_unselected,
        track_delivery_timing=track_delivery_timing,
    )


def test_cli_exposes_selected_only_exemplar_delivery_flags() -> None:
    production_help = run_cli(ROOT, "production-build", "--help")
    parity_help = run_cli(ROOT, "exemplar-delivery-parity", "--help")

    assert production_help.returncode == 0, production_help.stderr
    assert "--exemplar-delivery-path" in production_help.stdout
    assert "--exemplar-selection-enabled" in production_help.stdout
    assert "--selected-exemplar-budget" in production_help.stdout
    assert "--no-retain-unselected-exemplar-payloads" in production_help.stdout
    assert "--exemplar-score-policy" in production_help.stdout
    assert "--track-delivery-timing" in production_help.stdout
    assert parity_help.returncode == 0, parity_help.stderr
    assert "--path-a" in parity_help.stdout
    assert "--path-b" in parity_help.stdout


def test_path_b_selected_only_delivery_writes_payloads_without_unselected_retention(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
    )

    report = build_production_gpu_tome(config)
    output = config.output_dir
    delivery = _json(output / "delivery_report.json")
    selected = _json(output / "leaderboards" / "selected_exemplars.json")
    payload = _json(output / "selected_exemplars" / "selected-exemplars-00000.json")

    assert report["status"] == "pass"
    assert report["delivery_path"] == "two_pass_rerun_selected"
    assert report["num_examples_scored"] == 5
    assert report["num_selected_exemplars"] == 2
    assert report["non_selected_exemplar_payload_retained"] is False
    assert delivery["num_positions_scored"] == 25
    assert delivery["teacher_rerun_count"] == delivery["selected_example_count"]
    assert "timing_enabled" not in delivery
    assert "timing_enabled" not in report
    assert selected["selected_exemplars"]
    selected_payload = payload["selected_exemplars"][0]
    for field in (
        "selected_example_id",
        "selected_position",
        "selected_score",
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "bucket_masses",
        "dynamic_top_k",
    ):
        assert field in selected_payload
    assert not (output / "unselected_candidate_payloads").exists()
    assert validate_teacher_textbook(output).status == "pass"


def test_path_b_main_artifact_is_score_pass_without_broad_exemplar_payloads(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_b_score_pass",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    metadata = _json(config.output_dir / "metadata.json")
    shard_path = config.output_dir / "shards" / "shard-00000.npz"

    assert metadata["target_type"] == "corridor_exemplar_score_pass_v1"
    with np.load(shard_path) as shard:
        assert "score_selected_position" in shard.files
        assert "score_selected_position_entropy" in shard.files
        assert "exemplar_positions" not in shard.files
        assert "exemplar_selection_mask" not in shard.files
        assert "exemplar_source_top_mass" not in shard.files


def test_path_a_selected_only_delivery_prunes_temporary_candidates(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
    )

    report = build_production_gpu_tome(config)
    delivery = _json(config.output_dir / "delivery_report.json")

    assert report["status"] == "pass"
    assert delivery["delivery_path"] == "one_pass_pruned_candidate"
    assert delivery["teacher_rerun_count"] == 0
    assert delivery["selected_payload_source"] == "one_pass_candidate_shard_capture"
    assert delivery["pruned_candidate_payload_bytes"] > 0
    assert delivery["temporary_candidate_bytes"] > 0
    assert delivery["non_selected_exemplar_payload_retained"] is False
    with np.load(config.output_dir / "shards" / "shard-00000.npz") as shard:
        assert not any(key.startswith("exemplar_source_") for key in shard.files)
    assert not (config.output_dir / "temporary_candidates").exists()
    assert not (config.output_dir / "unselected_candidate_payloads").exists()
    assert validate_teacher_textbook(config.output_dir).status == "pass"


def test_path_a_materializes_payloads_from_capture_without_backend_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_create_backend(_backend_config):
        raise AssertionError("Path A must not rerun the backend")

    monkeypatch.setattr(exemplar_delivery, "create_backend", fail_create_backend)
    config = _config(
        tmp_path,
        output_name="path_a_no_rerun",
        delivery_path="one_pass_pruned_candidate",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    payload = _json(
        config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )
    selected_payload = payload["selected_exemplars"][0]

    assert selected_payload["dynamic_top_k"]["source_payload"] == (
        "one_pass_candidate_shard"
    )
    assert len(selected_payload["top_token_ids"]) == config.vocab_size
    assert len(selected_payload["bucket_masses"]) == config.num_buckets


def test_path_a_path_b_delivery_parity_matches_selection_and_scores(
    tmp_path: Path,
) -> None:
    path_a = _config(
        tmp_path / "a",
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
    )
    path_b = _config(
        tmp_path / "b",
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(path_a)["status"] == "pass"
    assert build_production_gpu_tome(path_b)["status"] == "pass"

    report = compare_exemplar_delivery_artifacts(
        path_a.output_dir,
        path_b.output_dir,
        output=tmp_path / "parity_report.json",
    )

    assert report["status"] == "pass"
    assert report["selected_example_ids_match"] is True
    assert report["selected_positions_match"] is True
    assert report["selected_score_ranks_match"] is True
    assert report["path_a_teacher_rerun_count"] == 0
    assert (
        report["path_b_teacher_rerun_count"]
        == _json(path_b.output_dir / "delivery_report.json")["selected_example_count"]
    )


def test_path_a_path_b_1000_example_selected_only_parity_and_pruning(
    tmp_path: Path,
) -> None:
    path_a = _config(
        tmp_path / "a1000",
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
        example_count=1000,
        max_examples=1000,
        gpu_batch_size_preset=64,
        shard_size_examples=250,
        track_delivery_timing=True,
    )
    path_b = _config(
        tmp_path / "b1000",
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
        example_count=1000,
        max_examples=1000,
        gpu_batch_size_preset=64,
        shard_size_examples=250,
        track_delivery_timing=True,
    )

    assert build_production_gpu_tome(path_a)["status"] == "pass"
    assert build_production_gpu_tome(path_b)["status"] == "pass"
    report = compare_exemplar_delivery_artifacts(
        path_a.output_dir,
        path_b.output_dir,
        output=tmp_path / "parity_1000.json",
    )
    path_a_delivery = _json(path_a.output_dir / "delivery_report.json")
    path_b_delivery = _json(path_b.output_dir / "delivery_report.json")
    path_a_production = _json(path_a.output_dir / "production_build_report.json")
    path_b_production = _json(path_b.output_dir / "production_build_report.json")

    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["selected_example_ids_match"] is True
    assert report["selected_positions_match"] is True
    assert report["selected_score_ranks_match"] is True
    assert report["selected_mode_keys_match"] is True
    assert report["payload_shape_compatible"] is True
    assert report["corridor_artifact_shape_match"] is True
    assert report["path_a_corridor_mode_count"] >= 1
    assert report["path_b_corridor_mode_count"] >= 1
    assert report["timing_enabled"] is True
    assert report["path_a_wall_seconds"] >= 0.0
    assert report["path_b_wall_seconds"] >= 0.0
    assert report["faster_path"] in {"path_a", "path_b", "tie", "unknown"}
    assert report["timing_claims_not_made"]["no_speed_parity_requirement"] is True
    assert path_a_delivery["teacher_rerun_count"] == 0
    assert (
        path_b_delivery["teacher_rerun_count"]
        == path_b_delivery["selected_example_count"]
    )
    assert path_a_delivery["timing_enabled"] is True
    assert path_b_delivery["timing_enabled"] is True
    assert path_a_delivery["path_a_wall_seconds"] >= 0.0
    assert path_b_delivery["path_b_wall_seconds"] >= 0.0
    assert path_a_production["timing_enabled"] is True
    assert path_b_production["timing_enabled"] is True
    assert path_a_production["production_wall_seconds"] >= 0.0
    assert path_b_production["production_wall_seconds"] >= 0.0
    assert path_a_production["score_or_main_pass_wall_seconds"] >= 0.0
    assert path_b_production["score_or_main_pass_wall_seconds"] >= 0.0
    assert path_a_delivery["teacher_rerun_count"] == 0
    assert (
        path_b_delivery["teacher_rerun_count"]
        == path_b_delivery["selected_example_count"]
    )
    for shard_path in sorted((path_a.output_dir / "shards").glob("shard-*.npz")):
        with np.load(shard_path) as shard:
            retained = [
                key for key in shard.files if key.startswith("exemplar_source_")
            ]
        assert retained == []


def test_path_b_selected_rerun_invokes_backend_only_for_selected_examples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invocations: list[tuple[str, ...]] = []
    backend_configs: list[object] = []

    class MarkerBackend:
        def __init__(self, config):
            self.config = config
            backend_configs.append(config)

        def emit_batch(self, batch):
            invocations.append(tuple(batch.example_ids))
            return SimpleNamespace(payload=_marker_dynamic_payload(batch, self.config))

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        exemplar_delivery,
        "create_backend",
        lambda backend_config: MarkerBackend(backend_config),
    )
    config = _config(
        tmp_path,
        output_name="path_b_backend_marker",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery = _json(config.output_dir / "delivery_report.json")

    assert len(invocations) == 1
    assert invocations[0] == tuple(delivery["selected_rerun_example_ids"])
    assert len(invocations[0]) == delivery["teacher_rerun_count"]
    assert backend_configs[0].target_policy == "dynamic_cascaded_soft_labels_v1"


def test_selected_payload_values_come_from_backend_emission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class MarkerBackend:
        def __init__(self, config):
            self.config = config

        def emit_batch(self, batch):
            return SimpleNamespace(payload=_marker_dynamic_payload(batch, self.config))

    monkeypatch.setattr(
        exemplar_delivery,
        "create_backend",
        lambda backend_config: MarkerBackend(backend_config),
    )
    config = _config(
        tmp_path,
        output_name="path_b_marker_payload",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery = _json(config.output_dir / "delivery_report.json")
    payload = _json(
        config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )
    selected_payload = payload["selected_exemplars"][0]
    row = delivery["selected_rerun_example_ids"].index(
        selected_payload["selected_example_id"]
    )
    position = selected_payload["selected_position"]

    assert selected_payload["top_token_ids"][:3] == [
        _marker_token(row, position, index) for index in range(3)
    ]
    assert selected_payload["top_probs"][:3] == [
        _marker_prob(row, position, index) for index in range(3)
    ]
    assert selected_payload["top_mass"] == _marker_top_mass(row, position)
    assert selected_payload["bucket_masses"] == [
        _marker_bucket(row, position, index) for index in range(config.num_buckets)
    ]


def test_validation_fails_when_non_selected_payload_retention_is_claimed(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="retained",
        delivery_path="one_pass_pruned_candidate",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery_path = config.output_dir / "delivery_report.json"
    delivery = _json(delivery_path)
    delivery["non_selected_exemplar_payload_retained"] = True
    write_json(delivery_path, delivery)

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert "non_selected_exemplar_payload_retained=true" in validation.blockers


def _marker_dynamic_payload(batch, config) -> dict[str, np.ndarray]:
    batch_size = len(batch.example_ids)
    sequence_length = config.sequence_length
    top_k = min(5, config.vocab_size)
    shape = (batch_size, sequence_length, top_k)
    top_token_ids = np.zeros(shape, dtype=np.int32)
    top_probs = np.zeros(shape, dtype=np.float32)
    top_log_probs = np.zeros(shape, dtype=np.float32)
    top_selection_mask = np.ones(shape, dtype=bool)
    bucket_masses = np.zeros(
        (batch_size, sequence_length, config.num_buckets),
        dtype=np.float32,
    )
    top_mass = np.zeros((batch_size, sequence_length), dtype=np.float32)
    tail_mass = np.zeros((batch_size, sequence_length), dtype=np.float32)
    teacher_entropy = np.zeros((batch_size, sequence_length), dtype=np.float32)
    effective_top_k = np.full((batch_size, sequence_length), top_k, dtype=np.int32)
    for row in range(batch_size):
        for position in range(sequence_length):
            for index in range(top_k):
                top_token_ids[row, position, index] = _marker_token(
                    row,
                    position,
                    index,
                )
                top_probs[row, position, index] = _marker_prob(row, position, index)
                top_log_probs[row, position, index] = -_marker_prob(
                    row,
                    position,
                    index,
                )
            for bucket in range(config.num_buckets):
                bucket_masses[row, position, bucket] = _marker_bucket(
                    row,
                    position,
                    bucket,
                )
            top_mass[row, position] = _marker_top_mass(row, position)
            tail_mass[row, position] = 1.0 - _marker_top_mass(row, position)
            teacher_entropy[row, position] = 3.0 + row + (position / 100.0)
    return {
        "top_token_ids": top_token_ids,
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": top_mass,
        "tail_mass": tail_mass,
        "bucket_masses": bucket_masses,
        "teacher_entropy": teacher_entropy,
    }


def _marker_token(row: int, position: int, index: int) -> int:
    return 1000 + row * 100 + position * 10 + index


def _marker_prob(row: int, position: int, index: int) -> float:
    return np.float32(0.01 * (row + 1) + 0.001 * position + 0.0001 * index).item()


def _marker_top_mass(row: int, position: int) -> float:
    return np.float32(0.4 + 0.01 * row + 0.001 * position).item()


def _marker_bucket(row: int, position: int, index: int) -> float:
    return np.float32(0.03 + 0.01 * index + 0.001 * row + 0.0001 * position).item()
