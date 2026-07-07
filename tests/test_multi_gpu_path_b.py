from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.builder import (
    MultiGPUPathBConfig,
    build_path_b_assignments,
    merge_path_b_candidate_records,
    normalize_multi_gpu_devices,
    run_multi_gpu_path_b_candidate_harness,
)
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sources = []
    for index in range(7):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"path b example {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus_out"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = _fake_model_dir(tmp_path)
    provenance = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/local-test"),
        provenance,
    )
    return (
        corpus_dir / "corpus.jsonl",
        corpus_dir / "corpus_manifest.json",
        provenance,
        model,
    )


def _config(tmp_path: Path, **overrides: object) -> MultiGPUPathBConfig:
    dataset, corpus_manifest, provenance, model = _inputs(tmp_path)
    payload = {
        "teacher_model": str(model),
        "dataset_path": dataset,
        "corpus_manifest_path": corpus_manifest,
        "teacher_model_provenance_path": provenance,
        "output_dir": tmp_path / "multi_gpu_path_b_out",
        "devices": ("cuda:0", "cuda:1"),
        "target_policy": "corridor_exemplar_v1",
        "sequence_length": 8,
        "batch_size_per_device": 2,
        "shard_size_examples": 2,
        "max_examples": 5,
        "fake_workers": True,
    }
    payload.update(overrides)
    return MultiGPUPathBConfig(**payload)


def test_cli_writes_report_manifest_and_worker_local_outputs(tmp_path: Path) -> None:
    dataset, corpus_manifest, provenance, model = _inputs(tmp_path)
    output = tmp_path / "cli_multi_gpu_path_b"

    result = run_cli(
        ROOT,
        "multi-gpu-path-b",
        "--teacher-model",
        str(model),
        "--dataset",
        str(dataset),
        "--corpus-manifest",
        str(corpus_manifest),
        "--teacher-model-provenance",
        str(provenance),
        "--output",
        str(output),
        "--devices",
        "0,cuda:1",
        "--target-policy",
        "corridor",
        "--sequence-length",
        "8",
        "--batch-size-per-device",
        "2",
        "--shard-size-examples",
        "2",
        "--max-examples",
        "5",
        "--fake-workers",
    )

    assert result.returncode == 0, result.stderr
    assert "experimental multi-GPU Path B scheduling is enabled" in result.stdout
    report = _json(output / "multi_gpu_path_b_report.json")
    manifest = _json(output / "multi_gpu_worker_manifest.json")
    assert report["status"] == "pass"
    assert report["schema_version"] == "multi_gpu_path_b_report_v1"
    assert report["experimental"] is True
    assert report["recommended_path"] == "single_gpu_production_build"
    assert report["selected_devices"] == ["cuda:0", "cuda:1"]
    assert report["candidate_execution_mode"] == "fake_for_scheduler_test"
    assert report["claims_not_made"]["no_ddp"] is True
    assert report["claims_not_made"]["no_model_parallelism"] is True
    assert report["claims_not_made"]["no_combined_vram"] is True
    assert report["claims_not_made"]["no_model_download"] is True
    assert manifest["schema_version"] == "multi_gpu_worker_manifest_v1"
    assert manifest["candidate_execution_mode"] == "fake_for_scheduler_test"
    assert (output / "workers" / "worker-000" / "device.json").is_file()
    assert (output / "workers" / "worker-000" / "assignments.jsonl").is_file()
    assert not (
        output / "workers" / "worker-000" / "multi_gpu_worker_manifest.json"
    ).exists()
    assert _jsonl(output / "merged_candidates.jsonl")


def test_cli_requires_explicit_devices(tmp_path: Path) -> None:
    dataset, corpus_manifest, provenance, model = _inputs(tmp_path)

    result = run_cli(
        ROOT,
        "multi-gpu-path-b",
        "--teacher-model",
        str(model),
        "--dataset",
        str(dataset),
        "--corpus-manifest",
        str(corpus_manifest),
        "--teacher-model-provenance",
        str(provenance),
        "--output",
        str(tmp_path / "out"),
        "--batch-size-per-device",
        "2",
        "--shard-size-examples",
        "2",
        "--fake-workers",
    )

    assert result.returncode != 0
    assert "--devices" in result.stderr


def test_device_normalization_and_round_robin_assignments() -> None:
    blockers: list[str] = []
    devices = normalize_multi_gpu_devices(("0,cuda:1", " 2 "), blockers)
    assignments = build_path_b_assignments(
        num_examples_effective=7,
        shard_size_examples=2,
        devices=devices,
    )

    assert blockers == []
    assert devices == ("cuda:0", "cuda:1", "cuda:2")
    assert [item["device"] for item in assignments] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:0",
    ]
    assert [
        (item["example_start_index"], item["example_end_index_exclusive"])
        for item in assignments
    ] == [(0, 2), (2, 4), (4, 6), (6, 7)]


def test_deterministic_merge_is_stable_under_shuffle_and_ties() -> None:
    records = [
        {
            "score": 0.5,
            "tie_break_key": "b",
            "example_index": 0,
            "position_index": 0,
            "assignment_id": "assignment-00000",
            "worker_id": "worker-000",
        },
        {
            "score": 0.9,
            "tie_break_key": "z",
            "example_index": 2,
            "position_index": 0,
            "assignment_id": "assignment-00001",
            "worker_id": "worker-001",
        },
        {
            "score": 0.5,
            "tie_break_key": "a",
            "example_index": 1,
            "position_index": 0,
            "assignment_id": "assignment-00002",
            "worker_id": "worker-000",
        },
    ]

    first = merge_path_b_candidate_records(records)
    second = merge_path_b_candidate_records(list(reversed(records)))

    assert first == second
    assert [item["tie_break_key"] for item in first["records"]] == ["z", "a", "b"]
    assert first["deterministic_merge_policy"].startswith("score_desc")


def test_assignment_level_resume_reuses_completed_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_multi_gpu_path_b_candidate_harness(config)
    first_manifest = _json(config.output_dir / "multi_gpu_worker_manifest.json")

    resumed = run_multi_gpu_path_b_candidate_harness(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )
    resumed_manifest = _json(config.output_dir / "multi_gpu_worker_manifest.json")

    assert first["status"] == "pass"
    assert resumed["status"] == "pass"
    assert resumed["assignments_completed"] == first["assignments_completed"]
    assert resumed["candidate_record_count"] == first["candidate_record_count"]
    assert (
        first_manifest["completed_assignments"]
        == resumed_manifest["completed_assignments"]
    )


def test_non_fake_real_execution_is_truthfully_blocked(tmp_path: Path) -> None:
    report = run_multi_gpu_path_b_candidate_harness(
        _config(tmp_path, fake_workers=False)
    )

    assert report["status"] == "fail"
    assert any(
        "real multi-GPU backend candidate execution is not implemented" in item
        for item in report["blockers"]
    )
    assert report["claims_not_made"]["no_full_multi_gpu_burn_validation"] is True


def test_production_build_default_has_no_multi_gpu_flag() -> None:
    help_result = run_cli(ROOT, "production-build", "--help")

    assert help_result.returncode == 0, help_result.stderr
    assert "--experimental-multi-gpu-path-b" not in help_result.stdout
    assert "--devices" not in help_result.stdout
