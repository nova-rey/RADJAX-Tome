from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.reports import (
    GPU_RUN_PLAN_SCHEMA,
    GPURunPlanConfig,
    build_gpu_run_plan,
    write_gpu_run_plan,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _dataset(path: Path, rows: int = 3) -> Path:
    dataset = path / "corpus.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(json.dumps({"text": f"example {index}"}) + "\n")
    return dataset


def _cpu_config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dense_logits",
        "sequence_length": 4,
        "batch_size": 2,
        "vocab_size": 11,
        "top_k": 3,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def _gpu_config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "gpu_torch",
        "runtime_mode": "cpu_gpu",
        "target_policy": "dense_logits",
        "model_id": "local/model",
        "tokenizer_id": "local/model",
        "sequence_length": 4,
        "batch_size": 2,
        "vocab_size": 11,
        "top_k": 3,
        "local_files_only": True,
        "allow_downloads": False,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def _plan_config(tmp_path: Path, **overrides: object) -> GPURunPlanConfig:
    dataset_path = overrides.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = _dataset(tmp_path)
    payload = {
        "backend_config": _cpu_config(),
        "dataset_path": dataset_path,
    }
    payload.update(overrides)
    return GPURunPlanConfig(**payload)


def test_run_plan_schema_is_written(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(_plan_config(tmp_path))
    output = tmp_path / "run_plan.json"

    write_gpu_run_plan(plan, output)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["schema_version"] == GPU_RUN_PLAN_SCHEMA
    assert written["status"] == "warn"
    assert written["doctor_diagnostics"]["report_schema"] == "runtime_doctor_report_v1"
    assert written["claims_not_made"]["no_model_download"] is True
    assert written["claims_not_made"]["no_network_verification"] is True


def test_missing_dataset_path_fails(tmp_path: Path) -> None:
    config = GPURunPlanConfig(
        backend_config=_cpu_config(),
        dataset_path=tmp_path / "missing.jsonl",
    )

    plan = build_gpu_run_plan(config)

    assert plan["status"] == "fail"
    assert any("dataset path missing" in item for item in plan["blockers"])


def test_invalid_corpus_manifest_fails_when_supplied(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text('{"schema_version": "bad"}\n', encoding="utf-8")

    plan = build_gpu_run_plan(_plan_config(tmp_path, corpus_manifest_path=manifest))

    assert plan["status"] == "fail"
    assert any("corpus manifest invalid" in item for item in plan["blockers"])


def test_invalid_teacher_model_provenance_fails_when_supplied(tmp_path: Path) -> None:
    provenance = tmp_path / "teacher_model_provenance.json"
    provenance.write_text('{"schema_version": "bad"}\n', encoding="utf-8")

    plan = build_gpu_run_plan(
        _plan_config(tmp_path, teacher_model_provenance_path=provenance)
    )

    assert plan["status"] == "fail"
    assert any("teacher model provenance invalid" in item for item in plan["blockers"])


def test_missing_provenance_warns_by_default(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(_plan_config(tmp_path))

    assert plan["status"] == "warn"
    assert any("no corpus manifest" in item for item in plan["warnings"])
    assert any("no teacher model provenance" in item for item in plan["warnings"])


def test_valid_corpus_manifest_records_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    corpus_dir = tmp_path / "corpus_out"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=corpus_dir, overwrite=True)
    )

    plan = build_gpu_run_plan(
        GPURunPlanConfig(
            backend_config=_cpu_config(),
            dataset_path=corpus_dir / "corpus.jsonl",
            corpus_manifest_path=corpus_dir / "corpus_manifest.json",
        )
    )

    assert plan["corpus_provenance"]["status"] == "pass"
    assert str(plan["corpus_provenance"]["source_corpus_hash"])


def test_auto_batch_probe_selects_largest_passing_batch(tmp_path: Path) -> None:
    def fake_probe(
        _config: TeacherBackendConfig,
        candidate: int,
    ) -> dict[str, object]:
        if candidate <= 16:
            return {
                "candidate_batch_size": candidate,
                "status": "pass",
                "success": True,
                "observed_memory_allocated_bytes": candidate * 1024,
            }
        return {
            "candidate_batch_size": candidate,
            "status": "fail",
            "success": False,
            "failure_stage": "oom",
            "failure_reason": "out of memory",
            "oom_or_device_failure": True,
        }

    plan = build_gpu_run_plan(
        _plan_config(
            tmp_path,
            backend_config=_gpu_config(
                gpu_batch_size_mode="auto",
                gpu_batch_size_auto_min=1,
                gpu_batch_size_auto_max=32,
            ),
        ),
        probe_candidate_runner=fake_probe,
    )

    assert plan["auto_batch_probe"]["candidate_count"] == 6
    assert plan["auto_batch_probe"]["largest_passing_batch_size"] == 16
    assert plan["auto_batch_probe"]["first_failing_batch_size"] == 32
    assert plan["resolved_batch_policy"]["effective_gpu_batch_size"] == 16


def test_auto_batch_probe_fails_when_no_candidate_passes(tmp_path: Path) -> None:
    def fake_probe(
        _config: TeacherBackendConfig,
        candidate: int,
    ) -> dict[str, object]:
        return {
            "candidate_batch_size": candidate,
            "status": "fail",
            "success": False,
            "failure_stage": "oom",
            "failure_reason": "out of memory",
            "oom_or_device_failure": True,
        }

    plan = build_gpu_run_plan(
        _plan_config(
            tmp_path,
            backend_config=_gpu_config(gpu_batch_size_mode="auto"),
        ),
        probe_candidate_runner=fake_probe,
    )

    assert plan["status"] == "fail"
    assert plan["auto_batch_probe"]["largest_passing_batch_size"] is None
    assert any("no candidate" in item for item in plan["blockers"])


def test_memory_estimates_include_dense_logits_formula(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(_plan_config(tmp_path))

    assert plan["memory_estimates"]["estimated_dense_logits_bytes"] == 8 * 4 * 11 * 4
    assert plan["memory_estimates"]["estimate_confidence"] == "rough"


def test_artifact_estimates_use_dataset_count_and_max_examples(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(
        _plan_config(
            tmp_path,
            dataset_path=_dataset(tmp_path, rows=9),
            max_examples=4,
        )
    )

    assert plan["artifact_estimates"]["dataset_count"] == 9
    assert plan["artifact_estimates"]["num_examples_effective"] == 4
    assert plan["artifact_estimates"]["estimated_total_artifact_bytes"] > 0


def test_dense_logits_large_corpus_warns(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(
        _plan_config(tmp_path, dataset_path=_dataset(tmp_path, rows=1_000))
    )

    assert any("dense logits selected" in item for item in plan["warnings"])


def test_capture_mode_summary_records_effective_mode(tmp_path: Path) -> None:
    plan = build_gpu_run_plan(
        _plan_config(
            tmp_path,
            backend_config=_cpu_config(
                target_policy="corridor_exemplar_v1",
                exemplar_capture_mode="one_pass_candidate",
            ),
        )
    )

    assert (
        plan["capture_mode_estimates"]["exemplar_capture_mode_effective"]
        == "one_pass_candidate"
    )
    assert plan["capture_mode_estimates"]["selection_enabled"] is False


def test_plan_cli_writes_run_plan_and_exits_nonzero_on_fail(tmp_path: Path) -> None:
    output = tmp_path / "run_plan.json"

    result = run_cli(
        ROOT,
        "plan",
        "--teacher-model",
        "local/model",
        "--dataset",
        str(tmp_path / "missing.jsonl"),
        "--output",
        str(output),
    )

    assert result.returncode == 1
    assert "status=fail" in result.stdout
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == GPU_RUN_PLAN_SCHEMA
