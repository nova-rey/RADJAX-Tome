from __future__ import annotations

import json
from pathlib import Path

import pytest

import radjax_tome.builder.production as production
from radjax_tome.backends import (
    CPUReferenceTeacherEmissionBackend,
    TeacherBackendConfig,
)
from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import write_json
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
from radjax_tome.reports import TomeParityReport
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


def _production_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(6):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"production example {index}", encoding="utf-8")
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


def _config(tmp_path: Path, **overrides: object) -> ProductionBuildConfig:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(tmp_path)
    payload = {
        "teacher_model": str(model),
        "dataset_path": dataset,
        "corpus_manifest_path": corpus_manifest,
        "teacher_model_provenance_path": provenance_path,
        "output_dir": tmp_path / "production_tome",
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dynamic_cascaded_soft_labels_v1",
        "sequence_length": 5,
        "vocab_size": 13,
        "top_k": 4,
        "num_buckets": 3,
        "gpu_batch_size_mode": "preset",
        "gpu_batch_size_preset": 2,
        "shard_size_examples": 2,
        "max_examples": 5,
    }
    payload.update(overrides)
    return ProductionBuildConfig(**payload)


def test_production_build_writes_plan_report_cover_and_valid_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    report = build_production_gpu_tome(config)
    output = config.output_dir

    assert report["schema_version"] == "production_build_report_v1"
    assert report["status"] == "pass"
    assert report["run_plan_status"] == "pass"
    assert report["effective_batch_size"] == 2
    assert report["validation_status"] == "pass"
    assert report["streaming_build"] is True
    assert report["parity_status"] == "not_run"
    assert report["claims_not_made"]["no_model_download"] is True
    assert report["claims_not_made"]["no_silent_cpu_fallback"] is True
    assert report["inputs"]["allow_downloads"] is False
    assert (output / "run_plan.json").is_file()
    assert (output / "production_build_report.json").is_file()
    assert (output / "cover_page.json").is_file()
    assert (output / "run_manifest.json").is_file()
    assert (output / "progress_log.jsonl").is_file()
    assert _json(output / "run_manifest.json")["status"] == "complete"


def test_production_build_progress_sidecar_reaches_complete(tmp_path: Path) -> None:
    config = _config(tmp_path, progress=True)

    report = build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")

    assert report["status"] == "pass"
    assert report["production_progress_path"].endswith("production_progress.json")
    assert progress["schema_version"] == "production_progress_v1"
    assert progress["status"] == "complete"
    assert progress["phase"] == "complete"
    assert progress["production_status"] == "pass"
    assert progress["score_pass"]["status"] == "complete"
    assert progress["score_pass"]["examples_processed"] == 5
    assert progress["score_pass"]["examples_total"] == 5
    assert progress["score_pass"]["shard_count_written"] == 3
    assert progress["validation"]["status"] == "complete"
    assert progress["report_writing"]["status"] == "complete"


def test_production_config_passes_dynamic_top_k_controls_to_backend(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        dynamic_top_k_min=3,
        dynamic_top_k_max=128,
        dynamic_mass_threshold=0.975,
    )

    backend_config = production._backend_config(config)

    assert backend_config.dynamic_top_k_min == 3
    assert backend_config.dynamic_top_k_max == 128
    assert backend_config.dynamic_mass_threshold == 0.975


def test_production_build_stops_on_planner_failure(tmp_path: Path) -> None:
    config = _config(tmp_path, max_artifact_bytes=1)

    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["run_plan_status"] == "fail"
    assert any("estimated artifact size" in item for item in report["blockers"])
    assert (config.output_dir / "run_plan.json").is_file()
    assert not (config.output_dir / "metadata.json").exists()


def test_production_build_warns_by_default_and_can_fail_on_plan_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_plan(plan_config):
        status = "fail" if plan_config.fail_on_warnings else "warn"
        blockers = (
            ["fail_on_warnings enabled and planner emitted warnings"]
            if plan_config.fail_on_warnings
            else []
        )
        return {
            "status": status,
            "warnings": ["planner warning"],
            "blockers": blockers,
            "resolved_batch_policy": {"effective_gpu_batch_size": 2},
            "artifact_estimates": {"estimated_total_artifact_bytes": 123},
        }

    monkeypatch.setattr(production, "build_gpu_run_plan", fake_plan)
    warn_report = build_production_gpu_tome(_config(tmp_path / "warn"))
    fail_report = build_production_gpu_tome(
        _config(tmp_path / "fail", fail_on_plan_warnings=True)
    )

    assert warn_report["status"] == "warn"
    assert warn_report["run_plan_status"] == "warn"
    assert warn_report["effective_batch_size"] == 2
    assert fail_report["status"] == "fail"
    assert fail_report["run_plan_status"] == "fail"
    assert any("fail_on_warnings" in item for item in fail_report["blockers"])
    assert not (Path(fail_report["output_dir"]) / "metadata.json").exists()


def test_production_build_honors_no_build_if_plan_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "build_gpu_run_plan",
        lambda _config: {
            "status": "warn",
            "warnings": ["planner warning"],
            "blockers": [],
            "resolved_batch_policy": {"effective_gpu_batch_size": 2},
        },
    )

    report = build_production_gpu_tome(_config(tmp_path, no_build_if_plan_warn=True))

    assert report["status"] == "fail"
    assert any("no_build_if_plan_warn" in item for item in report["blockers"])
    assert not (config_output := Path(report["output_dir"]) / "metadata.json").exists()
    assert config_output.name == "metadata.json"


def test_production_build_passes_effective_batch_to_streaming_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "build_gpu_run_plan",
        lambda _config: {
            "status": "pass",
            "warnings": [],
            "blockers": [],
            "resolved_batch_policy": {"effective_gpu_batch_size": 3},
        },
    )

    config = _config(tmp_path, shard_size_examples=3)
    report = build_production_gpu_tome(config)
    manifest = _json(config.output_dir / "run_manifest.json")

    assert report["status"] == "pass"
    assert report["effective_batch_size"] == 3
    assert manifest["batch_size"] == 3


def test_production_build_missing_inputs_fail_before_build(tmp_path: Path) -> None:
    missing_corpus = build_production_gpu_tome(
        _config(tmp_path / "corpus_missing", corpus_manifest_path=tmp_path / "missing")
    )
    missing_provenance = build_production_gpu_tome(
        _config(
            tmp_path / "provenance_missing",
            teacher_model_provenance_path=tmp_path / "missing_provenance.json",
        )
    )

    assert missing_corpus["status"] == "fail"
    assert any(
        "corpus manifest path missing" in item for item in missing_corpus["blockers"]
    )
    assert missing_provenance["status"] == "fail"
    assert any(
        "teacher model provenance path missing" in item
        for item in missing_provenance["blockers"]
    )


def test_production_build_resume_completed_reports_already_complete(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "pass"
    assert resumed["resume_requested"] is True
    assert resumed["already_complete"] is True


def test_production_build_completed_resume_skips_failing_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)

    def fail_if_called(_config):
        raise AssertionError("planner should not run")

    monkeypatch.setattr(production, "build_gpu_run_plan", fail_if_called)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "pass"
    assert resumed["already_complete"] is True
    assert resumed["resume_requested"] is True
    assert resumed["validation_status"] == "pass"
    assert resumed["run_plan_status"] == "not_run"
    assert resumed["build_status"] == "already_complete"
    assert "planner should not run" not in resumed["blockers"]
    assert (config.output_dir / "production_build_report.json").is_file()


def test_production_build_completed_invalid_resume_fails_without_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    (config.output_dir / "metadata.json").unlink()

    def fail_if_called(_config):
        raise AssertionError("planner should not run")

    monkeypatch.setattr(production, "build_gpu_run_plan", fail_if_called)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "fail"
    assert resumed["already_complete"] is True
    assert resumed["validation_status"] == "fail"
    assert resumed["run_plan_status"] == "not_run"
    assert resumed["build_status"] == "already_complete_invalid"
    assert resumed["blockers"]
    assert "planner should not run" not in resumed["blockers"]


def test_production_build_overwrite_rebuilds_and_preserves_run_plan(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    stale = config.output_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    rebuilt = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, overwrite=True)
    )

    assert first["status"] == "pass"
    assert rebuilt["status"] == "pass"
    assert not stale.exists()
    assert (config.output_dir / "run_plan.json").is_file()
    assert (config.output_dir / "production_build_report.json").is_file()


def test_production_build_failure_report_is_preserved_on_streaming_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_backend = CPUReferenceTeacherEmissionBackend

    class FailingBackend:
        def __init__(self, config: TeacherBackendConfig) -> None:
            self.backend = real_backend(config)
            self.calls = 0

        def emit_batch(self, batch):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("injected production failure")
            return self.backend.emit_batch(batch)

        def close(self) -> None:
            self.backend.close()

    monkeypatch.setattr(
        "radjax_tome.builder.backend_textbook.create_backend",
        lambda config: FailingBackend(config),
    )

    config = _config(tmp_path, shard_size_examples=2)
    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert any(
        "completed shards remain available" in item for item in report["blockers"]
    )
    failure = _json(config.output_dir / "failure_report.json")
    assert failure["resume_available"] is True


def test_production_build_records_parity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compare(*_args, **_kwargs):
        return TomeParityReport(status="fail", blockers=("parity failed",))

    def fake_write(report: TomeParityReport, path: Path) -> None:
        write_json(path, report.to_dict())

    monkeypatch.setattr(production, "compare_tome_artifacts", fake_compare)
    monkeypatch.setattr(production, "write_tome_parity_report", fake_write)

    config = _config(tmp_path, parity_left=tmp_path / "baseline")
    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["parity_status"] == "fail"
    assert any("parity failed" in item for item in report["blockers"])
    assert (config.output_dir / "parity_report.json").is_file()


def test_production_build_custom_manifest_and_progress_paths_are_truthful(
    tmp_path: Path,
) -> None:
    output = tmp_path / "production_tome"
    config = _config(
        tmp_path,
        output_dir=output,
        run_manifest_path=output / "custom" / "run_manifest.json",
        progress_log_path=output / "custom" / "progress_log.jsonl",
    )

    report = build_production_gpu_tome(config)
    metadata = _json(output / "metadata.json")
    teacher_manifest = _json(output / "teacher_manifest.json")
    emission_config = _json(output / "emission_config.json")
    cover_page = _json(output / "cover_page.json")

    assert report["status"] == "pass"
    assert metadata["target_params"]["run_manifest_path"] == "custom/run_manifest.json"
    assert metadata["target_params"]["progress_log_path"] == "custom/progress_log.jsonl"
    assert teacher_manifest["run_manifest_path"] == "custom/run_manifest.json"
    assert emission_config["progress_log_path"] == "custom/progress_log.jsonl"
    assert cover_page["streaming"]["run_manifest_path"] == "custom/run_manifest.json"
    assert report["run_manifest_path"].endswith("custom/run_manifest.json")
    assert report["progress_log_path"].endswith("custom/progress_log.jsonl")


def test_production_build_cli_smoke_and_no_allow_downloads_flag(
    tmp_path: Path,
) -> None:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(tmp_path)
    output = tmp_path / "cli_production"

    result = run_cli(
        ROOT,
        "production-build",
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "dynamic",
        "--teacher-model",
        str(model),
        "--dataset",
        str(dataset),
        "--corpus-manifest",
        str(corpus_manifest),
        "--teacher-model-provenance",
        str(provenance_path),
        "--output",
        str(output),
        "--gpu-batch-size-mode",
        "preset",
        "--gpu-batch-size-preset",
        "2",
        "--shard-size-examples",
        "2",
        "--max-examples",
        "5",
        "--sequence-length",
        "5",
        "--vocab-size",
        "13",
        "--top-k",
        "4",
        "--dynamic-top-k-max",
        "128",
    )
    help_result = run_cli(ROOT, "production-build", "--help")
    production_report = _json(output / "production_build_report.json")
    emission_config = _json(output / "emission_config.json")

    assert result.returncode == 0, result.stderr
    assert "status=pass" in result.stdout
    assert "phase=score_pass" in result.stdout
    assert (output / "production_build_report.json").is_file()
    assert _json(output / "production_progress.json")["status"] == "complete"
    assert production_report["dynamic_top_k_max"] == 128
    assert production_report["inputs"]["dynamic_top_k_max"] == 128
    assert emission_config["dynamic_top_k_max"] == 128
    assert "--allow-downloads" not in help_result.stdout
    assert "--dynamic-top-k-min" in help_result.stdout
    assert "--dynamic-top-k-max" in help_result.stdout
    assert "--dynamic-mass-threshold" in help_result.stdout
    assert "--progress" in help_result.stdout
    assert "--no-progress" in help_result.stdout


def test_fail_fast_is_not_user_facing() -> None:
    build_help = run_cli(ROOT, "build", "--help")
    production_help = run_cli(ROOT, "production-build", "--help")

    assert build_help.returncode == 0, build_help.stderr
    assert production_help.returncode == 0, production_help.stderr
    assert "--fail-fast" not in build_help.stdout
    assert "--fail-fast" not in production_help.stdout
