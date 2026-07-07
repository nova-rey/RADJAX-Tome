from __future__ import annotations

import json
from pathlib import Path

import pytest

import radjax_tome.builder.backend_textbook as backend_textbook
from radjax_tome.backends import (
    CPUReferenceTeacherEmissionBackend,
    TeacherBackendConfig,
)
from radjax_tome.builder import (
    BackendTeacherTextbookBuildConfig,
    build_streaming_backend_teacher_textbook,
    validate_teacher_textbook,
)
from radjax_tome.targets.store import TeacherTargetStore
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _write_corpus(path: Path, count: int = 5) -> Path:
    corpus = path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "example_id": f"ex-{index}",
                        "text": f"streaming example {index}",
                    }
                )
                + "\n"
            )
    return corpus


def _config(tmp_path: Path, **overrides: object) -> BackendTeacherTextbookBuildConfig:
    dataset_path = overrides.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = _write_corpus(tmp_path)
    payload = {
        "output_dir": tmp_path / "streaming_tome",
        "dataset_path": dataset_path,
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dynamic_cascaded_soft_labels_v1",
        "sequence_length": 5,
        "batch_size": 1,
        "max_examples": 5,
        "vocab_size": 11,
        "top_k": 4,
        "num_buckets": 3,
        "streaming": True,
        "shard_size_examples": 2,
    }
    payload.update(overrides)
    return BackendTeacherTextbookBuildConfig(**payload)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_streaming_build_writes_manifest_progress_sidecars_and_cover_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_textbook,
        "load_text_examples",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("streaming build must not load all examples")
        ),
    )

    report = build_streaming_backend_teacher_textbook(_config(tmp_path))
    output = tmp_path / "streaming_tome"
    manifest = _json(output / "run_manifest.json")
    progress = _events(output / "progress_log.jsonl")
    metadata = _json(output / "metadata.json")
    teacher_manifest = _json(output / "teacher_manifest.json")
    emission_config = _json(output / "emission_config.json")
    cover_page = _json(output / "cover_page.json")

    assert report.status == "pass"
    assert validate_teacher_textbook(output).status == "pass"
    assert manifest["schema_version"] == "streaming_run_manifest_v1"
    assert manifest["status"] == "complete"
    assert manifest["num_examples_completed"] == 5
    assert manifest["num_shards_completed"] == 3
    assert [item["event"] for item in progress] == [
        "run_started",
        "shard_started",
        "shard_completed",
        "shard_started",
        "shard_completed",
        "shard_started",
        "shard_completed",
        "run_completed",
    ]
    assert (output / "validation_report.json").is_file()
    assert not list((output / "shards").glob("*.tmp"))
    first_shard = manifest["completed_shards"][0]
    assert first_shard["path"] == "shards/shard-00000.npz"
    assert first_shard["example_start_index"] == 0
    assert first_shard["example_end_index_exclusive"] == 2
    assert str(first_shard["sha256"]).startswith("sha256:")
    params = metadata["target_params"]
    assert params["streaming_build"] == "true"
    assert params["resume_supported"] == "true"
    assert params["run_manifest_path"] == "run_manifest.json"
    assert params["progress_log_path"] == "progress_log.jsonl"
    assert params["num_examples_completed"] == "5"
    assert teacher_manifest["streaming_build"] is True
    assert emission_config["resume_supported"] is True
    assert cover_page["streaming"]["streaming_build"] is True


def test_streaming_resume_skips_completed_shards_and_finishes_after_failure(
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
            if self.calls > 2:
                raise RuntimeError("injected failure")
            return self.backend.emit_batch(batch)

        def close(self) -> None:
            self.backend.close()

    monkeypatch.setattr(
        backend_textbook,
        "create_backend",
        lambda config: FailingBackend(config),
    )
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="completed shards remain available"):
        build_streaming_backend_teacher_textbook(config)

    output = config.output_dir
    failed_manifest = _json(output / "run_manifest.json")
    failure_report = _json(output / "failure_report.json")
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["num_shards_completed"] == 1
    assert failure_report["resume_available"] is True
    assert "rerun with --resume" in str(failure_report["recommended_action"])
    assert (output / "shards" / "shard-00000.npz").is_file()
    assert not (output / "shards" / "shard-00001.npz").exists()

    monkeypatch.setattr(
        backend_textbook,
        "create_backend",
        lambda config: real_backend(config),
    )
    stale_tmp = output / "shards" / "shard-00001.npz.tmp"
    stale_tmp.write_text("stale", encoding="utf-8")
    resumed = build_streaming_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            dataset_path=config.dataset_path,
            resume=True,
        )
    )
    manifest = _json(output / "run_manifest.json")
    events = _events(output / "progress_log.jsonl")

    assert resumed.status == "pass"
    assert manifest["status"] == "complete"
    assert manifest["num_shards_completed"] == 3
    assert not stale_tmp.exists()
    assert "run_resumed" in [item["event"] for item in events]
    assert "shard_started" in [item["event"] for item in events]
    assert TeacherTargetStore.open(output).read_shard(0)["input_ids"].shape[0] == 2


def test_streaming_resume_refuses_dataset_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysFailBackend:
        def __init__(self, config: TeacherBackendConfig) -> None:
            self.backend = CPUReferenceTeacherEmissionBackend(config)

        def emit_batch(self, batch):
            raise RuntimeError("first shard fails")

        def close(self) -> None:
            self.backend.close()

    monkeypatch.setattr(
        backend_textbook,
        "create_backend",
        lambda config: AlwaysFailBackend(config),
    )
    config = _config(tmp_path)
    with pytest.raises(RuntimeError):
        build_streaming_backend_teacher_textbook(config)
    config.dataset_path.write_text(
        json.dumps({"example_id": "changed", "text": "changed"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resume_config_hash mismatch"):
        build_streaming_backend_teacher_textbook(
            _config(
                tmp_path,
                output_dir=config.output_dir,
                dataset_path=config.dataset_path,
                resume=True,
            )
        )

    failure_report = _json(config.output_dir / "failure_report.json")
    assert failure_report["resume_available"] is False
    assert "Config changed" in str(failure_report["recommended_action"])


def test_streaming_output_exists_without_resume_or_overwrite_fails(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    build_streaming_backend_teacher_textbook(config)

    with pytest.raises(ValueError, match="already exists"):
        build_streaming_backend_teacher_textbook(config)


def test_streaming_refuses_global_exemplar_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus-global exemplar selection"):
        build_streaming_backend_teacher_textbook(
            _config(
                tmp_path,
                target_policy="corridor_exemplar_v1",
                exemplar_selection_enabled=True,
            )
        )


def test_streaming_build_cli_and_resume_completed_path(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path)
    output = tmp_path / "cli_streaming"

    build = run_cli(
        ROOT,
        "build",
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "dynamic",
        "--dataset",
        str(corpus),
        "--output",
        str(output),
        "--streaming",
        "--shard-size-examples",
        "2",
        "--batch-size",
        "1",
        "--max-examples",
        "5",
        "--sequence-length",
        "5",
        "--vocab-size",
        "11",
        "--top-k",
        "4",
    )
    assert build.returncode == 0, build.stderr
    assert "status=pass" in build.stdout

    resume = run_cli(
        ROOT,
        "build",
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "dynamic",
        "--dataset",
        str(corpus),
        "--output",
        str(output),
        "--streaming",
        "--resume",
        "--shard-size-examples",
        "2",
        "--batch-size",
        "1",
        "--max-examples",
        "5",
        "--sequence-length",
        "5",
        "--vocab-size",
        "11",
        "--top-k",
        "4",
    )
    assert resume.returncode == 0, resume.stderr
    assert _json(output / "run_manifest.json")["status"] == "complete"
    validate = run_cli(ROOT, "validate", "--path", str(output))
    assert validate.returncode == 0, validate.stderr
    assert "cover_page_status=pass" in validate.stdout
