from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import radjax_tome.builder.native_path_b.orchestrator as orchestrator
import radjax_tome.builder.production as production
from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(tmp_path: Path, **overrides: object) -> ProductionBuildConfig:
    sources: list[Path] = []
    for index in range(5):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"M4B production integration {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/m4b-test", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/m4b-test"), provenance
    )
    payload = {
        "teacher_model": str(model),
        "tokenizer_id": str(model),
        "dataset_path": corpus_dir / "corpus.jsonl",
        "corpus_manifest_path": corpus_dir / "corpus_manifest.json",
        "teacher_model_provenance_path": provenance,
        "output_dir": tmp_path / "production_tome",
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "corridor_exemplar_v1",
        "sequence_length": 5,
        "vocab_size": 64,
        "top_k": 32,
        "num_buckets": 3,
        "gpu_batch_size_mode": "preset",
        "gpu_batch_size_preset": 2,
        "shard_size_examples": 2,
        "max_examples": 5,
        "progress": True,
        "exemplar_selection_enabled": True,
        "exemplar_delivery_path": "two_pass_rerun_selected",
        "selected_exemplar_budget": 4,
        "retain_unselected_exemplar_payloads": False,
        "selection_integration_policy": "corridor_first_global_backfill_v1",
        "total_selected_exemplar_budget": 4,
    }
    payload.update(overrides)
    return ProductionBuildConfig(**payload)


def test_exact_canonical_build_uses_real_slice_one_before_compatible_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []
    real_slice_one = orchestrator.run_preflight_then_score_pass

    def observe_slice_one(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_slice_one(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "run_preflight_then_score_pass",
        observe_slice_one,
    )
    report = production.build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")
    corridor = _json(config.output_dir / "corridors" / "corridor_summary.json")

    assert report["status"] == "pass", report["blockers"]
    assert len(calls) == 1
    assert calls[0]["propagate_exceptions"] is True
    assert progress["status"] == "complete"
    assert progress["score_pass"]["status"] == "complete"
    assert progress["selected_rerun"]["status"] == "complete"
    assert corridor["selected_exemplars_linked_to_corridor_modes"] is True


def test_global_only_build_bypasses_real_slice_one_and_retains_legacy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        target_policy="dynamic_cascaded_soft_labels_v1",
        exemplar_selection_enabled=False,
        exemplar_delivery_path=None,
        selection_integration_policy="global_only_v1",
        total_selected_exemplar_budget=None,
    )

    def slice_one_must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("global-only build entered native slice-one adapter")

    monkeypatch.setattr(
        orchestrator,
        "run_preflight_then_score_pass",
        slice_one_must_not_run,
    )
    report = production.build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert report["selection_integration_policy"] == "global_only_v1"
    assert (config.output_dir / "production_build_report.json").is_file()
    assert (config.output_dir / "run_manifest.json").is_file()
    assert not (config.output_dir / "c6").exists()


def test_canonical_slice_one_preserves_configured_doctor_exception_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[dict[str, object]] = []
    real_slice_one = orchestrator.run_preflight_then_score_pass

    def observe_slice_one(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_slice_one(*args, **kwargs)

    def exploding_doctor(_config: object) -> dict[str, Any]:
        raise RuntimeError("accelerator preflight required")

    monkeypatch.setattr(
        orchestrator,
        "run_preflight_then_score_pass",
        observe_slice_one,
    )
    monkeypatch.setattr(production, "build_runtime_doctor_report", exploding_doctor)

    with pytest.raises(RuntimeError, match="accelerator preflight required"):
        production.build_production_gpu_tome(config)

    assert len(calls) == 1
    assert calls[0]["propagate_exceptions"] is True
