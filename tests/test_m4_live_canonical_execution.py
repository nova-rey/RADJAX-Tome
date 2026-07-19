"""Live production checks for canonical M4B stages and M4C resume routing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import radjax_tome.builder.native_path_b.orchestrator as orchestrator
import radjax_tome.builder.native_path_b.resume as native_resume
import radjax_tome.builder.production as production
from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_config(tmp_path: Path, **overrides: object) -> ProductionBuildConfig:
    sources: list[Path] = []
    for index in range(5):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"M4 live canonical execution {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/m4-live", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/m4-live"), provenance
    )

    payload: dict[str, object] = {
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


def _observe_stages(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    for name in (
        "run_slice_two",
        "run_slice_three",
        "run_slice_four",
        "run_slice_five",
    ):
        operation = getattr(orchestrator, name)

        def observe(
            *args: Any,
            _name: str = name,
            _operation: Any = operation,
            **kwargs: Any,
        ) -> Any:
            calls.append(_name)
            return _operation(*args, **kwargs)

        monkeypatch.setattr(orchestrator, name, observe)


def _canonical_artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    paths = [
        output_dir / "run_manifest.json",
        output_dir / "metadata.json",
        output_dir / "corridors" / "corridor_summary.json",
        output_dir / "leaderboards" / "selected_exemplars.json",
        output_dir / "selected_exemplars" / "payload_index.json",
        *sorted((output_dir / "selected_exemplars").glob("selected-exemplars-*.json")),
    ]
    return {str(path.relative_to(output_dir)): path.read_bytes() for path in paths}


def test_exact_canonical_build_traverses_all_native_post_score_slices_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _observe_stages(monkeypatch, calls)

    config = _canonical_config(tmp_path)
    report = production.build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")
    corridor = _json(config.output_dir / "corridors" / "corridor_summary.json")

    assert calls == [
        "run_slice_two",
        "run_slice_three",
        "run_slice_four",
        "run_slice_five",
    ]
    assert report["status"] == "pass", report["blockers"]
    assert progress["status"] == "complete"
    assert progress["production_status"] == "pass"
    assert corridor["selected_exemplars_linked_to_corridor_modes"] is True


def test_global_only_build_bypasses_all_native_post_score_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _canonical_config(
        tmp_path,
        target_policy="dynamic_cascaded_soft_labels_v1",
        exemplar_selection_enabled=False,
        exemplar_delivery_path=None,
        selection_integration_policy="global_only_v1",
        total_selected_exemplar_budget=None,
    )

    def native_stage_must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "global-only build entered a native Path-B post-score stage"
        )

    for name in (
        "run_slice_two",
        "run_slice_three",
        "run_slice_four",
        "run_slice_five",
    ):
        monkeypatch.setattr(orchestrator, name, native_stage_must_not_run)

    report = production.build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert report["selection_integration_policy"] == "global_only_v1"
    assert (config.output_dir / "production_build_report.json").is_file()
    assert (config.output_dir / "run_manifest.json").is_file()
    assert not (config.output_dir / "c6").exists()


def test_canonical_resume_uses_resolver_without_changing_terminal_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _canonical_config(tmp_path)
    first_report = production.build_production_gpu_tome(config)
    artifacts_before_resume = _canonical_artifact_bytes(config.output_dir)

    resolver_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_resolver = native_resume.resolve_native_path_b_resume

    def observe_resolver(*args: Any, **kwargs: Any) -> Any:
        resolver_calls.append((args, kwargs))
        return real_resolver(*args, **kwargs)

    monkeypatch.setattr(
        native_resume,
        "resolve_native_path_b_resume",
        observe_resolver,
    )
    resumed_config = replace(config, resume=True)
    resumed_report = production.build_production_gpu_tome(resumed_config)
    resumed_progress = _json(config.output_dir / "production_progress.json")

    assert len(resolver_calls) == 1
    resolver_args, resolver_kwargs = resolver_calls[0]
    assert resolver_args == (config.output_dir,)
    assert resolver_kwargs["config"].source_config is resumed_config
    assert _canonical_artifact_bytes(config.output_dir) == artifacts_before_resume
    assert first_report["status"] == resumed_report["status"] == "pass"
    assert (
        resumed_report["selection_integration_policy"]
        == (first_report["selection_integration_policy"])
    )
    assert resumed_report["build_status"] == first_report["build_status"]
    assert resumed_report["resume_requested"] is first_report["resume_requested"]
    assert resumed_report["already_complete"] is first_report["already_complete"]
    assert resumed_progress["status"] == "complete"
    assert resumed_progress["production_status"] == "pass"
