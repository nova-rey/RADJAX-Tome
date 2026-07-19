from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import radjax_tome.builder.exemplar_delivery as exemplar_delivery
import radjax_tome.builder.production as production
from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(tmp_path: Path) -> ProductionBuildConfig:
    sources = []
    for index in range(5):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"M3A corridor characterization {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/m3a-test", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/m3a-test"), provenance
    )
    return ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id=str(model),
        dataset_path=corpus_dir / "corpus.jsonl",
        corpus_manifest_path=corpus_dir / "corpus_manifest.json",
        teacher_model_provenance_path=provenance,
        output_dir=tmp_path / "production_tome",
        teacher_backend="cpu_reference",
        runtime_mode="cpu",
        target_policy="corridor_exemplar_v1",
        sequence_length=5,
        vocab_size=64,
        top_k=32,
        num_buckets=3,
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=2,
        shard_size_examples=2,
        max_examples=5,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        selected_exemplar_budget=4,
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
        progress=True,
    )


def test_m3a_distinguishes_ordered_early_and_late_corridor_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterize the two existing corridor writes without moving either one."""

    config = _config(tmp_path)
    timeline: list[dict[str, Any]] = []
    early_builder = production.build_corridor_artifacts
    late_builder = exemplar_delivery.build_corridor_artifacts
    delivery = production.materialize_selected_exemplar_delivery

    def observe_early(**kwargs: Any) -> Any:
        assert kwargs["selected_records"] == []
        assert kwargs["selected_payloads"] == []
        assert kwargs.get("progress_callback") is None
        result = early_builder(**kwargs)
        timeline.append(
            {
                "phase": "early_corridor",
                "summary": _json(
                    config.output_dir / "corridors" / "corridor_summary.json"
                ),
            }
        )
        return result

    def observe_late(**kwargs: Any) -> Any:
        timeline.append(
            {
                "phase": "late_corridor_started",
                "selected_record_count": len(kwargs["selected_records"]),
                "has_progress_callback": kwargs.get("progress_callback") is not None,
            }
        )
        result = late_builder(**kwargs)
        timeline.append(
            {
                "phase": "late_corridor_complete",
                "summary": _json(
                    config.output_dir / "corridors" / "corridor_summary.json"
                ),
            }
        )
        return result

    def observe_delivery(delivery_config: Any) -> dict[str, Any]:
        timeline.append({"phase": "selected_delivery_started"})

        def observe_progress(event: dict[str, Any]) -> None:
            timeline.append(
                {
                    "phase": event.get("phase"),
                    "event": event.get("event"),
                }
            )
            assert delivery_config.progress_callback is not None
            delivery_config.progress_callback(event)

        return delivery(replace(delivery_config, progress_callback=observe_progress))

    monkeypatch.setattr(production, "build_corridor_artifacts", observe_early)
    monkeypatch.setattr(exemplar_delivery, "build_corridor_artifacts", observe_late)
    monkeypatch.setattr(
        production, "materialize_selected_exemplar_delivery", observe_delivery
    )

    report = build_production_gpu_tome(config)

    phases = [str(item["phase"]) for item in timeline]
    early_summary = next(
        item["summary"] for item in timeline if item["phase"] == "early_corridor"
    )
    late_summary = next(
        item["summary"]
        for item in timeline
        if item["phase"] == "late_corridor_complete"
    )
    late_start = phases.index("late_corridor_started")

    assert report["status"] == "pass", report["blockers"]
    assert early_summary["selected_exemplar_count"] == 0
    assert early_summary["selected_exemplars_linked_to_corridor_modes"] is False
    assert late_summary["selected_exemplar_count"] == 4
    assert late_summary["selected_exemplars_linked_to_corridor_modes"] is True
    assert (
        _json(config.output_dir / "corridors" / "corridor_summary.json") == late_summary
    )
    assert report["selected_exemplars_linked_to_corridor_modes"] is True
    assert phases.index("early_corridor") < phases.index("selected_delivery_started")
    assert phases.index("selected_rerun") < late_start
    assert any(
        item.get("phase") == "selected_rerun" and item.get("event") == "complete"
        for item in timeline[:late_start]
    )
    assert not any(
        item.get("phase") == "corridor_export" for item in timeline[:late_start]
    )
    assert any(
        item.get("phase") == "corridor_export"
        and item.get("event") == "assignments_written"
        for item in timeline[late_start:]
    )

    progress = _json(config.output_dir / "production_progress.json")
    assert progress["selected_rerun"]["status"] == "complete"
    assert progress["corridor_export"]["status"] == "complete"
    assert (
        progress["corridor_export"]["positions_processed"]
        == report["corridor_positions_used"]
    )

    (config.output_dir / "reports" / "c6_integrated_selection_validation.json").unlink()
    eligibility = production.probe_c6_finalization_only_resume(
        replace(config, resume=True)
    )
    assert eligibility.eligible, eligibility.to_dict()
