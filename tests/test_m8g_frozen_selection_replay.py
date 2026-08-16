from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import radjax_tome.builder.production as production_module
from radjax_tome.builder.config import normalize_production_build_request
from radjax_tome.builder.delivery.replay_authority import (
    adopt_verified_selection_replay,
)
from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.builder.production_stages.preflight import validate_required_inputs


def _config(tmp_path: Path) -> ProductionBuildConfig:
    return ProductionBuildConfig(
        teacher_model="teacher",
        tokenizer_id="tokenizer",
        dataset_path=tmp_path / "corpus.jsonl",
        corpus_manifest_path=tmp_path / "corpus_manifest.json",
        teacher_model_provenance_path=tmp_path / "teacher.json",
        output_dir=tmp_path / "out",
        target_policy="corridor_exemplar_v1",
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=256,
        selected_exemplar_budget=256,
        selected_rerun_batch_size=8,
        verified_selection_replay_path=tmp_path / "artifact",
        verified_selection_bundle_manifest_path=tmp_path / "bundle.json",
    )


def test_replay_authority_is_bound_into_resolved_configuration(tmp_path: Path) -> None:
    normalized = normalize_production_build_request(_config(tmp_path))
    selection = normalized.resolved.intent.selection
    assert selection.verified_selection_replay_path == tmp_path / "artifact"
    assert selection.verified_selection_bundle_manifest_path == tmp_path / "bundle.json"
    assert normalized.resolved.intent.selection.representation_mode == (
        "legacy_padded_monolithic"
    )


def test_replay_rejects_absolute_layout_before_adoption(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "m8g_benchmark_workload_bundle_v1",
        "provenance": {
            "status": "HISTORICAL_M8_WORKLOAD_RECOVERED",
            "artifact_layout": {"replay_root": "/escape"},
        },
    }
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(manifest), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(ValueError, match="layout is not relative"):
        adopt_verified_selection_replay(
            bundle_manifest=bundle,
            artifact_root=artifact,
            adopted_root=tmp_path / "adopted",
        )


def test_replay_rejects_symlinked_authority_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest_link = tmp_path / "bundle-link.json"
    artifact_link = tmp_path / "artifact-link"
    manifest_link.symlink_to(manifest)
    artifact_link.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ValueError, match="regular file"):
        adopt_verified_selection_replay(
            bundle_manifest=manifest_link,
            artifact_root=artifact,
            adopted_root=tmp_path / "adopted-a",
        )
    with pytest.raises(ValueError, match="regular directory"):
        adopt_verified_selection_replay(
            bundle_manifest=manifest,
            artifact_root=artifact_link,
            adopted_root=tmp_path / "adopted-b",
        )


def test_replay_preserves_external_c4_c5_rejection(tmp_path: Path) -> None:
    c4 = tmp_path / "c4.json"
    c5 = tmp_path / "c5.json"
    c4.write_text("{}", encoding="utf-8")
    c5.write_text("{}", encoding="utf-8")
    config = replace(_config(tmp_path), c4_claims_path=c4, c5_selection_path=c5)
    blockers: list[str] = []
    validate_required_inputs(config, blockers)
    assert any("cannot be combined with external C4/C5" in item for item in blockers)
    assert any(
        "external C4/C5 checkpoints are not accepted" in item for item in blockers
    )


def test_replay_adoption_precedes_input_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    adopted = tmp_path / "adopted"
    replay = SimpleNamespace(
        adopted_root=adopted,
        replay_identity="sha256:replay",
        records=(),
        checkpoint_digest="sha256:checkpoint",
    )
    from radjax_tome.builder.delivery import replay_authority

    monkeypatch.setattr(
        replay_authority, "adopt_verified_selection_replay", lambda **_: replay
    )
    observed: dict[str, Path] = {}

    def preflight(state: object) -> SimpleNamespace:
        observed["dataset"] = state.config.dataset_path
        observed["manifest"] = state.config.corpus_manifest_path
        observed["provenance"] = state.config.teacher_model_provenance_path
        return SimpleNamespace(status="fail", failure="test stop")

    monkeypatch.setattr(production_module, "_run_existing_preflight", preflight)
    monkeypatch.setattr(
        production_module,
        "_stage_adapter_failure_report",
        lambda *_args, **_kwargs: {"status": "stopped"},
    )
    result = production_module._build_production_gpu_tome_compatibility(config)
    assert result == {"status": "stopped"}
    assert observed == {
        "dataset": adopted / "input/corpus.jsonl",
        "manifest": adopted / "input/corpus_manifest.json",
        "provenance": adopted / "input/teacher_model_provenance.json",
    }
