from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

import radjax_tome.builder.production as production
from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.cli import main as cli_main
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance

EXPECTED_COMMANDS = (
    "allocate-fingerprint-corridor-coverage",
    "audit-selected-linkage",
    "build",
    "build-fingerprint-corridor-leaderboards",
    "build-multi-role-selected-exemplars",
    "claim-corridor-and-backfill-global",
    "corpus",
    "doctor",
    "exemplar-delivery-parity",
    "export-production-global-board-supply",
    "golden",
    "inspect",
    "model",
    "multi-gpu-path-b",
    "pack",
    "package-artifact",
    "parity",
    "plan",
    "production-build",
    "prove-capabilities",
    "unpack",
    "validate",
    "validate-package",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_api() -> Any:
    return importlib.import_module("radjax_tome.builder.native_path_b.api")


def _config(tmp_path: Path, **overrides: object) -> ProductionBuildConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(5):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"M3C canonical boundary {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/m3c-test", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/m3c-test"), provenance
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


def test_exact_native_c6_path_b_routes_once_without_public_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _canonical_api()
    config = _config(tmp_path)
    calls: list[Any] = []
    run_canonical_path_b = api.run_canonical_path_b

    def observe_native_delegation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(args[0] if args else kwargs["config"])
        return run_canonical_path_b(*args, **kwargs)

    monkeypatch.setattr(api, "run_canonical_path_b", observe_native_delegation)

    report = production.build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")
    delivery = _json(config.output_dir / "delivery_report.json")
    corridor = _json(config.output_dir / "corridors" / "corridor_summary.json")

    assert report["status"] == "pass", report["blockers"]
    assert report["schema_version"] == "production_build_report_v1"
    assert len(calls) == 1
    assert isinstance(calls[0], api.CanonicalPathBConfig)
    assert calls[0].source_config is config
    assert delivery["execution_mode"] == "native_c6_path_b_v1"
    assert corridor["selected_exemplars_linked_to_corridor_modes"] is True
    assert progress["schema_version"] == "production_progress_v1"
    assert progress["status"] == "complete"
    assert progress["selected_rerun"]["status"] == "complete"
    assert progress["corridor_export"]["status"] == "complete"


def test_global_only_config_retains_legacy_execution_without_c6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _canonical_api()
    config = _config(
        tmp_path,
        target_policy="dynamic_cascaded_soft_labels_v1",
        exemplar_selection_enabled=False,
        exemplar_delivery_path=None,
        selection_integration_policy="global_only_v1",
        total_selected_exemplar_budget=None,
    )

    def native_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("global-only configuration entered the native adapter")

    monkeypatch.setattr(api, "run_canonical_path_b", native_must_not_run)
    report = production.build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert report["selection_integration_policy"] == "global_only_v1"
    assert not (config.output_dir / "c6").exists()


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    (
        ({"exemplar_selection_enabled": False}, "fail"),
        ({"exemplar_delivery_path": "one_pass_pruned_candidate"}, "pass"),
        ({"target_policy": "corridor"}, "fail"),
    ),
    ids=("selection-disabled", "wrong-delivery", "unnormalized-alias"),
)
def test_partial_or_alias_c6_requests_do_not_route_through_native_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    expected_status: str,
) -> None:
    api = _canonical_api()
    config = _config(tmp_path, **overrides)

    def native_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("partial or alias C6 request entered native adapter")

    monkeypatch.setattr(api, "run_canonical_path_b", native_must_not_run)
    report = production.build_production_gpu_tome(config)

    assert report["status"] == expected_status


def test_research_delivery_config_remains_legacy_and_does_not_route_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _canonical_api()
    config = _config(
        tmp_path,
        selection_integration_policy="global_only_v1",
        exemplar_delivery_path="one_pass_pruned_candidate",
        total_selected_exemplar_budget=None,
    )

    def native_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("research delivery configuration entered native adapter")

    monkeypatch.setattr(api, "run_canonical_path_b", native_must_not_run)
    report = production.build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert report["selection_integration_policy"] == "global_only_v1"
    assert not (config.output_dir / "c6").exists()


def test_cli_production_defaults_and_exact_native_mapping_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = cli_main._build_parser()
    commands = sorted(
        choice
        for action in parser._actions
        for choice in (getattr(action, "choices", None) or {})
    )
    assert tuple(commands) == EXPECTED_COMMANDS

    import radjax_tome.builder as builder

    captured: list[ProductionBuildConfig] = []
    monkeypatch.setattr(
        builder,
        "build_production_gpu_tome",
        lambda config: captured.append(config) or {"status": "pass"},
    )
    monkeypatch.setattr(builder, "render_production_build_summary", lambda report: ())

    required = (
        "production-build",
        "--teacher-model",
        "model",
        "--dataset",
        str(tmp_path / "corpus.jsonl"),
        "--corpus-manifest",
        str(tmp_path / "corpus_manifest.json"),
        "--teacher-model-provenance",
        str(tmp_path / "provenance.json"),
        "--output",
        str(tmp_path / "output"),
    )
    assert cli_main.main(list(required)) == 0
    default = captured.pop()
    assert default.target_policy == "corridor_exemplar_v1"
    assert default.selection_integration_policy == "global_only_v1"
    assert default.exemplar_selection_enabled is False
    assert default.exemplar_delivery_path is None
    assert default.total_selected_exemplar_budget is None

    exact_native = required + (
        "--exemplar-selection-enabled",
        "--exemplar-delivery-path",
        "two_pass_rerun_selected",
        "--selection-integration-policy",
        "corridor_first_global_backfill_v1",
        "--total-selected-exemplar-budget",
        "4",
    )
    assert cli_main.main(list(exact_native)) == 0
    mapped = captured.pop()
    assert mapped.target_policy == "corridor_exemplar_v1"
    assert mapped.selection_integration_policy == "corridor_first_global_backfill_v1"
    assert mapped.exemplar_selection_enabled is True
    assert mapped.exemplar_delivery_path == "two_pass_rerun_selected"
    assert mapped.total_selected_exemplar_budget == 4
