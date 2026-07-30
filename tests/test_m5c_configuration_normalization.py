from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from radjax_tome.builder.config import (
    _PRODUCTION_OVERRIDE_SECTIONS,
    PRODUCTION_PRESETS,
    canonical_production_build_intent,
    normalize_cli_production_build_request,
    normalize_production_build_request,
    production_build_config_from_resolved,
    selection_authority_hash_v1,
)
from radjax_tome.builder.production import (
    ProductionBuildConfig,
    _selection_integration_hash,
)
from radjax_tome.cli import main as cli_main


def _legacy_config() -> ProductionBuildConfig:
    return ProductionBuildConfig(
        teacher_model="teacher",
        tokenizer_id="tokenizer",
        dataset_path=Path("/data/corpus.jsonl"),
        corpus_manifest_path=Path("/data/manifest.json"),
        teacher_model_provenance_path=Path("/data/provenance.json"),
        output_dir=Path("/out"),
        sequence_length=128,
        vocab_size=262144,
        top_k=32,
        num_buckets=4,
        dynamic_top_k_min=32,
        dynamic_top_k_max=262144,
        dynamic_mass_threshold=0.99,
        selected_rerun_batch_size=8,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=256,
        fingerprint_corridor_budget_fraction="0.50",
        fingerprint_corridor_budget_max=None,
        fingerprint_corridor_mode_cap=10,
        fingerprint_corridor_candidate_pool_cap=4,
        require_full_selected_budget=True,
        exemplar_delivery_path="two_pass_rerun_selected",
    )


def _cli_request(
    *,
    preset_name: str | None = None,
    advanced_overrides: dict[str, Any] | None = None,
):
    return normalize_cli_production_build_request(
        teacher_model="teacher",
        tokenizer_id="tokenizer",
        dataset_path=Path("/data/corpus.jsonl"),
        corpus_manifest_path=Path("/data/manifest.json"),
        teacher_model_provenance_path=Path("/data/provenance.json"),
        output_dir=Path("/out"),
        preset_name=preset_name,
        advanced_overrides=advanced_overrides or {},
    )


def test_m5c_legacy_flat_config_round_trips_through_the_execution_adapter() -> None:
    legacy = _legacy_config()

    normalized = normalize_production_build_request(legacy)
    execution = production_build_config_from_resolved(normalized.resolved)

    assert execution == legacy
    assert normalized.resolved.resolution.source == "legacy_production_adapter"
    assert normalized.selection_authority_hash == _selection_integration_hash(legacy)
    assert selection_authority_hash_v1(
        normalized.resolved
    ) == _selection_integration_hash(legacy)


def test_m5c_cli_and_programmatic_requests_resolve_identically() -> None:
    programmatic = normalize_production_build_request(_legacy_config())
    cli = _cli_request(
        advanced_overrides={
            "sequence_length": 128,
            "vocab_size": 262144,
            "top_k": 32,
            "num_buckets": 4,
            "dynamic_top_k_min": 32,
            "dynamic_top_k_max": 262144,
            "dynamic_mass_threshold": 0.99,
            "selected_rerun_batch_size": 8,
            "selection_integration_policy": "corridor_first_global_backfill_v1",
            "total_selected_exemplar_budget": 256,
            "fingerprint_corridor_budget_fraction": "0.50",
            "fingerprint_corridor_budget_max": None,
            "fingerprint_corridor_mode_cap": 10,
            "fingerprint_corridor_candidate_pool_cap": 4,
            "require_full_selected_budget": True,
            "exemplar_delivery_path": "two_pass_rerun_selected",
            "retain_unselected_exemplar_payloads": True,
            "progress": False,
        }
    )

    assert cli.resolved.intent == programmatic.resolved.intent
    assert cli.selection_authority_hash == programmatic.selection_authority_hash


def test_m5c_advanced_override_adapter_accounts_for_every_legacy_field() -> None:
    assert set(_PRODUCTION_OVERRIDE_SECTIONS) == {
        field.name for field in fields(ProductionBuildConfig)
    }


def test_m5c_t4_presets_only_change_semantic_size() -> None:
    one_k = _cli_request(preset_name="t4-1k")
    ten_k = _cli_request(preset_name="t4-10k")
    hundred_k = _cli_request(preset_name="production-100k")

    assert PRODUCTION_PRESETS == ("smoke", "t4-1k", "t4-10k", "production-100k")
    assert one_k.resolved.intent.corpus.max_examples == 1_000
    assert ten_k.resolved.intent.corpus.max_examples == 10_000
    assert hundred_k.resolved.intent.corpus.max_examples == 100_000
    for other in (ten_k, hundred_k):
        assert replace(
            other.resolved.intent,
            corpus=replace(other.resolved.intent.corpus, max_examples=None),
        ) == replace(
            one_k.resolved.intent,
            corpus=replace(one_k.resolved.intent.corpus, max_examples=None),
        )


def test_m5c_preset_then_explicit_override_is_recorded_and_wins() -> None:
    normalized = _cli_request(
        preset_name="t4-1k",
        advanced_overrides={"max_examples": 7, "gpu_batch_size_preset": 3},
    )

    assert normalized.resolved.intent.corpus.max_examples == 7
    assert normalized.resolved.intent.execution.gpu_batch_size_preset == 3
    assert normalized.resolved.resolution.preset_name == "t4-1k"
    assert normalized.resolved.resolution.explicit_override_fields == (
        "gpu_batch_size_preset",
        "max_examples",
    )


def test_m5c_smoke_preset_is_canonical_path_b_without_unselected_retention() -> None:
    normalized = _cli_request(preset_name="smoke")
    intent = normalized.resolved.intent

    assert intent.teacher.backend == "cpu_reference"
    assert intent.teacher.runtime_mode == "cpu"
    assert intent.corpus.max_examples == 4
    assert intent.selection.exemplar_selection_enabled is True
    assert intent.selection.exemplar_delivery_path == "two_pass_rerun_selected"
    assert intent.selection.total_selected_exemplar_budget == 4
    assert intent.selection.retain_unselected_exemplar_payloads is False


def test_m5c_contradiction_fails_before_execution_resolution() -> None:
    intent = canonical_production_build_intent(
        teacher_model="teacher",
        dataset_path=Path("/data/corpus.jsonl"),
        corpus_manifest_path=Path("/data/manifest.json"),
        teacher_model_provenance_path=Path("/data/provenance.json"),
        output_dir=Path("/out"),
    )
    invalid = replace(intent, behavior=replace(intent.behavior, top_k=33))

    with pytest.raises(ValueError, match="behavior.top_k"):
        normalize_production_build_request(invalid)


def test_m5c_cli_can_inspect_resolved_preset_without_running_production(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli_main.main(
        [
            "production-build",
            "--teacher-model",
            "teacher",
            "--dataset",
            "/data/corpus.jsonl",
            "--corpus-manifest",
            "/data/manifest.json",
            "--teacher-model-provenance",
            "/data/provenance.json",
            "--output",
            "/out",
            "--preset",
            "t4-1k",
            "--max-examples",
            "7",
            "--print-resolved-config",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["resolved_config"]["resolution"]["preset_name"] == "t4-1k"
    assert payload["resolved_config"]["intent"]["corpus"]["max_examples"] == 7
    assert payload["execution_plan"]["run_plan_path"] == "/out/run_plan.json"
