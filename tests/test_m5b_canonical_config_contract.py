from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from radjax_tome.builder.config import (
    PackageIntent,
    adapt_legacy_production_build_config,
    derive_execution_plan,
    resolve_tome_build_intent,
    selection_authority_hash_v1,
    selection_authority_payload_v1,
    validate_tome_build_intent,
)
from radjax_tome.builder.production import (
    ProductionBuildConfig,
    _selection_integration_hash,
)


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


def test_m5b_legacy_adapter_preserves_the_67_field_request_in_sections() -> None:
    config = _legacy_config()

    intent = adapt_legacy_production_build_config(config)

    assert intent.teacher.model == config.teacher_model
    assert intent.teacher.tokenizer_id == config.tokenizer_id
    assert intent.corpus.dataset_path == config.dataset_path
    assert intent.corpus.max_examples == config.max_examples
    assert intent.behavior.dynamic_mass_threshold == config.dynamic_mass_threshold
    assert intent.corridor_policy.include_perverse_tail_in_student == (
        config.include_perverse_tail_in_student
    )
    assert intent.execution.gpu_batch_size_auto_max == config.gpu_batch_size_auto_max
    assert intent.outputs.parity_left == config.parity_left
    assert intent.selection.total_selected_exemplar_budget == (
        config.total_selected_exemplar_budget
    )
    assert intent.compatibility.source_passports_path == config.source_passports_path


def test_m5b_resolved_projection_keeps_the_legacy_selection_authority_hash() -> None:
    config = _legacy_config()
    resolved = resolve_tome_build_intent(
        adapt_legacy_production_build_config(config),
        source="legacy_production_adapter",
    )

    assert selection_authority_hash_v1(resolved) == _selection_integration_hash(config)
    assert selection_authority_payload_v1(resolved) == {
        "selection_integration_policy": "corridor_first_global_backfill_v1",
        "teacher_model": "teacher",
        "tokenizer_id": "tokenizer",
        "dataset_path": "/data/corpus.jsonl",
        "corpus_manifest_path": "/data/manifest.json",
        "target_policy": "corridor_exemplar_v1",
        "sequence_length": 128,
        "vocab_size": 262144,
        "top_k": 32,
        "num_buckets": 4,
        "dynamic_top_k_min": 32,
        "dynamic_top_k_max": 262144,
        "dynamic_mass_threshold": 0.99,
        "selected_rerun_batch_size": 8,
        "total_selected_exemplar_budget": 256,
        "fingerprint_corridor_budget_fraction": "0.50",
        "fingerprint_corridor_budget_max": None,
        "fingerprint_corridor_mode_cap": 10,
        "fingerprint_corridor_candidate_pool_cap": 4,
        "require_full_selected_budget": True,
        "full_width_cap_numerator": 1,
        "full_width_cap_denominator": 3,
        "c2_schema": "radjax.c2_corridor_candidate_leaderboards.v1",
        "c3_schema": "radjax.c3_corridor_coverage_plan.v1",
        "c4_schema": "radjax.c4_corridor_global_claims.v1",
        "c5_schema": "radjax.multi_role_selected_exemplar.v1",
        "delivery_path": "two_pass_rerun_selected",
    }


def test_m5b_execution_plan_preserves_current_path_defaults_without_io() -> None:
    intent = adapt_legacy_production_build_config(_legacy_config())
    resolved = resolve_tome_build_intent(intent, source="legacy_production_adapter")

    plan = derive_execution_plan(resolved)

    assert plan.run_plan_path == Path("/out/run_plan.json")
    assert plan.production_report_path == Path("/out/production_build_report.json")
    assert plan.parity_report_path == Path("/out/parity_report.json")
    assert plan.run_manifest_path == Path("/out/run_manifest.json")
    assert plan.progress_log_path == Path("/out/progress_log.jsonl")


def test_m5b_validation_rejects_explicitly_contradictory_controls() -> None:
    intent = adapt_legacy_production_build_config(_legacy_config())
    invalid = replace(
        intent,
        behavior=replace(intent.behavior, top_k=intent.behavior.vocab_size + 1),
        execution=replace(
            intent.execution,
            gpu_batch_size_mode="custom",
            gpu_batch_size_custom=None,
        ),
        package=PackageIntent(profile="student", transport="tgz"),
    )

    errors = validate_tome_build_intent(invalid)

    assert "behavior.top_k must be positive and no greater than vocab_size" in errors
    assert "execution.gpu_batch_size_custom must be positive in custom mode" in errors
    with pytest.raises(ValueError, match="invalid Tome build intent"):
        resolve_tome_build_intent(invalid)
