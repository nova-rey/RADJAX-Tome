from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from radjax_tome.builder.production import (
    ProductionBuildConfig,
    _selection_integration_hash,
)
from radjax_tome.tome.cover_page import COVER_PAGE_VERSION
from radjax_tome.tome.packaging import PACKAGE_COVER_SCHEMA

_PRODUCTION_CONFIG_FIELDS = (
    "teacher_model",
    "dataset_path",
    "corpus_manifest_path",
    "teacher_model_provenance_path",
    "output_dir",
    "tokenizer_id",
    "teacher_backend",
    "runtime_mode",
    "target_policy",
    "sequence_length",
    "vocab_size",
    "top_k",
    "num_buckets",
    "dynamic_top_k_min",
    "dynamic_top_k_max",
    "dynamic_mass_threshold",
    "long_tail_warning_k",
    "very_long_tail_warning_k",
    "perverse_tail_warning_k",
    "reject_perverse_exemplars",
    "primary_selected_exemplar_budget",
    "long_tail_side_board_cap",
    "perverse_tail_side_board_cap",
    "include_long_tail_in_primary",
    "include_perverse_tail_in_primary",
    "include_perverse_tail_in_student",
    "gpu_batch_size_mode",
    "gpu_batch_size_preset",
    "gpu_batch_size_custom",
    "gpu_batch_size_auto_min",
    "gpu_batch_size_auto_max",
    "shard_size_examples",
    "payload_records_per_shard",
    "max_examples",
    "resume",
    "overwrite",
    "strict_provenance",
    "fail_on_plan_warnings",
    "no_build_if_plan_warn",
    "max_artifact_bytes",
    "run_plan_path",
    "production_report_path",
    "parity_left",
    "parity_report_path",
    "run_manifest_path",
    "progress_log_path",
    "progress",
    "exemplar_delivery_path",
    "exemplar_selection_enabled",
    "exemplar_leaderboard_capacity",
    "selected_exemplar_budget",
    "selected_exemplar_fraction",
    "retain_unselected_exemplar_payloads",
    "exemplar_score_policy",
    "selected_rerun_batch_size",
    "track_delivery_timing",
    "selection_integration_policy",
    "total_selected_exemplar_budget",
    "fingerprint_corridor_budget_fraction",
    "fingerprint_corridor_budget_max",
    "fingerprint_corridor_mode_cap",
    "fingerprint_corridor_candidate_pool_cap",
    "require_full_selected_budget",
    "corridor_feature_jsonl_path",
    "global_board_supply_path",
    "c4_claims_path",
    "c5_selection_path",
    "source_passports_path",
)

_AUTHORITY_FIELDS = (
    "teacher_model",
    "tokenizer_id",
    "dataset_path",
    "corpus_manifest_path",
    "target_policy",
    "sequence_length",
    "vocab_size",
    "top_k",
    "num_buckets",
    "dynamic_top_k_min",
    "dynamic_top_k_max",
    "dynamic_mass_threshold",
    "selected_rerun_batch_size",
    "total_selected_exemplar_budget",
    "fingerprint_corridor_budget_fraction",
    "fingerprint_corridor_budget_max",
    "fingerprint_corridor_mode_cap",
    "fingerprint_corridor_candidate_pool_cap",
    "require_full_selected_budget",
    "exemplar_delivery_path",
    "selection_integration_policy",
)


def _config() -> ProductionBuildConfig:
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


def test_m5a_inventory_covers_the_current_flat_production_surface() -> None:
    assert tuple(field.name for field in fields(ProductionBuildConfig)) == (
        _PRODUCTION_CONFIG_FIELDS
    )
    assert len(_PRODUCTION_CONFIG_FIELDS) == 68


def test_m5a_pins_the_existing_selection_authority_projection() -> None:
    config = _config()

    assert _selection_integration_hash(config) == (
        "sha256:c7bdbfe538c007db6b65c7fc87850b29355dfeef5300c5bd4fc6efb178e987ab"
    )
    for field in _AUTHORITY_FIELDS:
        value = getattr(config, field)
        replacement = _changed_value(value)
        assert _selection_integration_hash(replace(config, **{field: replacement})) != (
            _selection_integration_hash(config)
        ), field


def test_m5a_non_authority_policy_does_not_change_selection_authority() -> None:
    config = _config()

    assert _selection_integration_hash(
        replace(config, long_tail_warning_k=config.long_tail_warning_k + 1)
    ) == _selection_integration_hash(config)


def test_m5a_characterizes_both_historical_cover_contracts() -> None:
    assert COVER_PAGE_VERSION == 2
    assert PACKAGE_COVER_SCHEMA == "radjax_tome_package_cover_v1"
    assert PACKAGE_COVER_SCHEMA != "radjax_tome_cover_v2"


def _changed_value(value: object) -> object:
    if isinstance(value, Path):
        return value.with_name("changed-" + value.name)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.01
    if value is None:
        return 1
    return str(value) + "-changed"
