"""Dependency-light M5 canonical build contracts.

This module deliberately defines data contracts only.  M5B does not route a
production build through them yet: ``ProductionBuildConfig`` remains the
compatibility execution boundary until the post-M5B review gate has approved
M5C.  Keeping the conversion explicit means a legacy request cannot smuggle
an opaque, mutable configuration object into the canonical representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radjax_tome.builder.teacher_textbook import TeacherTextbookBuildConfig

if TYPE_CHECKING:
    from radjax_tome.builder.production import ProductionBuildConfig


CANONICAL_BUILD_INTENT_SCHEMA = "radjax_tome_build_intent_v1"
RESOLVED_BUILD_CONFIG_SCHEMA = "radjax_tome_resolved_build_config_v1"
EXECUTION_PLAN_SCHEMA = "radjax_tome_execution_plan_v1"
SELECTION_AUTHORITY_PAYLOAD_SCHEMA = "radjax_tome_selection_authority_v1"
NORMALIZED_PRODUCTION_REQUEST_SCHEMA = "radjax_tome_normalized_production_request_v1"

PRODUCTION_PRESETS = (
    "smoke",
    "t4-1k",
    "t4-10k",
    "production-100k",
)

_C2_SCHEMA = "radjax.c2_corridor_candidate_leaderboards.v1"
_C3_SCHEMA = "radjax.c3_corridor_coverage_plan.v1"
_C4_SCHEMA = "radjax.c4_corridor_global_claims.v1"
_C5_SCHEMA = "radjax.multi_role_selected_exemplar.v1"
_DECIMAL_FRACTION_TEXT = re.compile(r"(?:0|1|0\.\d+|1\.0+)")
_TARGET_POLICIES = frozenset(
    {
        "dense",
        "dense_logits",
        "topk",
        "topk_with_tail_v0",
        "cascaded",
        "cascaded_soft_labels_v1",
        "dynamic",
        "dynamic_cascaded_soft_labels_v1",
        "corridor",
        "corridor_exemplar_v1",
    }
)


@dataclass(frozen=True)
class TeacherIntent:
    """Teacher identity and backend choices requested by the caller."""

    model: str
    tokenizer_id: str | None
    backend: str
    runtime_mode: str
    model_provenance_path: Path


@dataclass(frozen=True)
class CorpusIntent:
    """Corpus sources and deliberate semantic-size limit."""

    dataset_path: Path
    corpus_manifest_path: Path
    max_examples: int | None


@dataclass(frozen=True)
class TokenBehaviorIntent:
    """Target and token-distribution behavior."""

    target_policy: str
    sequence_length: int
    vocab_size: int
    top_k: int
    num_buckets: int
    dynamic_top_k_min: int
    dynamic_top_k_max: int
    dynamic_mass_threshold: float


@dataclass(frozen=True)
class CorridorPolicyIntent:
    """Long-tail policy that controls retained training material."""

    long_tail_warning_k: int
    very_long_tail_warning_k: int
    perverse_tail_warning_k: int
    reject_perverse_exemplars: bool
    primary_selected_exemplar_budget: int | None
    long_tail_side_board_cap: int
    perverse_tail_side_board_cap: int
    include_long_tail_in_primary: bool
    include_perverse_tail_in_primary: bool
    include_perverse_tail_in_student: bool


@dataclass(frozen=True)
class ExecutionIntent:
    """Runtime controls; these do not themselves identify a Tome."""

    gpu_batch_size_mode: str
    gpu_batch_size_preset: int
    gpu_batch_size_custom: int | None
    gpu_batch_size_auto_min: int
    gpu_batch_size_auto_max: int
    shard_size_examples: int
    resume: bool
    overwrite: bool
    strict_provenance: bool
    fail_on_plan_warnings: bool
    no_build_if_plan_warn: bool
    max_artifact_bytes: int | None
    progress: bool
    payload_records_per_shard: int = 128


@dataclass(frozen=True)
class OutputIntent:
    """Destinations and reports, isolated from semantic build intent."""

    output_dir: Path
    run_plan_path: Path | None
    production_report_path: Path | None
    parity_report_path: Path | None
    run_manifest_path: Path | None
    progress_log_path: Path | None
    parity_left: Path | None


@dataclass(frozen=True)
class SelectionIntent:
    """Exemplar delivery, rerun, and C2--C6 integration policy."""

    exemplar_delivery_path: str | None
    exemplar_selection_enabled: bool
    exemplar_leaderboard_capacity: int
    selected_exemplar_budget: int | None
    selected_exemplar_fraction: float | None
    retain_unselected_exemplar_payloads: bool
    exemplar_score_policy: str
    selected_rerun_batch_size: int | None
    track_delivery_timing: bool
    selection_integration_policy: str
    total_selected_exemplar_budget: int | None
    fingerprint_corridor_budget_fraction: str
    fingerprint_corridor_budget_max: int | None
    fingerprint_corridor_mode_cap: int
    fingerprint_corridor_candidate_pool_cap: int
    require_full_selected_budget: bool


@dataclass(frozen=True)
class CompatibilityOverrides:
    """Explicit external or parity inputs; never inferred by canonical code."""

    corridor_feature_jsonl_path: Path | None
    global_board_supply_path: Path | None
    c4_claims_path: Path | None
    c5_selection_path: Path | None
    source_passports_path: Path | None


@dataclass(frozen=True)
class PackageIntent:
    """Requested package materialization, separate from Tome identity."""

    profile: Literal["unpacked", "student", "full_debug_provenance"] = "unpacked"
    transport: Literal["directory", "rtome", "tgz"] = "directory"
    artifact_contract_version: Literal["v2", "v3"] = "v2"


@dataclass(frozen=True)
class TomeBuildIntent:
    """The typed, user-facing build request before default resolution."""

    teacher: TeacherIntent
    corpus: CorpusIntent
    behavior: TokenBehaviorIntent
    corridor_policy: CorridorPolicyIntent
    selection: SelectionIntent
    execution: ExecutionIntent
    outputs: OutputIntent
    compatibility: CompatibilityOverrides
    package: PackageIntent = PackageIntent()
    schema_version: str = CANONICAL_BUILD_INTENT_SCHEMA


@dataclass(frozen=True)
class ResolutionMetadata:
    """How a resolved request was derived, without becoming semantic input."""

    source: str
    preset_name: str | None = None
    explicit_override_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTomeBuildConfig:
    """Validated canonical configuration, still independent of I/O."""

    intent: TomeBuildIntent
    resolution: ResolutionMetadata
    schema_version: str = RESOLVED_BUILD_CONFIG_SCHEMA


@dataclass(frozen=True)
class TomeExecutionPlan:
    """Derived operational paths and batch controls for a resolved request."""

    output_dir: Path
    run_plan_path: Path
    production_report_path: Path
    parity_report_path: Path
    run_manifest_path: Path
    progress_log_path: Path
    gpu_batch_size_mode: str
    gpu_batch_size_preset: int
    gpu_batch_size_custom: int | None
    gpu_batch_size_auto_min: int
    gpu_batch_size_auto_max: int
    shard_size_examples: int
    payload_records_per_shard: int
    schema_version: str = EXECUTION_PLAN_SCHEMA


@dataclass(frozen=True)
class NormalizedProductionBuildRequest:
    """One normalized request for every supported production entry point.

    The canonical configuration is validated before the execution plan and
    protected selection-authority projection are derived.  Runtime code may
    receive a legacy-shaped execution adapter, but it cannot skip this
    normalization boundary.
    """

    resolved: ResolvedTomeBuildConfig
    execution_plan: TomeExecutionPlan
    selection_authority_payload: Mapping[str, Any]
    selection_authority_hash: str
    schema_version: str = NORMALIZED_PRODUCTION_REQUEST_SCHEMA


def adapt_legacy_production_build_config(
    config: ProductionBuildConfig,
) -> TomeBuildIntent:
    """Copy all 67 legacy fields into explicit canonical sections.

    The adapter intentionally performs no validation or I/O.  Its caller can
    therefore characterize legacy requests before M5C moves any execution
    boundary.
    """

    return TomeBuildIntent(
        teacher=TeacherIntent(
            model=config.teacher_model,
            tokenizer_id=config.tokenizer_id,
            backend=config.teacher_backend,
            runtime_mode=config.runtime_mode,
            model_provenance_path=config.teacher_model_provenance_path,
        ),
        corpus=CorpusIntent(
            dataset_path=config.dataset_path,
            corpus_manifest_path=config.corpus_manifest_path,
            max_examples=config.max_examples,
        ),
        behavior=TokenBehaviorIntent(
            target_policy=config.target_policy,
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            top_k=config.top_k,
            num_buckets=config.num_buckets,
            dynamic_top_k_min=config.dynamic_top_k_min,
            dynamic_top_k_max=config.dynamic_top_k_max,
            dynamic_mass_threshold=config.dynamic_mass_threshold,
        ),
        corridor_policy=CorridorPolicyIntent(
            long_tail_warning_k=config.long_tail_warning_k,
            very_long_tail_warning_k=config.very_long_tail_warning_k,
            perverse_tail_warning_k=config.perverse_tail_warning_k,
            reject_perverse_exemplars=config.reject_perverse_exemplars,
            primary_selected_exemplar_budget=config.primary_selected_exemplar_budget,
            long_tail_side_board_cap=config.long_tail_side_board_cap,
            perverse_tail_side_board_cap=config.perverse_tail_side_board_cap,
            include_long_tail_in_primary=config.include_long_tail_in_primary,
            include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
            include_perverse_tail_in_student=config.include_perverse_tail_in_student,
        ),
        selection=SelectionIntent(
            exemplar_delivery_path=config.exemplar_delivery_path,
            exemplar_selection_enabled=config.exemplar_selection_enabled,
            exemplar_leaderboard_capacity=config.exemplar_leaderboard_capacity,
            selected_exemplar_budget=config.selected_exemplar_budget,
            selected_exemplar_fraction=config.selected_exemplar_fraction,
            retain_unselected_exemplar_payloads=config.retain_unselected_exemplar_payloads,
            exemplar_score_policy=config.exemplar_score_policy,
            selected_rerun_batch_size=config.selected_rerun_batch_size,
            track_delivery_timing=config.track_delivery_timing,
            selection_integration_policy=config.selection_integration_policy,
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            fingerprint_corridor_budget_fraction=config.fingerprint_corridor_budget_fraction,
            fingerprint_corridor_budget_max=config.fingerprint_corridor_budget_max,
            fingerprint_corridor_mode_cap=config.fingerprint_corridor_mode_cap,
            fingerprint_corridor_candidate_pool_cap=(
                config.fingerprint_corridor_candidate_pool_cap
            ),
            require_full_selected_budget=config.require_full_selected_budget,
        ),
        execution=ExecutionIntent(
            gpu_batch_size_mode=config.gpu_batch_size_mode,
            gpu_batch_size_preset=config.gpu_batch_size_preset,
            gpu_batch_size_custom=config.gpu_batch_size_custom,
            gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
            gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
            shard_size_examples=config.shard_size_examples,
            payload_records_per_shard=config.payload_records_per_shard,
            resume=config.resume,
            overwrite=config.overwrite,
            strict_provenance=config.strict_provenance,
            fail_on_plan_warnings=config.fail_on_plan_warnings,
            no_build_if_plan_warn=config.no_build_if_plan_warn,
            max_artifact_bytes=config.max_artifact_bytes,
            progress=config.progress,
        ),
        outputs=OutputIntent(
            output_dir=config.output_dir,
            run_plan_path=config.run_plan_path,
            production_report_path=config.production_report_path,
            parity_report_path=config.parity_report_path,
            run_manifest_path=config.run_manifest_path,
            progress_log_path=config.progress_log_path,
            parity_left=config.parity_left,
        ),
        compatibility=CompatibilityOverrides(
            corridor_feature_jsonl_path=config.corridor_feature_jsonl_path,
            global_board_supply_path=config.global_board_supply_path,
            c4_claims_path=config.c4_claims_path,
            c5_selection_path=config.c5_selection_path,
            source_passports_path=config.source_passports_path,
        ),
        package=PackageIntent(artifact_contract_version=config.artifact_contract_version),
    )


def canonical_production_build_intent(
    *,
    teacher_model: str,
    dataset_path: Path,
    corpus_manifest_path: Path,
    teacher_model_provenance_path: Path,
    output_dir: Path,
    tokenizer_id: str | None = None,
) -> TomeBuildIntent:
    """Create the inspectable canonical default request for production.

    These defaults preserve the former ``production-build`` command values,
    except that canonical requests do not retain unselected payloads by
    default.  Retention remains an explicit advanced compatibility override.
    """

    return TomeBuildIntent(
        teacher=TeacherIntent(
            model=teacher_model,
            tokenizer_id=tokenizer_id,
            backend="gpu_torch",
            runtime_mode="cpu_gpu",
            model_provenance_path=teacher_model_provenance_path,
        ),
        corpus=CorpusIntent(
            dataset_path=dataset_path,
            corpus_manifest_path=corpus_manifest_path,
            max_examples=None,
        ),
        behavior=TokenBehaviorIntent(
            target_policy="corridor_exemplar_v1",
            sequence_length=16,
            vocab_size=32,
            top_k=8,
            num_buckets=4,
            dynamic_top_k_min=1,
            dynamic_top_k_max=32,
            dynamic_mass_threshold=0.95,
        ),
        corridor_policy=CorridorPolicyIntent(
            long_tail_warning_k=8_192,
            very_long_tail_warning_k=32_768,
            perverse_tail_warning_k=65_536,
            reject_perverse_exemplars=False,
            primary_selected_exemplar_budget=None,
            long_tail_side_board_cap=128,
            perverse_tail_side_board_cap=32,
            include_long_tail_in_primary=False,
            include_perverse_tail_in_primary=False,
            include_perverse_tail_in_student=False,
        ),
        selection=SelectionIntent(
            exemplar_delivery_path=None,
            exemplar_selection_enabled=False,
            exemplar_leaderboard_capacity=16,
            selected_exemplar_budget=None,
            selected_exemplar_fraction=None,
            retain_unselected_exemplar_payloads=False,
            exemplar_score_policy="entropy_top_n_v1",
            selected_rerun_batch_size=None,
            track_delivery_timing=False,
            selection_integration_policy="global_only_v1",
            total_selected_exemplar_budget=None,
            fingerprint_corridor_budget_fraction="0.50",
            fingerprint_corridor_budget_max=None,
            fingerprint_corridor_mode_cap=10,
            fingerprint_corridor_candidate_pool_cap=4,
            require_full_selected_budget=True,
        ),
        execution=ExecutionIntent(
            gpu_batch_size_mode="auto",
            gpu_batch_size_preset=8,
            gpu_batch_size_custom=None,
            gpu_batch_size_auto_min=1,
            gpu_batch_size_auto_max=64,
            shard_size_examples=1024,
            payload_records_per_shard=128,
            resume=False,
            overwrite=False,
            strict_provenance=True,
            fail_on_plan_warnings=False,
            no_build_if_plan_warn=False,
            max_artifact_bytes=None,
            progress=True,
        ),
        outputs=OutputIntent(
            output_dir=output_dir,
            run_plan_path=None,
            production_report_path=None,
            parity_report_path=None,
            run_manifest_path=None,
            progress_log_path=None,
            parity_left=None,
        ),
        compatibility=CompatibilityOverrides(
            corridor_feature_jsonl_path=None,
            global_board_supply_path=None,
            c4_claims_path=None,
            c5_selection_path=None,
            source_passports_path=None,
        ),
    )


def apply_production_preset(
    intent: TomeBuildIntent,
    preset_name: str,
) -> TomeBuildIntent:
    """Apply one named preset without changing caller-provided identities.

    ``t4-*`` presets intentionally differ only by ``corpus.max_examples``.
    Model, corpus provenance, destination, and resource choices remain caller
    intent or explicit advanced overrides rather than hidden preset state.
    """

    if preset_name not in PRODUCTION_PRESETS:
        raise ValueError(
            "unknown production preset "
            f"{preset_name!r}; expected one of {', '.join(PRODUCTION_PRESETS)}"
        )
    if preset_name == "smoke":
        return replace(
            intent,
            teacher=replace(
                intent.teacher,
                backend="cpu_reference",
                runtime_mode="cpu",
            ),
            corpus=replace(intent.corpus, max_examples=4),
            selection=replace(
                intent.selection,
                exemplar_delivery_path="two_pass_rerun_selected",
                exemplar_selection_enabled=True,
                selected_rerun_batch_size=2,
                selection_integration_policy="corridor_first_global_backfill_v1",
                total_selected_exemplar_budget=4,
                retain_unselected_exemplar_payloads=False,
            ),
            execution=replace(
                intent.execution,
                gpu_batch_size_mode="preset",
                gpu_batch_size_preset=2,
            ),
        )

    max_examples = {
        "t4-1k": 1_000,
        "t4-10k": 10_000,
        "production-100k": 100_000,
    }[preset_name]
    return replace(
        intent,
        teacher=replace(intent.teacher, backend="gpu_torch", runtime_mode="cpu_gpu"),
        corpus=replace(intent.corpus, max_examples=max_examples),
        behavior=replace(
            intent.behavior,
            target_policy="corridor_exemplar_v1",
            sequence_length=128,
            vocab_size=262_144,
            top_k=32,
            num_buckets=4,
            dynamic_top_k_min=32,
            dynamic_top_k_max=262_144,
            dynamic_mass_threshold=0.99,
        ),
        selection=replace(
            intent.selection,
            exemplar_delivery_path="two_pass_rerun_selected",
            exemplar_selection_enabled=True,
            selected_rerun_batch_size=8,
            selection_integration_policy="corridor_first_global_backfill_v1",
            total_selected_exemplar_budget=256,
            fingerprint_corridor_budget_fraction="0.50",
            fingerprint_corridor_budget_max=None,
            fingerprint_corridor_mode_cap=10,
            fingerprint_corridor_candidate_pool_cap=4,
            require_full_selected_budget=True,
            retain_unselected_exemplar_payloads=False,
        ),
        execution=replace(
            intent.execution,
            gpu_batch_size_mode="preset",
            gpu_batch_size_preset=8,
            strict_provenance=True,
        ),
    )


_PRODUCTION_OVERRIDE_SECTIONS = {
    "artifact_contract_version": ("package", "artifact_contract_version"),
    "teacher_model": ("teacher", "model"),
    "tokenizer_id": ("teacher", "tokenizer_id"),
    "teacher_backend": ("teacher", "backend"),
    "runtime_mode": ("teacher", "runtime_mode"),
    "teacher_model_provenance_path": ("teacher", "model_provenance_path"),
    "dataset_path": ("corpus", "dataset_path"),
    "corpus_manifest_path": ("corpus", "corpus_manifest_path"),
    "max_examples": ("corpus", "max_examples"),
    "target_policy": ("behavior", "target_policy"),
    "sequence_length": ("behavior", "sequence_length"),
    "vocab_size": ("behavior", "vocab_size"),
    "top_k": ("behavior", "top_k"),
    "num_buckets": ("behavior", "num_buckets"),
    "dynamic_top_k_min": ("behavior", "dynamic_top_k_min"),
    "dynamic_top_k_max": ("behavior", "dynamic_top_k_max"),
    "dynamic_mass_threshold": ("behavior", "dynamic_mass_threshold"),
    "long_tail_warning_k": ("corridor_policy", "long_tail_warning_k"),
    "very_long_tail_warning_k": ("corridor_policy", "very_long_tail_warning_k"),
    "perverse_tail_warning_k": ("corridor_policy", "perverse_tail_warning_k"),
    "reject_perverse_exemplars": ("corridor_policy", "reject_perverse_exemplars"),
    "primary_selected_exemplar_budget": (
        "corridor_policy",
        "primary_selected_exemplar_budget",
    ),
    "long_tail_side_board_cap": ("corridor_policy", "long_tail_side_board_cap"),
    "perverse_tail_side_board_cap": ("corridor_policy", "perverse_tail_side_board_cap"),
    "include_long_tail_in_primary": ("corridor_policy", "include_long_tail_in_primary"),
    "include_perverse_tail_in_primary": (
        "corridor_policy",
        "include_perverse_tail_in_primary",
    ),
    "include_perverse_tail_in_student": (
        "corridor_policy",
        "include_perverse_tail_in_student",
    ),
    "gpu_batch_size_mode": ("execution", "gpu_batch_size_mode"),
    "gpu_batch_size_preset": ("execution", "gpu_batch_size_preset"),
    "gpu_batch_size_custom": ("execution", "gpu_batch_size_custom"),
    "gpu_batch_size_auto_min": ("execution", "gpu_batch_size_auto_min"),
    "gpu_batch_size_auto_max": ("execution", "gpu_batch_size_auto_max"),
    "shard_size_examples": ("execution", "shard_size_examples"),
    "payload_records_per_shard": ("execution", "payload_records_per_shard"),
    "resume": ("execution", "resume"),
    "overwrite": ("execution", "overwrite"),
    "strict_provenance": ("execution", "strict_provenance"),
    "fail_on_plan_warnings": ("execution", "fail_on_plan_warnings"),
    "no_build_if_plan_warn": ("execution", "no_build_if_plan_warn"),
    "max_artifact_bytes": ("execution", "max_artifact_bytes"),
    "progress": ("execution", "progress"),
    "output_dir": ("outputs", "output_dir"),
    "run_plan_path": ("outputs", "run_plan_path"),
    "production_report_path": ("outputs", "production_report_path"),
    "parity_left": ("outputs", "parity_left"),
    "parity_report_path": ("outputs", "parity_report_path"),
    "run_manifest_path": ("outputs", "run_manifest_path"),
    "progress_log_path": ("outputs", "progress_log_path"),
    "exemplar_delivery_path": ("selection", "exemplar_delivery_path"),
    "exemplar_selection_enabled": ("selection", "exemplar_selection_enabled"),
    "exemplar_leaderboard_capacity": ("selection", "exemplar_leaderboard_capacity"),
    "selected_exemplar_budget": ("selection", "selected_exemplar_budget"),
    "selected_exemplar_fraction": ("selection", "selected_exemplar_fraction"),
    "retain_unselected_exemplar_payloads": (
        "selection",
        "retain_unselected_exemplar_payloads",
    ),
    "exemplar_score_policy": ("selection", "exemplar_score_policy"),
    "selected_rerun_batch_size": ("selection", "selected_rerun_batch_size"),
    "track_delivery_timing": ("selection", "track_delivery_timing"),
    "selection_integration_policy": ("selection", "selection_integration_policy"),
    "total_selected_exemplar_budget": (
        "selection",
        "total_selected_exemplar_budget",
    ),
    "fingerprint_corridor_budget_fraction": (
        "selection",
        "fingerprint_corridor_budget_fraction",
    ),
    "fingerprint_corridor_budget_max": ("selection", "fingerprint_corridor_budget_max"),
    "fingerprint_corridor_mode_cap": ("selection", "fingerprint_corridor_mode_cap"),
    "fingerprint_corridor_candidate_pool_cap": (
        "selection",
        "fingerprint_corridor_candidate_pool_cap",
    ),
    "require_full_selected_budget": ("selection", "require_full_selected_budget"),
    "corridor_feature_jsonl_path": (
        "compatibility",
        "corridor_feature_jsonl_path",
    ),
    "global_board_supply_path": ("compatibility", "global_board_supply_path"),
    "c4_claims_path": ("compatibility", "c4_claims_path"),
    "c5_selection_path": ("compatibility", "c5_selection_path"),
    "source_passports_path": ("compatibility", "source_passports_path"),
}


def apply_production_advanced_overrides(
    intent: TomeBuildIntent,
    overrides: Mapping[str, Any],
) -> TomeBuildIntent:
    """Apply explicitly supplied legacy-named overrides to canonical intent.

    The flat names remain at this compatibility seam only.  Unknown fields are
    rejected rather than being silently carried into a second resolution path.
    """

    unknown = sorted(set(overrides).difference(_PRODUCTION_OVERRIDE_SECTIONS))
    if unknown:
        raise ValueError("unknown production override fields: " + ", ".join(unknown))
    sections: dict[str, dict[str, Any]] = {}
    for legacy_name, value in overrides.items():
        section_name, field_name = _PRODUCTION_OVERRIDE_SECTIONS[legacy_name]
        sections.setdefault(section_name, {})[field_name] = value
    changes = {
        section_name: replace(getattr(intent, section_name), **field_values)
        for section_name, field_values in sections.items()
    }
    return replace(intent, **changes)


def normalize_production_build_request(
    config: ProductionBuildConfig | TomeBuildIntent | ResolvedTomeBuildConfig,
) -> NormalizedProductionBuildRequest:
    """Normalize each supported programmatic production input exactly once."""

    if isinstance(config, ResolvedTomeBuildConfig):
        errors = validate_resolved_tome_build_config(config)
        if errors:
            raise ValueError("invalid resolved Tome build config: " + "; ".join(errors))
        resolved = config
    elif isinstance(config, TomeBuildIntent):
        resolved = resolve_tome_build_intent(config)
    else:
        resolved = resolve_tome_build_intent(
            adapt_legacy_production_build_config(config),
            source="legacy_production_adapter",
        )
    execution_plan = derive_execution_plan(resolved)
    authority_payload = selection_authority_payload_v1(resolved)
    return NormalizedProductionBuildRequest(
        resolved=resolved,
        execution_plan=execution_plan,
        selection_authority_payload=authority_payload,
        selection_authority_hash=selection_authority_hash_v1(resolved),
    )


def normalize_cli_production_build_request(
    *,
    teacher_model: str,
    dataset_path: Path,
    corpus_manifest_path: Path,
    teacher_model_provenance_path: Path,
    output_dir: Path,
    tokenizer_id: str | None,
    preset_name: str | None,
    advanced_overrides: Mapping[str, Any],
) -> NormalizedProductionBuildRequest:
    """Use the sole production normalization sequence for current CLI input."""

    intent = canonical_production_build_intent(
        teacher_model=teacher_model,
        dataset_path=dataset_path,
        corpus_manifest_path=corpus_manifest_path,
        teacher_model_provenance_path=teacher_model_provenance_path,
        output_dir=output_dir,
        tokenizer_id=tokenizer_id,
    )
    if preset_name is not None:
        intent = apply_production_preset(intent, preset_name)
    intent = apply_production_advanced_overrides(intent, advanced_overrides)
    resolved = resolve_tome_build_intent(
        intent,
        source="cli_production_adapter",
        preset_name=preset_name,
        explicit_override_fields=tuple(sorted(advanced_overrides)),
    )
    return normalize_production_build_request(resolved)


def production_build_config_from_resolved(
    resolved: ResolvedTomeBuildConfig,
) -> ProductionBuildConfig:
    """Make the explicit, removable legacy execution adapter for M5C.

    The native Path-B and production stages retain their tested flat input for
    this checkpoint.  This adapter copies every legacy field and performs no
    policy resolution; therefore execution cannot change the resolved 25-field
    selection authority.
    """

    from radjax_tome.builder.production import ProductionBuildConfig

    intent = resolved.intent
    return ProductionBuildConfig(
        teacher_model=intent.teacher.model,
        tokenizer_id=intent.teacher.tokenizer_id,
        dataset_path=intent.corpus.dataset_path,
        corpus_manifest_path=intent.corpus.corpus_manifest_path,
        teacher_model_provenance_path=intent.teacher.model_provenance_path,
        output_dir=intent.outputs.output_dir,
        artifact_contract_version=intent.package.artifact_contract_version,
        teacher_backend=intent.teacher.backend,
        runtime_mode=intent.teacher.runtime_mode,
        target_policy=intent.behavior.target_policy,
        sequence_length=intent.behavior.sequence_length,
        vocab_size=intent.behavior.vocab_size,
        top_k=intent.behavior.top_k,
        num_buckets=intent.behavior.num_buckets,
        dynamic_top_k_min=intent.behavior.dynamic_top_k_min,
        dynamic_top_k_max=intent.behavior.dynamic_top_k_max,
        dynamic_mass_threshold=intent.behavior.dynamic_mass_threshold,
        long_tail_warning_k=intent.corridor_policy.long_tail_warning_k,
        very_long_tail_warning_k=intent.corridor_policy.very_long_tail_warning_k,
        perverse_tail_warning_k=intent.corridor_policy.perverse_tail_warning_k,
        reject_perverse_exemplars=intent.corridor_policy.reject_perverse_exemplars,
        primary_selected_exemplar_budget=(
            intent.corridor_policy.primary_selected_exemplar_budget
        ),
        long_tail_side_board_cap=intent.corridor_policy.long_tail_side_board_cap,
        perverse_tail_side_board_cap=(
            intent.corridor_policy.perverse_tail_side_board_cap
        ),
        include_long_tail_in_primary=(
            intent.corridor_policy.include_long_tail_in_primary
        ),
        include_perverse_tail_in_primary=(
            intent.corridor_policy.include_perverse_tail_in_primary
        ),
        include_perverse_tail_in_student=(
            intent.corridor_policy.include_perverse_tail_in_student
        ),
        gpu_batch_size_mode=intent.execution.gpu_batch_size_mode,
        gpu_batch_size_preset=intent.execution.gpu_batch_size_preset,
        gpu_batch_size_custom=intent.execution.gpu_batch_size_custom,
        gpu_batch_size_auto_min=intent.execution.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=intent.execution.gpu_batch_size_auto_max,
        shard_size_examples=intent.execution.shard_size_examples,
        payload_records_per_shard=intent.execution.payload_records_per_shard,
        max_examples=intent.corpus.max_examples,
        resume=intent.execution.resume,
        overwrite=intent.execution.overwrite,
        strict_provenance=intent.execution.strict_provenance,
        fail_on_plan_warnings=intent.execution.fail_on_plan_warnings,
        no_build_if_plan_warn=intent.execution.no_build_if_plan_warn,
        max_artifact_bytes=intent.execution.max_artifact_bytes,
        run_plan_path=intent.outputs.run_plan_path,
        production_report_path=intent.outputs.production_report_path,
        parity_left=intent.outputs.parity_left,
        parity_report_path=intent.outputs.parity_report_path,
        run_manifest_path=intent.outputs.run_manifest_path,
        progress_log_path=intent.outputs.progress_log_path,
        progress=intent.execution.progress,
        exemplar_delivery_path=intent.selection.exemplar_delivery_path,
        exemplar_selection_enabled=intent.selection.exemplar_selection_enabled,
        exemplar_leaderboard_capacity=intent.selection.exemplar_leaderboard_capacity,
        selected_exemplar_budget=intent.selection.selected_exemplar_budget,
        selected_exemplar_fraction=intent.selection.selected_exemplar_fraction,
        retain_unselected_exemplar_payloads=(
            intent.selection.retain_unselected_exemplar_payloads
        ),
        exemplar_score_policy=intent.selection.exemplar_score_policy,
        selected_rerun_batch_size=intent.selection.selected_rerun_batch_size,
        track_delivery_timing=intent.selection.track_delivery_timing,
        selection_integration_policy=intent.selection.selection_integration_policy,
        total_selected_exemplar_budget=intent.selection.total_selected_exemplar_budget,
        fingerprint_corridor_budget_fraction=(
            intent.selection.fingerprint_corridor_budget_fraction
        ),
        fingerprint_corridor_budget_max=intent.selection.fingerprint_corridor_budget_max,
        fingerprint_corridor_mode_cap=intent.selection.fingerprint_corridor_mode_cap,
        fingerprint_corridor_candidate_pool_cap=(
            intent.selection.fingerprint_corridor_candidate_pool_cap
        ),
        require_full_selected_budget=intent.selection.require_full_selected_budget,
        corridor_feature_jsonl_path=intent.compatibility.corridor_feature_jsonl_path,
        global_board_supply_path=intent.compatibility.global_board_supply_path,
        c4_claims_path=intent.compatibility.c4_claims_path,
        c5_selection_path=intent.compatibility.c5_selection_path,
        source_passports_path=intent.compatibility.source_passports_path,
    )


def resolved_tome_build_config_payload(
    normalized: NormalizedProductionBuildRequest,
) -> dict[str, Any]:
    """Return stable, JSON-ready evidence for pre-execution inspection."""

    return {
        "schema_version": NORMALIZED_PRODUCTION_REQUEST_SCHEMA,
        "resolved_config": _json_ready(asdict(normalized.resolved)),
        "execution_plan": _json_ready(asdict(normalized.execution_plan)),
        "selection_authority_payload": dict(normalized.selection_authority_payload),
        "selection_authority_hash": normalized.selection_authority_hash,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def validate_tome_build_intent(
    intent: TomeBuildIntent,
    *,
    allow_legacy_compatibility: bool = False,
) -> tuple[str, ...]:
    """Reject malformed canonical requests before any runtime initialization."""

    errors: list[str] = []
    if intent.schema_version != CANONICAL_BUILD_INTENT_SCHEMA:
        errors.append("build intent schema_version mismatch")
    for name, value, expected in (
        ("teacher", intent.teacher, TeacherIntent),
        ("corpus", intent.corpus, CorpusIntent),
        ("behavior", intent.behavior, TokenBehaviorIntent),
        ("corridor_policy", intent.corridor_policy, CorridorPolicyIntent),
        ("selection", intent.selection, SelectionIntent),
        ("execution", intent.execution, ExecutionIntent),
        ("outputs", intent.outputs, OutputIntent),
        ("compatibility", intent.compatibility, CompatibilityOverrides),
        ("package", intent.package, PackageIntent),
    ):
        if not isinstance(value, expected):
            errors.append(f"{name} has an invalid section type")
    if errors:
        return tuple(errors)
    _require_nonempty_string(errors, intent.teacher.model, "teacher.model")
    _require_optional_string(
        errors, intent.teacher.tokenizer_id, "teacher.tokenizer_id"
    )
    _require_nonempty_string(errors, intent.teacher.backend, "teacher.backend")
    _require_nonempty_string(
        errors, intent.teacher.runtime_mode, "teacher.runtime_mode"
    )
    _require_path(
        errors, intent.teacher.model_provenance_path, "teacher.model_provenance_path"
    )
    _require_path(errors, intent.corpus.dataset_path, "corpus.dataset_path")
    _require_path(
        errors, intent.corpus.corpus_manifest_path, "corpus.corpus_manifest_path"
    )
    _require_path(errors, intent.outputs.output_dir, "outputs.output_dir")
    for name, value in (
        ("outputs.run_plan_path", intent.outputs.run_plan_path),
        ("outputs.production_report_path", intent.outputs.production_report_path),
        ("outputs.parity_report_path", intent.outputs.parity_report_path),
        ("outputs.run_manifest_path", intent.outputs.run_manifest_path),
        ("outputs.progress_log_path", intent.outputs.progress_log_path),
        ("outputs.parity_left", intent.outputs.parity_left),
        (
            "compatibility.corridor_feature_jsonl_path",
            intent.compatibility.corridor_feature_jsonl_path,
        ),
        (
            "compatibility.global_board_supply_path",
            intent.compatibility.global_board_supply_path,
        ),
        ("compatibility.c4_claims_path", intent.compatibility.c4_claims_path),
        ("compatibility.c5_selection_path", intent.compatibility.c5_selection_path),
        (
            "compatibility.source_passports_path",
            intent.compatibility.source_passports_path,
        ),
    ):
        if value is not None:
            _require_path(errors, value, name)
    if intent.teacher.backend not in {"gpu_torch", "cpu_reference", "smoke_tokenizer"}:
        errors.append("teacher.backend is unsupported")
    if intent.teacher.runtime_mode not in {"cpu", "cpu_gpu"}:
        errors.append("teacher.runtime_mode is unsupported")
    if (
        intent.teacher.backend == "gpu_torch"
        and intent.teacher.runtime_mode != "cpu_gpu"
    ):
        errors.append("teacher.backend gpu_torch requires runtime_mode cpu_gpu")
    if (
        intent.teacher.backend == "cpu_reference"
        and intent.teacher.runtime_mode != "cpu"
    ):
        errors.append("teacher.backend cpu_reference requires runtime_mode cpu")
    if (
        intent.teacher.backend == "smoke_tokenizer"
        and intent.teacher.runtime_mode != "cpu"
    ):
        errors.append("teacher.backend smoke_tokenizer requires runtime_mode cpu")
    if intent.behavior.target_policy not in _TARGET_POLICIES:
        errors.append("behavior.target_policy is unsupported")
    for name, value in _integer_fields(intent):
        if not _is_int(value):
            errors.append(f"{name} must be an integer")
    for name, value in _boolean_fields(intent):
        if not isinstance(value, bool):
            errors.append(f"{name} must be boolean")
    for name, value in _optional_integer_fields(intent):
        if value is not None and not _is_int(value):
            errors.append(f"{name} must be an integer when supplied")
    if not _is_number(intent.behavior.dynamic_mass_threshold):
        errors.append("behavior.dynamic_mass_threshold must be a finite number")
    elif not 0 < intent.behavior.dynamic_mass_threshold <= 1:
        errors.append("behavior.dynamic_mass_threshold must be in (0, 1]")
    if intent.selection.selected_exemplar_fraction is not None:
        if not _is_number(intent.selection.selected_exemplar_fraction):
            errors.append(
                "selection.selected_exemplar_fraction must be a finite number"
            )
        elif not 0 < intent.selection.selected_exemplar_fraction <= 1:
            errors.append("selection.selected_exemplar_fraction must be in (0, 1]")
    _validate_fraction(errors, intent.selection.fingerprint_corridor_budget_fraction)
    if errors:
        return tuple(errors)
    if intent.behavior.sequence_length <= 0:
        errors.append("behavior.sequence_length must be positive")
    if intent.behavior.vocab_size <= 0:
        errors.append("behavior.vocab_size must be positive")
    if intent.behavior.top_k <= 0 or intent.behavior.top_k > intent.behavior.vocab_size:
        errors.append("behavior.top_k must be positive and no greater than vocab_size")
    if intent.behavior.num_buckets <= 0:
        errors.append("behavior.num_buckets must be positive")
    if intent.behavior.dynamic_top_k_min <= 0:
        errors.append("behavior.dynamic_top_k_min must be positive")
    if intent.behavior.dynamic_top_k_max < intent.behavior.dynamic_top_k_min:
        errors.append("behavior.dynamic_top_k_max must be at least dynamic_top_k_min")
    if intent.corpus.max_examples is not None and intent.corpus.max_examples <= 0:
        errors.append("corpus.max_examples must be positive when supplied")
    if intent.execution.gpu_batch_size_mode not in {"auto", "preset", "custom"}:
        errors.append("execution.gpu_batch_size_mode must be auto, preset, or custom")
    if intent.execution.gpu_batch_size_preset <= 0:
        errors.append("execution.gpu_batch_size_preset must be positive")
    if intent.execution.gpu_batch_size_auto_min <= 0:
        errors.append("execution.gpu_batch_size_auto_min must be positive")
    if (
        intent.execution.gpu_batch_size_auto_max
        < intent.execution.gpu_batch_size_auto_min
    ):
        errors.append("execution.gpu_batch_size_auto_max must be at least auto_min")
    if intent.execution.gpu_batch_size_mode == "custom":
        if (
            intent.execution.gpu_batch_size_custom is None
            or intent.execution.gpu_batch_size_custom <= 0
        ):
            errors.append(
                "execution.gpu_batch_size_custom must be positive in custom mode"
            )
    elif intent.execution.gpu_batch_size_custom is not None:
        errors.append("execution.gpu_batch_size_custom is only valid in custom mode")
    if intent.execution.shard_size_examples <= 0:
        errors.append("execution.shard_size_examples must be positive")
    if intent.execution.payload_records_per_shard <= 0:
        errors.append("execution.payload_records_per_shard must be a positive integer")
    if (
        intent.execution.max_artifact_bytes is not None
        and intent.execution.max_artifact_bytes <= 0
    ):
        errors.append("execution.max_artifact_bytes must be positive when supplied")
    if intent.selection.exemplar_leaderboard_capacity <= 0:
        errors.append("selection.exemplar_leaderboard_capacity must be positive")
    if (
        intent.selection.selected_exemplar_budget is not None
        and intent.selection.selected_exemplar_budget <= 0
    ):
        errors.append(
            "selection.selected_exemplar_budget must be positive when supplied"
        )
    if (
        intent.selection.total_selected_exemplar_budget is not None
        and intent.selection.total_selected_exemplar_budget <= 0
    ):
        errors.append(
            "selection.total_selected_exemplar_budget must be positive when supplied"
        )
    if (
        intent.selection.selected_rerun_batch_size is not None
        and intent.selection.selected_rerun_batch_size <= 0
    ):
        errors.append(
            "selection.selected_rerun_batch_size must be positive when supplied"
        )
    if intent.selection.fingerprint_corridor_mode_cap <= 0:
        errors.append("selection.fingerprint_corridor_mode_cap must be positive")
    if intent.selection.fingerprint_corridor_candidate_pool_cap <= 0:
        errors.append(
            "selection.fingerprint_corridor_candidate_pool_cap must be positive"
        )
    if (
        intent.selection.fingerprint_corridor_budget_max is not None
        and intent.selection.fingerprint_corridor_budget_max <= 0
    ):
        errors.append(
            "selection.fingerprint_corridor_budget_max must be positive when supplied"
        )
    if intent.execution.resume and intent.execution.overwrite:
        errors.append("execution.resume and execution.overwrite are mutually exclusive")
    _validate_selection_dependencies(errors, intent, allow_legacy_compatibility)
    if intent.package.profile not in {"unpacked", "student", "full_debug_provenance"}:
        errors.append("package.profile is invalid")
    if intent.package.transport not in {"directory", "rtome", "tgz"}:
        errors.append("package.transport is invalid")
    if intent.package.artifact_contract_version not in {"v2", "v3"}:
        errors.append("package.artifact_contract_version is invalid")
    return tuple(errors)


def resolve_tome_build_intent(
    intent: TomeBuildIntent,
    *,
    source: str = "canonical_request",
    preset_name: str | None = None,
    explicit_override_fields: tuple[str, ...] = (),
) -> ResolvedTomeBuildConfig:
    """Validate a request and bind only its explicit resolution provenance."""

    errors = validate_tome_build_intent(
        intent,
        allow_legacy_compatibility=source == "legacy_production_adapter",
    )
    if errors:
        raise ValueError("invalid Tome build intent: " + "; ".join(errors))
    return ResolvedTomeBuildConfig(
        intent=intent,
        resolution=ResolutionMetadata(
            source=source,
            preset_name=preset_name,
            explicit_override_fields=explicit_override_fields,
        ),
    )


def validate_resolved_tome_build_config(
    resolved: ResolvedTomeBuildConfig,
) -> tuple[str, ...]:
    """Validate resolved envelope metadata as well as its nested intent."""

    if not isinstance(resolved.resolution, ResolutionMetadata):
        return ("resolution has an invalid section type",)
    errors = list(
        validate_tome_build_intent(
            resolved.intent,
            allow_legacy_compatibility=(
                resolved.resolution.source == "legacy_production_adapter"
            ),
        )
    )
    if resolved.schema_version != RESOLVED_BUILD_CONFIG_SCHEMA:
        errors.append("resolved config schema_version mismatch")
    source = resolved.resolution.source
    if not isinstance(source, str) or not source:
        errors.append("resolution.source must be non-empty")
    if resolved.resolution.preset_name is not None and (
        not isinstance(resolved.resolution.preset_name, str)
        or resolved.resolution.preset_name not in PRODUCTION_PRESETS
    ):
        errors.append("resolution.preset_name is invalid")
    fields = resolved.resolution.explicit_override_fields
    if not isinstance(fields, tuple) or any(
        not isinstance(field, str) or field not in _PRODUCTION_OVERRIDE_SECTIONS
        for field in fields
    ):
        errors.append("resolution.explicit_override_fields are invalid")
    elif tuple(sorted(set(fields))) != fields:
        errors.append("resolution.explicit_override_fields must be sorted and unique")
    return tuple(errors)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_nonempty_string(errors: list[str], value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{name} must be a non-empty string")


def _require_optional_string(errors: list[str], value: Any, name: str) -> None:
    if value is not None:
        _require_nonempty_string(errors, value, name)


def _require_path(errors: list[str], value: Any, name: str) -> None:
    if not isinstance(value, Path):
        errors.append(f"{name} must be a Path")


def _integer_fields(intent: TomeBuildIntent) -> tuple[tuple[str, Any], ...]:
    return (
        ("behavior.sequence_length", intent.behavior.sequence_length),
        ("behavior.vocab_size", intent.behavior.vocab_size),
        ("behavior.top_k", intent.behavior.top_k),
        ("behavior.num_buckets", intent.behavior.num_buckets),
        ("behavior.dynamic_top_k_min", intent.behavior.dynamic_top_k_min),
        ("behavior.dynamic_top_k_max", intent.behavior.dynamic_top_k_max),
        (
            "corridor_policy.long_tail_warning_k",
            intent.corridor_policy.long_tail_warning_k,
        ),
        (
            "corridor_policy.very_long_tail_warning_k",
            intent.corridor_policy.very_long_tail_warning_k,
        ),
        (
            "corridor_policy.perverse_tail_warning_k",
            intent.corridor_policy.perverse_tail_warning_k,
        ),
        (
            "corridor_policy.long_tail_side_board_cap",
            intent.corridor_policy.long_tail_side_board_cap,
        ),
        (
            "corridor_policy.perverse_tail_side_board_cap",
            intent.corridor_policy.perverse_tail_side_board_cap,
        ),
        ("execution.gpu_batch_size_preset", intent.execution.gpu_batch_size_preset),
        ("execution.gpu_batch_size_auto_min", intent.execution.gpu_batch_size_auto_min),
        ("execution.gpu_batch_size_auto_max", intent.execution.gpu_batch_size_auto_max),
        ("execution.shard_size_examples", intent.execution.shard_size_examples),
        (
            "execution.payload_records_per_shard",
            intent.execution.payload_records_per_shard,
        ),
        (
            "selection.exemplar_leaderboard_capacity",
            intent.selection.exemplar_leaderboard_capacity,
        ),
        (
            "selection.fingerprint_corridor_mode_cap",
            intent.selection.fingerprint_corridor_mode_cap,
        ),
        (
            "selection.fingerprint_corridor_candidate_pool_cap",
            intent.selection.fingerprint_corridor_candidate_pool_cap,
        ),
    )


def _optional_integer_fields(intent: TomeBuildIntent) -> tuple[tuple[str, Any], ...]:
    return (
        ("corpus.max_examples", intent.corpus.max_examples),
        (
            "corridor_policy.primary_selected_exemplar_budget",
            intent.corridor_policy.primary_selected_exemplar_budget,
        ),
        ("execution.gpu_batch_size_custom", intent.execution.gpu_batch_size_custom),
        ("execution.max_artifact_bytes", intent.execution.max_artifact_bytes),
        (
            "selection.selected_exemplar_budget",
            intent.selection.selected_exemplar_budget,
        ),
        (
            "selection.selected_rerun_batch_size",
            intent.selection.selected_rerun_batch_size,
        ),
        (
            "selection.total_selected_exemplar_budget",
            intent.selection.total_selected_exemplar_budget,
        ),
        (
            "selection.fingerprint_corridor_budget_max",
            intent.selection.fingerprint_corridor_budget_max,
        ),
    )


def _boolean_fields(intent: TomeBuildIntent) -> tuple[tuple[str, Any], ...]:
    return (
        (
            "corridor_policy.reject_perverse_exemplars",
            intent.corridor_policy.reject_perverse_exemplars,
        ),
        (
            "corridor_policy.include_long_tail_in_primary",
            intent.corridor_policy.include_long_tail_in_primary,
        ),
        (
            "corridor_policy.include_perverse_tail_in_primary",
            intent.corridor_policy.include_perverse_tail_in_primary,
        ),
        (
            "corridor_policy.include_perverse_tail_in_student",
            intent.corridor_policy.include_perverse_tail_in_student,
        ),
        ("execution.resume", intent.execution.resume),
        ("execution.overwrite", intent.execution.overwrite),
        ("execution.strict_provenance", intent.execution.strict_provenance),
        ("execution.fail_on_plan_warnings", intent.execution.fail_on_plan_warnings),
        ("execution.no_build_if_plan_warn", intent.execution.no_build_if_plan_warn),
        ("execution.progress", intent.execution.progress),
        (
            "selection.exemplar_selection_enabled",
            intent.selection.exemplar_selection_enabled,
        ),
        (
            "selection.retain_unselected_exemplar_payloads",
            intent.selection.retain_unselected_exemplar_payloads,
        ),
        ("selection.track_delivery_timing", intent.selection.track_delivery_timing),
        (
            "selection.require_full_selected_budget",
            intent.selection.require_full_selected_budget,
        ),
    )


def _validate_fraction(errors: list[str], value: Any) -> None:
    if not isinstance(value, str) or _DECIMAL_FRACTION_TEXT.fullmatch(value) is None:
        errors.append(
            "selection.fingerprint_corridor_budget_fraction must be canonical "
            "decimal text"
        )
        return
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        errors.append(
            "selection.fingerprint_corridor_budget_fraction must be finite "
            "and in [0, 1]"
        )
        return
    if not decimal.is_finite() or not Decimal("0") <= decimal <= Decimal("1"):
        errors.append(
            "selection.fingerprint_corridor_budget_fraction must be finite "
            "and in [0, 1]"
        )


def _validate_selection_dependencies(
    errors: list[str], intent: TomeBuildIntent, legacy: bool
) -> None:
    selection = intent.selection
    c6 = selection.selection_integration_policy == "corridor_first_global_backfill_v1"
    if selection.selection_integration_policy not in {
        "global_only_v1",
        "corridor_first_global_backfill_v1",
    }:
        errors.append("selection.selection_integration_policy is unsupported")
    if selection.exemplar_delivery_path not in {
        None,
        "one_pass_pruned_candidate",
        "two_pass_rerun_selected",
    }:
        errors.append("selection.exemplar_delivery_path is unsupported")
    if selection.exemplar_score_policy != "entropy_top_n_v1":
        errors.append("selection.exemplar_score_policy is unsupported")
    if (
        selection.selected_exemplar_budget is not None
        and selection.selected_exemplar_fraction is not None
    ):
        errors.append(
            "selection.selected_exemplar_budget and fraction are mutually exclusive"
        )
    if not legacy and not selection.exemplar_selection_enabled:
        if any(
            (
                selection.exemplar_delivery_path is not None,
                selection.selected_exemplar_budget is not None,
                selection.selected_exemplar_fraction is not None,
                selection.selected_rerun_batch_size is not None,
                c6,
                selection.total_selected_exemplar_budget is not None,
            )
        ):
            errors.append("selection controls require exemplar_selection_enabled=true")
    if selection.exemplar_selection_enabled:
        if not legacy and intent.behavior.target_policy != "corridor_exemplar_v1":
            errors.append(
                "selection requires behavior.target_policy corridor_exemplar_v1"
            )
        if selection.exemplar_delivery_path is None:
            errors.append("selection requires exemplar_delivery_path")
    if (
        selection.selected_rerun_batch_size is not None
        and selection.exemplar_delivery_path != "two_pass_rerun_selected"
    ):
        errors.append(
            "selection.selected_rerun_batch_size requires two_pass_rerun_selected"
        )
    if c6 and not legacy:
        if (
            selection.exemplar_delivery_path != "two_pass_rerun_selected"
            or not selection.exemplar_selection_enabled
        ):
            errors.append(
                "corridor-first integration requires enabled two-pass selection"
            )
        if selection.total_selected_exemplar_budget is None:
            errors.append(
                "corridor-first integration requires total_selected_exemplar_budget"
            )
    if selection.total_selected_exemplar_budget is not None and not c6:
        errors.append(
            "total_selected_exemplar_budget requires corridor-first integration"
        )


def derive_execution_plan(resolved: ResolvedTomeBuildConfig) -> TomeExecutionPlan:
    """Derive current legacy default locations without performing filesystem I/O."""

    outputs = resolved.intent.outputs
    execution = resolved.intent.execution
    output_dir = outputs.output_dir
    return TomeExecutionPlan(
        output_dir=output_dir,
        run_plan_path=outputs.run_plan_path or output_dir / "run_plan.json",
        production_report_path=(
            outputs.production_report_path
            or output_dir / "production_build_report.json"
        ),
        parity_report_path=(
            outputs.parity_report_path or output_dir / "parity_report.json"
        ),
        run_manifest_path=(
            outputs.run_manifest_path or output_dir / "run_manifest.json"
        ),
        progress_log_path=(
            outputs.progress_log_path or output_dir / "progress_log.jsonl"
        ),
        gpu_batch_size_mode=execution.gpu_batch_size_mode,
        gpu_batch_size_preset=execution.gpu_batch_size_preset,
        gpu_batch_size_custom=execution.gpu_batch_size_custom,
        gpu_batch_size_auto_min=execution.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=execution.gpu_batch_size_auto_max,
        shard_size_examples=execution.shard_size_examples,
        payload_records_per_shard=execution.payload_records_per_shard,
    )


def selection_authority_payload_v1(
    resolved: ResolvedTomeBuildConfig,
) -> dict[str, Any]:
    """Produce the exact M1--M4 25-field selection-authority projection."""

    intent = resolved.intent
    teacher = intent.teacher
    corpus = intent.corpus
    behavior = intent.behavior
    selection = intent.selection
    return {
        "selection_integration_policy": selection.selection_integration_policy,
        "teacher_model": teacher.model,
        "tokenizer_id": teacher.tokenizer_id or teacher.model,
        "dataset_path": str(corpus.dataset_path),
        "corpus_manifest_path": str(corpus.corpus_manifest_path),
        "target_policy": behavior.target_policy,
        "sequence_length": behavior.sequence_length,
        "vocab_size": behavior.vocab_size,
        "top_k": behavior.top_k,
        "num_buckets": behavior.num_buckets,
        "dynamic_top_k_min": behavior.dynamic_top_k_min,
        "dynamic_top_k_max": behavior.dynamic_top_k_max,
        "dynamic_mass_threshold": behavior.dynamic_mass_threshold,
        "selected_rerun_batch_size": selection.selected_rerun_batch_size,
        "total_selected_exemplar_budget": selection.total_selected_exemplar_budget,
        "fingerprint_corridor_budget_fraction": (
            selection.fingerprint_corridor_budget_fraction
        ),
        "fingerprint_corridor_budget_max": selection.fingerprint_corridor_budget_max,
        "fingerprint_corridor_mode_cap": selection.fingerprint_corridor_mode_cap,
        "fingerprint_corridor_candidate_pool_cap": (
            selection.fingerprint_corridor_candidate_pool_cap
        ),
        "require_full_selected_budget": selection.require_full_selected_budget,
        "c2_schema": _C2_SCHEMA,
        "c3_schema": _C3_SCHEMA,
        "c4_schema": _C4_SCHEMA,
        "c5_schema": _C5_SCHEMA,
        "delivery_path": selection.exemplar_delivery_path,
    }


def selection_authority_hash_v1(resolved: ResolvedTomeBuildConfig) -> str:
    """Hash the unchanged projection using the historical compact JSON recipe."""

    encoded = json.dumps(
        selection_authority_payload_v1(resolved),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CANONICAL_BUILD_INTENT_SCHEMA",
    "EXECUTION_PLAN_SCHEMA",
    "NORMALIZED_PRODUCTION_REQUEST_SCHEMA",
    "PRODUCTION_PRESETS",
    "RESOLVED_BUILD_CONFIG_SCHEMA",
    "SELECTION_AUTHORITY_PAYLOAD_SCHEMA",
    "CompatibilityOverrides",
    "CorridorPolicyIntent",
    "CorpusIntent",
    "ExecutionIntent",
    "OutputIntent",
    "PackageIntent",
    "NormalizedProductionBuildRequest",
    "ResolvedTomeBuildConfig",
    "ResolutionMetadata",
    "SelectionIntent",
    "TeacherIntent",
    "TeacherTextbookBuildConfig",
    "TokenBehaviorIntent",
    "TomeBuildIntent",
    "TomeExecutionPlan",
    "adapt_legacy_production_build_config",
    "apply_production_advanced_overrides",
    "apply_production_preset",
    "canonical_production_build_intent",
    "derive_execution_plan",
    "normalize_cli_production_build_request",
    "normalize_production_build_request",
    "production_build_config_from_resolved",
    "resolved_tome_build_config_payload",
    "resolve_tome_build_intent",
    "selection_authority_hash_v1",
    "selection_authority_payload_v1",
    "validate_tome_build_intent",
    "validate_resolved_tome_build_config",
]
