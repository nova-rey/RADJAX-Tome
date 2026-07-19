from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from radjax_tome.audit import audit_selected_linkage, write_selected_linkage_audit
from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder.backend_textbook import (
    BackendTeacherTextbookBuildConfig,
    build_streaming_backend_teacher_textbook,
)
from radjax_tome.builder.c6_integration import (
    C6_SELECTION_INTEGRATION_POLICY,
    GLOBAL_ONLY_SELECTION_POLICY,
    build_corridor_coverage_report,
    c5_records_for_delivery,
    export_corridor_candidate_features,
    export_production_global_board_supply,
    export_production_source_passports,
    load_curriculum_route_records,
    validate_integrated_selection_contract,
    write_corridor_coverage_report,
)
from radjax_tome.builder.corridor_artifacts import build_corridor_artifacts
from radjax_tome.builder.exemplar_delivery import (
    ENTROPY_PARITY_QUANTIZATION_STEP,
    EXEMPLAR_DELIVERY_REPORT_FILENAME,
    ExemplarDeliveryConfig,
    SelectedExemplarDeliveryError,
    SelectedRerunCudaOOMError,
    assemble_selected_delivery_artifacts,
    finalize_selected_delivery_corridor,
    materialize_selected_exemplar_delivery,
    run_selected_delivery_rerun,
    selected_delivery_staging_diagnostic,
)
from radjax_tome.builder.exemplar_selection import build_exemplar_selection_manifest
from radjax_tome.builder.long_tail import (
    DEFAULT_LONG_TAIL_WARNING_K,
    DEFAULT_PERVERSE_TAIL_WARNING_K,
    DEFAULT_VERY_LONG_TAIL_WARNING_K,
)
from radjax_tome.builder.teacher_textbook import (
    load_text_examples,
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from radjax_tome.corpora import (
    corpus_provenance_from_manifest,
    validate_corpus_artifact,
)
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    inspect_corridor_coverage_plan,
    write_corridor_coverage_plan,
)
from radjax_tome.fingerprint.corridor_claims import (
    CorridorGlobalClaimPolicy,
    claim_corridor_then_backfill_global,
    load_global_board_input,
    write_corridor_global_claim_result,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    inspect_corridor_candidate_leaderboards,
    load_candidate_records_jsonl,
    write_corridor_candidate_leaderboards,
)
from radjax_tome.fingerprint.multi_role_selection import (
    build_multi_role_selected_exemplars,
    load_source_passports_for_coordinates,
    write_multi_role_selection_artifact,
)
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance import validate_teacher_model_provenance
from radjax_tome.reports import (
    GPURunPlanConfig,
    TomeParityConfig,
    build_gpu_run_plan,
    build_runtime_doctor_report,
    compare_tome_artifacts,
    write_gpu_run_plan,
    write_tome_parity_report,
)
from radjax_tome.targets.store import TeacherTargetStore
from radjax_tome.tome import write_cover_page

PRODUCTION_BUILD_REPORT_SCHEMA = "production_build_report_v1"
PRODUCTION_BUILD_REPORT_FILENAME = "production_build_report.json"
PRODUCTION_PROGRESS_SCHEMA = "production_progress_v1"
PRODUCTION_PROGRESS_FILENAME = "production_progress.json"


class C6BudgetShortfallError(ValueError):
    """Stops native Path B before teacher pass two with retained diagnostics."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            "C6 selected budget underfilled before selected rerun: "
            + json.dumps(diagnostic, sort_keys=True)
        )


@dataclass(frozen=True)
class ProductionBuildConfig:
    teacher_model: str
    dataset_path: Path
    corpus_manifest_path: Path
    teacher_model_provenance_path: Path
    output_dir: Path
    tokenizer_id: str | None = None
    teacher_backend: str = "gpu_torch"
    runtime_mode: str = "cpu_gpu"
    target_policy: str = "corridor_exemplar_v1"
    sequence_length: int = 16
    vocab_size: int = 32
    top_k: int = 8
    num_buckets: int = 4
    dynamic_top_k_min: int = 1
    dynamic_top_k_max: int = 32
    dynamic_mass_threshold: float = 0.95
    long_tail_warning_k: int = DEFAULT_LONG_TAIL_WARNING_K
    very_long_tail_warning_k: int = DEFAULT_VERY_LONG_TAIL_WARNING_K
    perverse_tail_warning_k: int = DEFAULT_PERVERSE_TAIL_WARNING_K
    reject_perverse_exemplars: bool = False
    primary_selected_exemplar_budget: int | None = None
    long_tail_side_board_cap: int = 128
    perverse_tail_side_board_cap: int = 32
    include_long_tail_in_primary: bool = False
    include_perverse_tail_in_primary: bool = False
    include_perverse_tail_in_student: bool = False
    gpu_batch_size_mode: str = "auto"
    gpu_batch_size_preset: int = 8
    gpu_batch_size_custom: int | None = None
    gpu_batch_size_auto_min: int = 1
    gpu_batch_size_auto_max: int = 64
    shard_size_examples: int = 1024
    max_examples: int | None = None
    resume: bool = False
    overwrite: bool = False
    strict_provenance: bool = True
    fail_on_plan_warnings: bool = False
    no_build_if_plan_warn: bool = False
    max_artifact_bytes: int | None = None
    run_plan_path: Path | None = None
    production_report_path: Path | None = None
    parity_left: Path | None = None
    parity_report_path: Path | None = None
    run_manifest_path: Path | None = None
    progress_log_path: Path | None = None
    progress: bool = False
    exemplar_delivery_path: str | None = None
    exemplar_selection_enabled: bool = False
    exemplar_leaderboard_capacity: int = 16
    selected_exemplar_budget: int | None = None
    selected_exemplar_fraction: float | None = None
    retain_unselected_exemplar_payloads: bool = True
    exemplar_score_policy: str = "entropy_top_n_v1"
    selected_rerun_batch_size: int | None = None
    track_delivery_timing: bool = False
    selection_integration_policy: str = GLOBAL_ONLY_SELECTION_POLICY
    total_selected_exemplar_budget: int | None = None
    fingerprint_corridor_budget_fraction: str = "0.50"
    fingerprint_corridor_budget_max: int | None = None
    fingerprint_corridor_mode_cap: int = 10
    fingerprint_corridor_candidate_pool_cap: int = 4
    require_full_selected_budget: bool = True
    corridor_feature_jsonl_path: Path | None = None
    global_board_supply_path: Path | None = None
    c4_claims_path: Path | None = None
    c5_selection_path: Path | None = None
    source_passports_path: Path | None = None


@dataclass(frozen=True)
class FinalizationResumeEligibility:
    eligible: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class CompatibilityMigrationResult:
    applicable: bool
    applied: bool = False
    reasons: tuple[str, ...] = ()
    from_schema: str | None = None
    payload_index_hashes_backfilled: int = 0
    payload_bodies_modified: bool = False
    teacher_work_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "applied": self.applied,
            "reasons": list(self.reasons),
            "compatibility_migration_applied": self.applied,
            "compatibility_migration_from": self.from_schema,
            "payload_index_hashes_backfilled": self.payload_index_hashes_backfilled,
            "payload_bodies_modified": self.payload_bodies_modified,
            "teacher_work_performed": self.teacher_work_performed,
        }


@dataclass
class _ProductionRunState:
    """Mutable private continuation shared by extracted production stages."""

    config: ProductionBuildConfig
    created_at: str
    production_started: float
    preflight_started: float
    report_path: Path
    run_plan_path: Path
    parity_report_path: Path
    progress: Any
    blockers: list[str]
    warnings: list[str]
    already_complete: bool
    doctor_report: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    effective_batch_size: int | None = None
    preflight_wall_seconds: float = 0.0
    build_report: Any | None = None
    main_pass_wall_seconds: float = 0.0
    terminal_report: dict[str, Any] | None = None
    native_resume_resolution: Any | None = None


def build_production_gpu_tome(config: ProductionBuildConfig) -> dict[str, Any]:
    """Build through the exact native Path-B boundary when it applies.

    M3C establishes the exact routing boundary. M4B now sends the selected
    canonical route through typed preflight and score-pass stages before the
    preserved continuation; artifact semantics remain identical for canonical
    and compatibility configurations.
    """
    from radjax_tome.builder.native_path_b import api as native_path_b_api

    canonical_config = native_path_b_api.resolve_canonical_path_b_config(config)
    if canonical_config is None:
        return _build_production_gpu_tome_compatibility(config)

    def execute_native_path_b(source_config: Any) -> dict[str, Any]:
        return _build_production_gpu_tome_compatibility(
            source_config,
            canonical_config=canonical_config,
        )

    return native_path_b_api.run_canonical_path_b(
        canonical_config,
        compatibility_executor=execute_native_path_b,
    )


def _build_production_gpu_tome_compatibility(
    config: ProductionBuildConfig,
    *,
    canonical_config: Any | None = None,
) -> dict[str, Any]:
    state = _new_production_run_state(config)
    if canonical_config is None:
        preflight_result = _run_existing_preflight(state)
        score_result = (
            _run_existing_score_pass(state)
            if preflight_result.status == "pass"
            else None
        )
    else:
        from radjax_tome.builder.native_path_b.orchestrator import (
            SliceOneOperations,
            run_preflight_then_score_pass,
        )

        slice_one = run_preflight_then_score_pass(
            canonical_config,
            operations=SliceOneOperations(
                preflight=lambda _: _run_existing_preflight(state),
                score_pass=lambda _, ready: _run_existing_score_pass(ready),
            ),
            propagate_exceptions=True,
        )
        preflight_result = slice_one.preflight
        score_result = slice_one.score_pass

    if preflight_result.status != "pass":
        return state.terminal_report or _stage_adapter_failure_report(
            state,
            preflight_result.failure,
        )
    if score_result is None or score_result.status != "pass":
        return state.terminal_report or _stage_adapter_failure_report(
            state,
            None if score_result is None else score_result.failure,
        )

    if canonical_config is not None:
        return _run_native_path_b_post_score_stages(
            state,
            canonical_config=canonical_config,
            slice_one=slice_one,
        )

    created_at = state.created_at
    production_started = state.production_started
    report_path = state.report_path
    run_plan_path = state.run_plan_path
    parity_report_path = state.parity_report_path
    progress = state.progress
    blockers = state.blockers
    warnings = state.warnings
    already_complete = state.already_complete
    doctor_report = state.doctor_report or {}
    plan = state.plan or {}
    effective_batch_size = state.effective_batch_size
    preflight_wall_seconds = state.preflight_wall_seconds
    build_report = state.build_report
    main_pass_wall_seconds = state.main_pass_wall_seconds
    c6_context: dict[str, Any] | None = None

    if config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY:
        try:
            progress.stage("fingerprint_corridor_export")
            store = TeacherTargetStore.open(config.output_dir)
            examples = load_text_examples(
                config.dataset_path,
                max_examples=store.metadata.num_examples,
            )
            build_corridor_artifacts(
                output_dir=config.output_dir,
                examples=examples,
                selected_records=[],
                selected_payloads=[],
                delivery_path=(
                    config.exemplar_delivery_path or "one_pass_pruned_candidate"
                ),
                non_selected_exemplar_payload_retained=(
                    config.retain_unselected_exemplar_payloads
                ),
            )
            progress.stage("selection_authority_export")
            authorities = _export_c6_selection_authorities(config)
            progress.memory_checkpoint("authority_export_complete")
            progress.stage("corridor_global_selection")
            c6_context = _prepare_c6_selection(config, authorities)
            progress.memory_checkpoint("c2_c5_selection_complete")
        except C6BudgetShortfallError as exc:
            blockers.append(str(exc))
            report = _production_report(
                config,
                created_at=created_at,
                completed_at=_now(),
                status="fail",
                blockers=blockers,
                warnings=warnings,
                doctor_report=doctor_report,
                run_plan_path=run_plan_path,
                run_plan=plan,
                effective_batch_size=effective_batch_size,
                already_complete=already_complete,
                parity_report_path=parity_report_path,
                parity_status="not_run",
                build_status="selection_underfilled_before_selected_rerun",
            )
            return _finalize_production_report(report, report_path, progress)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            blockers.append(f"C6 selection integration failed: {exc}")
            report = _production_report(
                config,
                created_at=created_at,
                completed_at=_now(),
                status="fail",
                blockers=blockers,
                warnings=warnings,
                doctor_report=doctor_report,
                run_plan_path=run_plan_path,
                run_plan=plan,
                effective_batch_size=effective_batch_size,
                already_complete=already_complete,
                parity_report_path=parity_report_path,
                parity_status="not_run",
            )
            return _finalize_production_report(report, report_path, progress)

    delivery_report: dict[str, Any] | None = None
    selected_delivery_failure: dict[str, Any] | None = None
    if _selected_exemplar_delivery_enabled(config):
        try:
            delivery_report = materialize_selected_exemplar_delivery(
                _exemplar_delivery_config(
                    config,
                    effective_batch_size,
                    progress_callback=progress.handle_delivery_event,
                    authoritative_records=(
                        None
                        if c6_context is None
                        else tuple(c6_context["delivery_records"])
                    ),
                    delivery_authority_hash=(
                        None
                        if c6_context is None
                        else str(
                            (c6_context.get("authorities") or {}).get(
                                "score_pass_authority_hash"
                            )
                            or ""
                        )
                    ),
                )
            )
        except SelectedExemplarDeliveryError as exc:
            selected_delivery_failure = _selected_delivery_failure_with_staging(
                config,
                c6_context,
                exc.diagnostic,
            )
            blockers.append(str(exc))
        except SelectedRerunCudaOOMError as exc:
            selected_delivery_failure = _selected_delivery_failure_with_staging(
                config,
                c6_context,
                {
                    "failure_stage": "selected_rerun",
                    "failure_reason": str(exc),
                    "delivery_path": config.exemplar_delivery_path,
                    **exc.diagnostic,
                },
            )
            blockers.append(str(exc))
        except Exception as exc:
            selected_delivery_failure = _selected_delivery_failure_with_staging(
                config,
                c6_context,
                {
                    "failure_stage": "selected_exemplar_delivery",
                    "failure_reason": str(exc),
                    "delivery_path": config.exemplar_delivery_path,
                },
            )
            blockers.append(str(exc))
    validation_status = "not_run"
    validation_wall_seconds = 0.0
    if selected_delivery_failure is None:
        progress.validation_started()
        validation_started = perf_counter()
        validation = validate_teacher_textbook(config.output_dir)
        write_teacher_textbook_validation_report(
            validation,
            config.output_dir / "validation_report.json",
        )
        validation_wall_seconds = _elapsed(validation_started)
        validation_status = validation.status
        progress.memory_checkpoint("validation_complete")
        progress.validation_completed(validation.status)
        if c6_context is not None:
            linkage_audit = audit_selected_linkage(config.output_dir, strict=True)
            write_selected_linkage_audit(
                linkage_audit,
                config.output_dir / "selected_linkage_audit.json",
            )
            c6_validation, c6_coverage = _finalize_c6_selection(
                config,
                c6_context,
                delivery_report=delivery_report,
                audit_report=linkage_audit.to_dict(),
            )
            audit_payload = linkage_audit.to_dict()
            audit_payload["c6_integration"] = {
                "status": c6_validation["status"],
                "selected_unique_count": c6_validation["selected_unique_count"],
                "selected_obligation_count": c6_validation["selected_obligation_count"],
                "coordinate_set_authority": "c5",
            }
            write_json(config.output_dir / "selected_linkage_audit.json", audit_payload)
            (config.output_dir / "reports").mkdir(parents=True, exist_ok=True)
            write_json(
                config.output_dir
                / "reports"
                / "c6_integrated_selection_validation.json",
                c6_validation,
            )
            if c6_validation["status"] == "fail":
                blockers.extend(c6_validation["blockers"])
            write_corridor_coverage_report(
                c6_coverage,
                config.output_dir / "reports" / "fingerprint_corridor_coverage.json",
            )
        if validation_status == "pass":
            try:
                write_cover_page(config.output_dir)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                blockers.append(f"cover-page generation failed: {exc}")
    else:
        # The score-pass builder may have emitted an intermediate cover.  It
        # is not a valid public surface until selected delivery completes.
        (config.output_dir / "cover_page.json").unlink(missing_ok=True)
        progress.failure(
            "selected_delivery",
            selected_delivery_failure,
        )
    parity_status = "not_run"
    if config.parity_left is not None and validation_status == "pass":
        parity_report = compare_tome_artifacts(
            config.parity_left,
            config.output_dir,
            TomeParityConfig(max_examples=config.max_examples),
            left_label="baseline",
            right_label="production",
        )
        write_tome_parity_report(parity_report, parity_report_path)
        parity_status = parity_report.status
        if parity_report.status == "fail":
            blockers.extend(parity_report.blockers)
        elif parity_report.status == "warn":
            warnings.extend(parity_report.warnings)

    if selected_delivery_failure is None and validation_status != "pass":
        blockers.extend(validation.blockers)
    if delivery_report is not None:
        if delivery_report.get("status") == "fail":
            blockers.extend(str(item) for item in delivery_report.get("blockers", ()))
        elif delivery_report.get("status") == "warn":
            warnings.extend(str(item) for item in delivery_report.get("warnings", ()))
        else:
            warnings = _filter_fulfilled_delivery_warnings(warnings)
    status = "fail" if blockers else "warn" if warnings else "pass"
    report = _production_report(
        config,
        created_at=created_at,
        completed_at=_now(),
        status=status,
        blockers=blockers,
        warnings=warnings,
        doctor_report=doctor_report,
        run_plan_path=run_plan_path,
        run_plan=plan,
        effective_batch_size=effective_batch_size,
        already_complete=already_complete,
        parity_report_path=parity_report_path,
        parity_status=parity_status,
        validation_status=validation_status,
        build_status=getattr(build_report, "status", None),
        delivery_report=delivery_report,
        selected_delivery_failure=selected_delivery_failure,
        timing=_production_timing_fields(
            config,
            started_at=created_at,
            completed_at=_now(),
            production_wall_seconds=_elapsed(production_started),
            preflight_wall_seconds=preflight_wall_seconds,
            main_pass_wall_seconds=main_pass_wall_seconds,
            validation_wall_seconds=validation_wall_seconds,
            delivery_report=delivery_report,
        ),
    )
    return _finalize_production_report(report, report_path, progress)


def _new_production_run_state(config: ProductionBuildConfig) -> _ProductionRunState:
    created_at = _now()
    production_started = perf_counter()
    preflight_started = perf_counter()
    progress = _ProductionProgressReporter(
        enabled=config.progress,
        output_dir=config.output_dir,
        path=_production_progress_path(config),
    )
    progress.start()
    return _ProductionRunState(
        config=config,
        created_at=created_at,
        production_started=production_started,
        preflight_started=preflight_started,
        report_path=_production_report_path(config),
        run_plan_path=_run_plan_path(config),
        parity_report_path=_parity_report_path(config),
        progress=progress,
        blockers=[],
        warnings=[],
        already_complete=_already_complete(config),
    )


def _run_existing_preflight(
    state: _ProductionRunState,
) -> Any:
    config = state.config
    created_at = state.created_at
    production_started = state.production_started
    report_path = state.report_path
    run_plan_path = state.run_plan_path
    parity_report_path = state.parity_report_path
    progress = state.progress
    blockers = state.blockers
    warnings = state.warnings
    already_complete = state.already_complete
    _validate_required_inputs(config, blockers)
    progress.memory_checkpoint("preflight_complete")
    if blockers:
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={},
            effective_batch_size=None,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        _record_terminal_report(state, report)
        return _terminal_stage_failure(state, "preflight")

    c6_resume_requested = (
        config.resume
        and config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
        and config.target_policy == "corridor_exemplar_v1"
        and config.exemplar_selection_enabled
        and config.exemplar_delivery_path == "two_pass_rerun_selected"
    )
    compatibility_migration = CompatibilityMigrationResult(False)
    if c6_resume_requested:
        progress.stage("compatibility_migration")
        compatibility_migration = migrate_c6_3_5_1_metadata(config)
        if compatibility_migration.applicable and not compatibility_migration.applied:
            blockers.append(
                "C6.3.5.1 metadata compatibility migration failed: "
                + "; ".join(compatibility_migration.reasons)
            )
            report = _production_report(
                config,
                created_at=created_at,
                completed_at=_now(),
                status="fail",
                blockers=blockers,
                warnings=warnings,
                doctor_report={},
                run_plan_path=run_plan_path,
                run_plan={"status": "not_run"},
                effective_batch_size=None,
                already_complete=already_complete,
                parity_report_path=parity_report_path,
                parity_status="not_run",
            )
            report["compatibility_migration"] = compatibility_migration.to_dict()
            _record_terminal_report(state, report)
            return _terminal_stage_failure(state, "preflight")
        from radjax_tome.builder.native_path_b.api import (
            resolve_canonical_path_b_config,
        )
        from radjax_tome.builder.native_path_b.resume import (
            resolve_native_path_b_resume,
        )

        canonical_config = resolve_canonical_path_b_config(config)
        state.native_resume_resolution = resolve_native_path_b_resume(
            config.output_dir,
            config=canonical_config,
            run_manifest_path=config.run_manifest_path,
        )
        if state.native_resume_resolution.complete:
            existing_report = read_json_object(report_path)
            _record_terminal_report(state, existing_report)
            return _terminal_stage_failure(state, "preflight")
    finalization_probe = probe_c6_finalization_only_resume(config)
    output_has_artifact = _has_existing_artifact(config)
    if output_has_artifact and not (config.resume or config.overwrite):
        blockers.append("output exists; use --resume or --overwrite")
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={},
            effective_batch_size=None,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        _record_terminal_report(state, report)
        return _terminal_stage_failure(state, "preflight")
    if (
        already_complete
        and not _c6_finalization_pending(config)
        and not c6_resume_requested
    ):
        validation = validate_teacher_textbook(config.output_dir)
        if validation.status == "pass":
            report = _production_report(
                config,
                created_at=created_at,
                completed_at=_now(),
                status="pass",
                blockers=[],
                warnings=[],
                doctor_report={},
                run_plan_path=run_plan_path,
                run_plan={"status": "not_run"},
                effective_batch_size=None,
                already_complete=True,
                parity_report_path=parity_report_path,
                parity_status="not_run",
                validation_status="pass",
                build_status="already_complete",
            )
            _record_terminal_report(state, report)
            return _terminal_stage_failure(state, "preflight")
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=list(validation.blockers),
            warnings=list(validation.warnings),
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={"status": "not_run"},
            effective_batch_size=None,
            already_complete=True,
            parity_report_path=parity_report_path,
            parity_status="not_run",
            validation_status=validation.status,
            build_status="already_complete_invalid",
        )
        _record_terminal_report(state, report)
        return _terminal_stage_failure(state, "preflight")
    if finalization_probe.eligible and (
        not c6_resume_requested
        or state.native_resume_resolution is None
        or state.native_resume_resolution.stage
        in {"validation_linkage", "reconciliation_cover", "final_reporting"}
    ):
        state.terminal_report = _resume_c6_finalization(
            config,
            created_at=created_at,
            production_started=production_started,
            report_path=report_path,
            parity_report_path=parity_report_path,
            progress=progress,
        )
        return _terminal_stage_failure(state, "preflight")
    if output_has_artifact and config.overwrite:
        shutil.rmtree(config.output_dir)

    backend_config = _backend_config(config)
    doctor_report = build_runtime_doctor_report(backend_config)
    plan = build_gpu_run_plan(
        GPURunPlanConfig(
            backend_config=backend_config,
            dataset_path=config.dataset_path,
            corpus_manifest_path=config.corpus_manifest_path,
            teacher_model_provenance_path=config.teacher_model_provenance_path,
            max_examples=config.max_examples,
            strict_provenance=config.strict_provenance,
            max_artifact_bytes=config.max_artifact_bytes,
            fail_on_warnings=config.fail_on_plan_warnings,
            selection_integration_policy=config.selection_integration_policy,
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            fingerprint_corridor_budget_fraction=(
                config.fingerprint_corridor_budget_fraction
            ),
            fingerprint_corridor_budget_max=config.fingerprint_corridor_budget_max,
            fingerprint_corridor_mode_cap=config.fingerprint_corridor_mode_cap,
            fingerprint_corridor_candidate_pool_cap=(
                config.fingerprint_corridor_candidate_pool_cap
            ),
            require_full_selected_budget=config.require_full_selected_budget,
        )
    )
    write_gpu_run_plan(plan, run_plan_path)
    warnings.extend(str(item) for item in plan.get("warnings", ()))
    run_plan_status = str(plan.get("status"))
    if run_plan_status == "fail":
        blockers.extend(str(item) for item in plan.get("blockers", ()))
    if run_plan_status == "warn" and config.no_build_if_plan_warn:
        blockers.append("run plan status is warn and no_build_if_plan_warn is enabled")
    effective_batch_size = _effective_batch_size(plan)
    if effective_batch_size is None:
        blockers.append("run plan did not select an effective batch size")
    if blockers:
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report=doctor_report,
            run_plan_path=run_plan_path,
            run_plan=plan,
            effective_batch_size=effective_batch_size,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        _record_terminal_report(state, report)
        return _terminal_stage_failure(state, "preflight")

    state.doctor_report = doctor_report
    state.plan = plan
    state.effective_batch_size = effective_batch_size
    state.preflight_wall_seconds = _elapsed(state.preflight_started)
    return _existing_stage_success(
        state,
        "preflight",
        paths=(run_plan_path,),
    )


def _run_existing_score_pass(
    state: _ProductionRunState,
) -> Any:
    config = state.config
    created_at = state.created_at
    run_plan_path = state.run_plan_path
    parity_report_path = state.parity_report_path
    progress = state.progress
    blockers = state.blockers
    warnings = state.warnings
    already_complete = state.already_complete
    doctor_report = state.doctor_report or {}
    plan = state.plan or {}
    effective_batch_size = state.effective_batch_size
    main_pass_started = perf_counter()
    progress.start_score_pass(
        examples_total=_planned_example_count(plan),
        shard_count_total=_planned_shard_count(config, plan),
    )
    try:
        build_report = build_streaming_backend_teacher_textbook(
            _streaming_config(
                config,
                effective_batch_size,
                progress_callback=progress.handle_streaming_event,
            )
        )
    except Exception as exc:
        blockers.append(str(exc))
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report=doctor_report,
            run_plan_path=run_plan_path,
            run_plan=plan,
            effective_batch_size=effective_batch_size,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        _record_terminal_report(state, report)
        return _terminal_stage_failure(state, "score_pass")
    main_pass_wall_seconds = _elapsed(main_pass_started)
    progress.memory_checkpoint("score_pass_complete")
    state.build_report = build_report
    state.main_pass_wall_seconds = main_pass_wall_seconds
    return _existing_stage_success(
        state,
        "score_pass",
        paths=(
            config.output_dir / "run_manifest.json",
            config.output_dir / "metadata.json",
        ),
        prior_stage="preflight",
        prior_paths=(run_plan_path,),
    )


def _existing_stage_success(
    state: _ProductionRunState,
    stage: str,
    *,
    paths: tuple[Path, ...],
    prior_stage: str | None = None,
    prior_paths: tuple[Path, ...] = (),
) -> Any:
    from radjax_tome.builder.native_path_b.contracts import (
        FileHash,
        PriorStageProof,
        StageEvidence,
        StageResult,
    )

    existing_paths = tuple(path for path in paths if path.is_file())
    hashes = tuple(
        FileHash(path=path, sha256=_file_sha256(path)) for path in existing_paths
    )
    prior_proof = None
    if prior_stage is not None:
        existing_prior_paths = tuple(path for path in prior_paths if path.is_file())
        prior_proof = PriorStageProof(
            stage=prior_stage,
            paths=existing_prior_paths,
            hashes=tuple(
                FileHash(path=path, sha256=_file_sha256(path))
                for path in existing_prior_paths
            ),
        )
    return StageResult(
        status="pass",
        value=state,
        evidence=StageEvidence(
            stage=stage,
            paths=existing_paths,
            hashes=hashes,
            counts=(),
            prior_stage_proof=prior_proof,
        ),
    )


def _record_terminal_report(
    state: _ProductionRunState,
    report: dict[str, Any],
) -> None:
    state.terminal_report = _finalize_production_report(
        report,
        state.report_path,
        state.progress,
    )


def _terminal_stage_failure(state: _ProductionRunState, stage: str) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageFailure, StageResult

    report = state.terminal_report or {}
    blockers = tuple(str(item) for item in report.get("blockers", ()))
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=stage,
            reason="existing_production_stage_terminal_report",
            blockers=(
                blockers or ("existing production stage returned terminal report",)
            ),
            resumable=bool(state.config.resume),
            remediation="inspect the preserved production report",
        ),
    )


def _stage_adapter_failure_report(
    state: _ProductionRunState,
    failure: Any,
) -> dict[str, Any]:
    blockers = list(getattr(failure, "blockers", ()) or ())
    if not blockers:
        blockers.append("native Path-B stage adapter failed without blockers")
    report = _production_report(
        state.config,
        created_at=state.created_at,
        completed_at=_now(),
        status="fail",
        blockers=blockers,
        warnings=state.warnings,
        doctor_report=state.doctor_report or {},
        run_plan_path=state.run_plan_path,
        run_plan=state.plan or {},
        effective_batch_size=state.effective_batch_size,
        already_complete=state.already_complete,
        parity_report_path=state.parity_report_path,
        parity_status="not_run",
    )
    return _finalize_production_report(report, state.report_path, state.progress)


def _selection_underfilled_stage_report(
    state: _ProductionRunState,
    failure: Any,
) -> dict[str, Any]:
    """Preserve the legacy named terminal report before selected rerun starts."""

    blockers = list(getattr(failure, "blockers", ()) or ())
    report = _production_report(
        state.config,
        created_at=state.created_at,
        completed_at=_now(),
        status="fail",
        blockers=blockers,
        warnings=state.warnings,
        doctor_report=state.doctor_report or {},
        run_plan_path=state.run_plan_path,
        run_plan=state.plan or {},
        effective_batch_size=state.effective_batch_size,
        already_complete=state.already_complete,
        parity_report_path=state.parity_report_path,
        parity_status="not_run",
        build_status="selection_underfilled_before_selected_rerun",
    )
    return _finalize_production_report(report, state.report_path, state.progress)


def _run_native_path_b_post_score_stages(
    state: _ProductionRunState,
    *,
    canonical_config: Any,
    slice_one: Any,
) -> dict[str, Any]:
    """Run the real post-score native Path-B stages in their fixed order.

    The callbacks retain ownership of the existing artifact writes.  The
    orchestrator only carries typed, in-memory evidence between them; it does
    not introduce a checkpoint schema or a second production algorithm.
    """

    from radjax_tome.builder.native_path_b.orchestrator import (
        SliceFiveOperations,
        SliceFourOperations,
        SliceThreeOperations,
        SliceTwoOperations,
        run_slice_five,
        run_slice_four,
        run_slice_three,
        run_slice_two,
    )

    slice_two = run_slice_two(
        canonical_config,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=lambda _, __: _native_early_corridor_operation(state),
            fingerprint_authority=lambda _, __: _native_fingerprint_authority_operation(
                state
            ),
            global_authority=lambda _, __, fingerprint: (
                _native_global_authority_operation(state, fingerprint)
            ),
        ),
    )
    if slice_two.status != "pass":
        return _stage_adapter_failure_report(
            state,
            slice_two.global_authority.failure
            if slice_two.global_authority is not None
            else (
                slice_two.fingerprint_authority.failure
                if slice_two.fingerprint_authority is not None
                else slice_two.early_corridor.failure
            ),
        )
    slice_three = run_slice_three(
        canonical_config,
        slice_two,
        operations=SliceThreeOperations(
            integrated_selection=lambda _, authorities: (
                _native_integrated_selection_operation(state, authorities)
            ),
        ),
    )
    if slice_three.status != "pass":
        failure = (
            None
            if slice_three.integrated_selection is None
            else slice_three.integrated_selection.failure
        )
        if failure is not None and any(
            blocker.startswith("C6 selected budget underfilled before selected rerun")
            for blocker in failure.blockers
        ):
            return _selection_underfilled_stage_report(state, failure)
        return _stage_adapter_failure_report(
            state,
            failure,
        )
    slice_four = run_slice_four(
        canonical_config,
        slice_three,
        operations=SliceFourOperations(
            selected_rerun=lambda _, inputs: _native_selected_rerun_operation(
                state, inputs
            ),
            late_corridor=lambda _, inputs: _native_late_corridor_operation(inputs),
            assembly=lambda _, inputs: _native_artifact_assembly_operation(inputs),
        ),
    )
    if slice_four.status != "pass":
        return _stage_adapter_failure_report(
            state,
            slice_four.assembly.failure
            if slice_four.assembly is not None
            else (
                slice_four.late_corridor.failure
                if slice_four.late_corridor is not None
                else (
                    slice_four.selected_rerun.failure
                    if slice_four.selected_rerun is not None
                    else None
                )
            ),
        )
    slice_five = run_slice_five(
        canonical_config,
        slice_four,
        operations=SliceFiveOperations(
            validation_linkage=lambda _, inputs: _native_validation_linkage_operation(
                state, inputs
            ),
            reconciliation_cover=lambda _, inputs: (
                _native_reconciliation_cover_operation(state, inputs)
            ),
            final_reporting=lambda _, inputs: _native_final_reporting_operation(
                state, inputs
            ),
        ),
    )
    if slice_five.final_result is not None and state.terminal_report is not None:
        return state.terminal_report
    return _stage_adapter_failure_report(
        state,
        slice_five.validation.failure if slice_five.validation is not None else None,
    )


def _native_early_corridor_operation(state: _ProductionRunState) -> Any:
    """Materialize the provisional corridor immediately after score pass."""

    from radjax_tome.builder.native_path_b.evidence import (
        read_score_surface_corridor_evidence,
    )

    config = state.config
    state.progress.stage("fingerprint_corridor_export")
    store = TeacherTargetStore.open(config.output_dir)
    examples = load_text_examples(
        config.dataset_path,
        max_examples=store.metadata.num_examples,
    )
    build_corridor_artifacts(
        output_dir=config.output_dir,
        examples=examples,
        selected_records=[],
        selected_payloads=[],
        delivery_path=config.exemplar_delivery_path or "one_pass_pruned_candidate",
        non_selected_exemplar_payload_retained=(
            config.retain_unselected_exemplar_payloads
        ),
    )
    return read_score_surface_corridor_evidence(config.output_dir)


def _native_fingerprint_authority_operation(state: _ProductionRunState) -> Any:
    """Write selector/features bound to the passing provisional corridor."""

    from radjax_tome.builder.native_path_b.contracts import StageResult

    config = state.config
    state.progress.stage("selection_authority_export")
    fingerprint = _export_c6_fingerprint_selection_authority(config)
    paths = (
        Path(str(fingerprint["selector_path"])),
        Path(str(fingerprint["feature_path"])),
        config.output_dir / "c6" / "corridor-features" / "manifest.json",
    )
    evidence = _native_file_evidence(
        "fingerprint_corridor_selection_authority_export",
        paths,
    )
    return StageResult(
        status="pass",
        value=fingerprint,
        evidence=evidence,
    )


def _native_global_authority_operation(
    state: _ProductionRunState,
    fingerprint: Mapping[str, Path | str],
) -> Any:
    """Write global supply/passports only after matching fingerprint authority."""

    from radjax_tome.builder.native_path_b.contracts import StageResult

    authorities = _export_c6_global_authority(state.config, fingerprint)
    paths = (
        Path(str(authorities["global_board_supply_path"])),
        Path(str(authorities["source_passports_path"])),
        Path(str(authorities["authority_manifest_path"])),
    )
    evidence = _native_file_evidence("global_authority_export", paths)
    state.progress.memory_checkpoint("authority_export_complete")
    return StageResult(status="pass", value=authorities, evidence=evidence)


def _native_integrated_selection_operation(
    state: _ProductionRunState, inputs: Any
) -> Any:
    """Run C2--C5 from the two explicit authority handoffs."""

    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.selection import IntegratedSelectionHandoff

    state.progress.stage("corridor_global_selection")
    context = _prepare_c6_selection(state.config, inputs.global_value)
    root = state.config.output_dir / "c6"
    c2 = _native_file_evidence(
        "c2_corridor_candidate_leaderboards",
        (root / "corridor-leaderboards" / "manifest.json",),
        prior=inputs.fingerprint_evidence,
    )
    c3 = _native_file_evidence(
        "c3_corridor_coverage_plan",
        (root / "coverage-plan" / "coverage_plan.json",),
        prior=c2,
    )
    c4 = _native_file_evidence(
        "c4_corridor_global_claims",
        (root / "claims" / "claim_manifest.json",),
        prior=c3,
    )
    c5 = _native_file_evidence(
        "c5_multi_role_selection",
        (root / "multi-role-selection" / "manifest.json",),
        prior=c4,
    )
    evidence = _native_file_evidence(
        "integrated_selection",
        (
            root / "selection_budget_diagnostics.json",
            root / "multi-role-selection" / "manifest.json",
        ),
        prior=c5,
    )
    state.progress.memory_checkpoint("c2_c5_selection_complete")
    return StageResult(
        status="pass",
        value=IntegratedSelectionHandoff(
            value=context,
            stage_evidence=evidence,
            c2_evidence=c2,
            c3_evidence=c3,
            c4_evidence=c4,
            c5_evidence=c5,
        ),
        evidence=evidence,
    )


def _native_selected_rerun_operation(state: _ProductionRunState, inputs: Any) -> Any:
    """Run the selected-only teacher pass without promoting public payloads."""

    from radjax_tome.builder.native_path_b.contracts import EvidenceCount, StageResult
    from radjax_tome.builder.native_path_b.delivery import SelectedRerunHandoff

    context = inputs.selection
    config = _exemplar_delivery_config(
        state.config,
        state.effective_batch_size,
        progress_callback=state.progress.handle_delivery_event,
        authoritative_records=tuple(context["delivery_records"]),
        delivery_authority_hash=str(
            (context.get("authorities") or {}).get("score_pass_authority_hash") or ""
        ),
    )
    prepared = run_selected_delivery_rerun(config)
    staging_root = (
        config.artifact_dir
        / ".staging-native-c6"
        / (str(config.delivery_authority_hash or "unbound").replace(":", "-"))
    )
    staging_paths = tuple(
        sorted(path for path in staging_root.rglob("*") if path.is_file())
    )
    if not staging_paths:
        raise ValueError("selected rerun produced no native staged payload evidence")
    evidence = _native_file_evidence(
        "selected_delivery_rerun",
        staging_paths,
        counts=(
            EvidenceCount("selected_record_count", len(prepared.selected_records)),
            EvidenceCount("selected_payload_count", len(prepared.selected_payloads)),
            EvidenceCount(
                "selected_rerun_example_count", prepared.rerun_selected_example_count
            ),
        ),
        prior=inputs.c5_evidence,
    )
    return StageResult(
        status="pass",
        value=SelectedRerunHandoff(
            value={"prepared": prepared, "context": context},
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )


def _native_late_corridor_operation(inputs: Any) -> Any:
    """Overwrite the provisional public corridors only after selected rerun."""

    from radjax_tome.builder.native_path_b.contracts import (
        SelectedArtifactCorridorEvidence,
        StageResult,
    )

    rerun = inputs.selected_rerun
    prepared = rerun["prepared"]
    finalized = finalize_selected_delivery_corridor(prepared)
    # Slice Four deliberately keeps one selected-rerun handoff across late
    # finalization and assembly.  Update only that ephemeral callback value so
    # assembly consumes the finalized context rather than recreating a stage.
    rerun["prepared"] = finalized
    corridor = finalized.corridor_result
    if corridor is None:
        raise ValueError("late corridor finalization returned no corridor evidence")
    output = finalized.config.artifact_dir
    evidence = _native_file_evidence(
        "selected_artifact_corridor_finalization",
        (
            corridor.summary_path,
            corridor.fingerprints_path,
            corridor.modes_path,
            corridor.assignments_path,
        ),
        prior=inputs.selected_rerun_evidence,
    )
    return StageResult(
        status="pass",
        value=SelectedArtifactCorridorEvidence(
            stage_evidence=evidence,
            summary_path=corridor.summary_path,
            fingerprints_path=corridor.fingerprints_path,
            modes_path=corridor.modes_path,
            assignments_path=corridor.assignments_path,
            positions_available=corridor.positions_available,
            positions_used=corridor.positions_used,
            fingerprint_count=corridor.fingerprint_count,
            mode_count=corridor.mode_count,
            assignment_count=corridor.assignment_count,
            selected_exemplar_count=len(finalized.selected_payloads),
            selected_exemplars_linked=corridor.selected_exemplars_linked,
            # Assembly owns the durable delivery report; retain its established
            # destination as the subsequent proof path without writing early.
            delivery_report_path=output / "delivery_report.json",
            authority_manifest_path=output / "c6" / "authority_manifest.json",
            delivery_authority_hash=str(finalized.config.delivery_authority_hash or ""),
        ),
        evidence=evidence,
    )


def _native_artifact_assembly_operation(inputs: Any) -> Any:
    """Promote payloads and write the established delivery artifact surface."""

    from radjax_tome.builder.native_path_b.assembly import ArtifactAssemblyHandoff
    from radjax_tome.builder.native_path_b.contracts import EvidenceCount, StageResult

    rerun = inputs.selected_rerun
    finalized = rerun["prepared"]
    report = assemble_selected_delivery_artifacts(finalized)
    output = finalized.config.artifact_dir
    evidence = _native_file_evidence(
        "artifact_assembly",
        (
            output / "delivery_report.json",
            output / "leaderboards" / "selected_exemplars.json",
            output / "selected_exemplars" / "payload_index.json",
        ),
        counts=(
            EvidenceCount(
                "selected_exemplar_count", int(report["num_selected_exemplars"])
            ),
        ),
        prior=inputs.final_corridor_evidence,
    )
    return StageResult(
        status="pass",
        value=ArtifactAssemblyHandoff(
            value={
                "delivery_report": report,
                "context": rerun["context"],
                "prepared": finalized,
            },
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )


def _native_validation_linkage_operation(
    state: _ProductionRunState, inputs: Any
) -> Any:
    """Run the existing strict validation and selected-linkage audit."""

    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.verification import ValidationLinkageHandoff

    output = state.config.output_dir
    state.progress.validation_started()
    validation_started = perf_counter()
    validation = validate_teacher_textbook(output)
    write_teacher_textbook_validation_report(
        validation, output / "validation_report.json"
    )
    validation_wall_seconds = _elapsed(validation_started)
    state.progress.memory_checkpoint("validation_complete")
    state.progress.validation_completed(validation.status)
    linkage_audit = audit_selected_linkage(output, strict=True)
    write_selected_linkage_audit(linkage_audit, output / "selected_linkage_audit.json")
    evidence = _native_file_evidence(
        "validation_linkage",
        (output / "validation_report.json", output / "selected_linkage_audit.json"),
        prior=inputs.assembly_evidence,
    )
    return StageResult(
        status="pass",
        value=ValidationLinkageHandoff(
            value={
                **inputs.assembly,
                "validation": validation,
                "linkage_audit": linkage_audit,
                "validation_wall_seconds": validation_wall_seconds,
            },
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )


def _native_reconciliation_cover_operation(
    state: _ProductionRunState, inputs: Any
) -> Any:
    """Reconcile C2--C5 delivery proof, coverage, and cover page."""

    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.verification import (
        ReconciliationCoverHandoff,
    )

    output = state.config.output_dir
    value = inputs.validation
    c6_validation, c6_coverage = _finalize_c6_selection(
        state.config,
        value["context"],
        delivery_report=value["delivery_report"],
        audit_report=value["linkage_audit"].to_dict(),
    )
    audit_payload = value["linkage_audit"].to_dict()
    audit_payload["c6_integration"] = {
        "status": c6_validation["status"],
        "selected_unique_count": c6_validation["selected_unique_count"],
        "selected_obligation_count": c6_validation["selected_obligation_count"],
        "coordinate_set_authority": "c5",
    }
    write_json(output / "selected_linkage_audit.json", audit_payload)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    write_json(
        output / "reports" / "c6_integrated_selection_validation.json",
        c6_validation,
    )
    write_corridor_coverage_report(
        c6_coverage,
        output / "reports" / "fingerprint_corridor_coverage.json",
    )
    if c6_validation["status"] == "fail":
        state.blockers.extend(str(item) for item in c6_validation["blockers"])
    if value["validation"].status == "pass":
        write_cover_page(output)
    evidence = _native_file_evidence(
        "reconciliation_cover",
        (
            output / "reports" / "c6_integrated_selection_validation.json",
            output / "reports" / "fingerprint_corridor_coverage.json",
            output / "cover_page.json",
        ),
        prior=inputs.validation_evidence,
    )
    return StageResult(
        status="pass",
        value=ReconciliationCoverHandoff(
            value={**value, "c6_validation": c6_validation},
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )


def _native_final_reporting_operation(state: _ProductionRunState, inputs: Any) -> Any:
    """Render the existing report/progress result from completed native proof."""

    from radjax_tome.builder.native_path_b.contracts import (
        NativePathBRunResult,
        StageFailure,
    )

    value = inputs.reconciliation
    config = state.config
    validation = value["validation"]
    delivery_report = value["delivery_report"]
    parity_status = "not_run"
    if config.parity_left is not None and validation.status == "pass":
        parity_report = compare_tome_artifacts(
            config.parity_left,
            config.output_dir,
            TomeParityConfig(max_examples=config.max_examples),
            left_label="baseline",
            right_label="production",
        )
        write_tome_parity_report(parity_report, state.parity_report_path)
        parity_status = parity_report.status
        if parity_report.status == "fail":
            state.blockers.extend(parity_report.blockers)
        elif parity_report.status == "warn":
            state.warnings.extend(parity_report.warnings)
    if validation.status != "pass":
        state.blockers.extend(validation.blockers)
    if delivery_report.get("status") == "fail":
        state.blockers.extend(str(item) for item in delivery_report.get("blockers", ()))
    elif delivery_report.get("status") == "warn":
        state.warnings.extend(str(item) for item in delivery_report.get("warnings", ()))
    else:
        state.warnings = _filter_fulfilled_delivery_warnings(state.warnings)
    status = "fail" if state.blockers else "warn" if state.warnings else "pass"
    report = _production_report(
        config,
        created_at=state.created_at,
        completed_at=_now(),
        status=status,
        blockers=state.blockers,
        warnings=state.warnings,
        doctor_report=state.doctor_report or {},
        run_plan_path=state.run_plan_path,
        run_plan=state.plan or {},
        effective_batch_size=state.effective_batch_size,
        already_complete=state.already_complete,
        parity_report_path=state.parity_report_path,
        parity_status=parity_status,
        validation_status=validation.status,
        build_status=getattr(state.build_report, "status", None),
        delivery_report=delivery_report,
        selected_delivery_failure=None,
        timing=_production_timing_fields(
            config,
            started_at=state.created_at,
            completed_at=_now(),
            production_wall_seconds=_elapsed(state.production_started),
            preflight_wall_seconds=state.preflight_wall_seconds,
            main_pass_wall_seconds=state.main_pass_wall_seconds,
            validation_wall_seconds=float(value["validation_wall_seconds"]),
            delivery_report=delivery_report,
        ),
    )
    state.terminal_report = _finalize_production_report(
        report,
        state.report_path,
        state.progress,
    )
    evidence = _native_file_evidence(
        "final_reporting",
        (state.report_path, config.output_dir / "production_progress.json"),
        prior=inputs.reconciliation_evidence,
    )
    if status == "fail":
        return NativePathBRunResult(
            status="fail",
            production_report_path=state.report_path,
            validation_report_path=config.output_dir / "validation_report.json",
            evidence=None,
            failure=StageFailure(
                stage="final_reporting",
                reason="existing_production_terminal_blockers",
                blockers=tuple(state.blockers),
                resumable=bool(config.resume),
                remediation="inspect the preserved production report",
            ),
        )
    return NativePathBRunResult(
        status="pass",
        production_report_path=state.report_path,
        validation_report_path=config.output_dir / "validation_report.json",
        evidence=evidence,
    )


def _native_file_evidence(
    stage: str,
    paths: tuple[Path, ...],
    *,
    counts: tuple[Any, ...] = (),
    prior: Any | None = None,
) -> Any:
    """Hash existing files as immutable typed evidence for one native stage."""

    from radjax_tome.builder.native_path_b.contracts import (
        FileHash,
        PriorStageProof,
        StageEvidence,
    )

    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise ValueError(
            "native Path-B stage evidence is missing: "
            + ", ".join(str(path) for path in missing)
        )
    prior_proof = None
    if prior is not None:
        prior_proof = PriorStageProof(
            stage=prior.stage,
            paths=prior.paths,
            hashes=prior.hashes,
            counts=prior.counts,
        )
    return StageEvidence(
        stage=stage,
        paths=paths,
        hashes=tuple(FileHash(path=path, sha256=_file_sha256(path)) for path in paths),
        counts=counts,
        prior_stage_proof=prior_proof,
    )


def write_production_build_report(report: dict[str, Any], path: Path) -> None:
    write_json(path, report)


def _finalize_production_report(
    report: dict[str, Any],
    path: Path,
    progress: _ProductionProgressReporter,
) -> dict[str, Any]:
    progress.memory_checkpoint("finalization")
    report["phase_host_memory_bytes"] = dict(progress.memory_points)
    progress.report_writing_started()
    write_production_build_report(report, path)
    progress.complete(str(report.get("status") or "unknown"))
    return report


def render_production_build_summary(report: dict[str, Any]) -> list[str]:
    return [
        (
            f"status={report.get('status')} output={report.get('output_dir')} "
            f"effective_batch_size={report.get('effective_batch_size')}"
        ),
        f"run_plan_status={report.get('run_plan_status')}",
        f"validation_status={report.get('validation_status')}",
        f"parity_status={report.get('parity_status')}",
        f"already_complete={str(report.get('already_complete')).lower()}",
        f"warnings={len(report.get('warnings', ()) or ())}",
        f"blockers={len(report.get('blockers', ()) or ())}",
    ]


class _ProductionProgressReporter:
    def __init__(self, *, enabled: bool, output_dir: Path, path: Path) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.path = path
        self.started_at = perf_counter()
        self.started_at_iso = _now()
        self.score_started_at = self.started_at
        self.selected_started_at = self.started_at
        self.last_emit_at = 0.0
        self.last_score_examples = 0
        self.score_emit_interval_examples = 1
        self.memory_points: dict[str, int] = {}
        self.payload: dict[str, Any] = {
            "schema_version": PRODUCTION_PROGRESS_SCHEMA,
            "status": "running",
            "phase": "preflight",
            "created_at": self.started_at_iso,
            "updated_at": self.started_at_iso,
            "output_dir": str(output_dir),
        }

    def start(self) -> None:
        if self.path.is_file():
            try:
                previous = read_json_object(self.path)
            except (OSError, ValueError):
                previous = {}
            if previous.get("status") == "running":
                self.payload["interrupted_previous_phase"] = previous.get("phase")
                self.payload["previous_progress_status"] = "stale_running"
        if not self.enabled:
            return
        self._write()
        self._emit("phase=preflight status=running", force=True)

    def memory_checkpoint(self, phase: str) -> None:
        self.memory_points[phase] = _host_rss_bytes()
        if self.enabled:
            self.payload["phase_host_memory_bytes"] = dict(self.memory_points)
            self._write()

    def start_score_pass(
        self,
        *,
        examples_total: int | None,
        shard_count_total: int | None,
    ) -> None:
        if not self.enabled:
            return
        self.score_started_at = perf_counter()
        total = int(examples_total or 0)
        self.score_emit_interval_examples = max(1, min(1_000, total or 1))
        self.payload["phase"] = "score_pass"
        self.payload["score_pass"] = {
            "status": "running",
            "examples_processed": 0,
            "examples_total": total,
            "examples_per_second": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "shard_count_written": 0,
            "shard_count_total": shard_count_total,
        }
        self._write()
        self._emit_score(force=True)

    def stage(self, phase: str) -> None:
        """Publish a lightweight orchestration checkpoint to the sidecar."""

        if not self.enabled:
            return
        self.payload["phase"] = phase
        self.payload[phase] = {
            "status": "running",
            "elapsed_seconds": round(_elapsed(self.started_at), 3),
        }
        self._write()
        self._emit(f"phase={phase} status=running", force=True)

    def handle_streaming_event(self, event: dict[str, object]) -> None:
        if not self.enabled:
            return
        event_name = str(event.get("event") or "")
        if event_name in {"shard_completed", "shard_skipped_existing"}:
            score = dict(self.payload.get("score_pass") or {})
            processed = int(
                event.get("example_end_index_exclusive")
                or score.get("examples_processed")
                or 0
            )
            shard_count = int(score.get("shard_count_written") or 0) + 1
            self._update_score_pass(
                examples_processed=processed,
                shard_count_written=shard_count,
                status="running",
            )
        elif event_name == "run_completed":
            score = dict(self.payload.get("score_pass") or {})
            total = int(score.get("examples_total") or 0)
            processed = total or int(score.get("examples_processed") or 0)
            self._update_score_pass(
                examples_processed=processed,
                shard_count_written=int(score.get("shard_count_written") or 0),
                status="complete",
                force=True,
            )
        elif event_name == "run_failed":
            self.payload["status"] = "failed"
            self.payload["phase"] = "score_pass"
            self.payload["message"] = event.get("message")
            self._write()
            self._emit("phase=score_pass status=failed", force=True)

    def handle_delivery_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        phase = str(event.get("phase") or "")
        if phase == "selected_rerun":
            self._update_selected_rerun(event)
        elif phase == "corridor_export":
            self._update_corridor_export(event)

    def validation_started(self) -> None:
        if not self.enabled:
            return
        self.payload["phase"] = "validation"
        self.payload["validation"] = {
            "status": "running",
            "elapsed_seconds": round(_elapsed(self.started_at), 3),
        }
        self._write()
        self._emit("phase=validation status=running", force=True)

    def validation_completed(self, validation_status: str) -> None:
        if not self.enabled:
            return
        self.payload["phase"] = "validation"
        self.payload["validation"] = {
            "status": "complete",
            "validation_status": validation_status,
            "elapsed_seconds": round(_elapsed(self.started_at), 3),
        }
        self._write()
        self._emit(
            f"phase=validation status=complete validation_status={validation_status}",
            force=True,
        )

    def failure(self, phase: str, diagnostic: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self.payload["status"] = "failed"
        self.payload["phase"] = phase
        self.payload["failure"] = dict(diagnostic)
        self._write()
        self._emit(f"phase={phase} status=failed", force=True)

    def report_writing_started(self) -> None:
        if not self.enabled:
            return
        self.payload["phase"] = "report_writing"
        self.payload["report_writing"] = {
            "status": "running",
            "elapsed_seconds": round(_elapsed(self.started_at), 3),
        }
        self._write()
        self._emit("phase=report_writing status=running", force=True)

    def complete(self, production_status: str) -> None:
        if not self.enabled:
            return
        progress_status = (
            "complete" if production_status in {"pass", "warn"} else "failed"
        )
        if isinstance(self.payload.get("report_writing"), dict):
            report_writing = dict(self.payload["report_writing"])
            report_writing["status"] = "complete"
            report_writing["elapsed_seconds"] = round(_elapsed(self.started_at), 3)
            self.payload["report_writing"] = report_writing
        self.payload.update(
            {
                "status": progress_status,
                "phase": "complete" if progress_status == "complete" else "failed",
                "production_status": production_status,
                "completed_at": _now(),
                "elapsed_seconds": round(_elapsed(self.started_at), 3),
            }
        )
        self._write()
        self._emit(
            "phase="
            f"{self.payload['phase']} status={progress_status} "
            f"production_status={production_status}",
            force=True,
        )

    def _update_score_pass(
        self,
        *,
        examples_processed: int,
        shard_count_written: int,
        status: str,
        force: bool = False,
    ) -> None:
        score = dict(self.payload.get("score_pass") or {})
        total = int(score.get("examples_total") or 0)
        elapsed = _elapsed(self.score_started_at)
        eps = _rate(examples_processed, elapsed)
        score.update(
            {
                "status": status,
                "examples_processed": examples_processed,
                "examples_total": total,
                "examples_per_second": eps,
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": _eta(total - examples_processed, eps),
                "shard_count_written": shard_count_written,
            }
        )
        self.payload["phase"] = "score_pass"
        self.payload["score_pass"] = score
        self._write()
        examples_delta = examples_processed - self.last_score_examples
        if force or examples_delta >= self.score_emit_interval_examples:
            self.last_score_examples = examples_processed
            self._emit_score(force=True)

    def _update_selected_rerun(self, event: dict[str, Any]) -> None:
        total = int(event.get("selected_examples_total") or 0)
        processed = int(event.get("selected_examples_processed") or 0)
        if str(event.get("event")) == "started":
            self.selected_started_at = perf_counter()
        elapsed = _elapsed(self.selected_started_at)
        rps = _rate(processed, elapsed)
        self.payload["phase"] = "selected_rerun"
        self.payload["selected_rerun"] = {
            "status": "complete" if total and processed >= total else "running",
            "selected_examples_processed": processed,
            "selected_examples_total": total,
            "selected_coordinates_committed": int(
                event.get("selected_coordinates_committed") or 0
            ),
            "selected_coordinates_total": int(
                event.get("selected_coordinates_total") or 0
            ),
            "reruns_per_second": rps,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": _eta(total - processed, rps),
        }
        self._write()
        self._emit_selected(force=True)

    def _update_corridor_export(self, event: dict[str, Any]) -> None:
        self.payload["phase"] = "corridor_export"
        self.payload["corridor_export"] = {
            "status": (
                "complete"
                if str(event.get("event")) == "assignments_written"
                else "running"
            ),
            "positions_processed": int(event.get("positions_processed") or 0),
            "positions_total": int(event.get("positions_total") or 0),
            "modes_discovered": int(event.get("modes_discovered") or 0),
            "fingerprints_discovered": int(event.get("fingerprints_discovered") or 0),
            "assignment_storage_kind": event.get("assignment_storage_kind"),
            "elapsed_seconds": round(_elapsed(self.started_at), 3),
        }
        self._write()
        self._emit_corridor(force=True)

    def _emit_score(self, *, force: bool = False) -> None:
        score = dict(self.payload.get("score_pass") or {})
        self._emit(
            "phase=score_pass "
            f"examples={score.get('examples_processed', 0)}/"
            f"{score.get('examples_total', 0)} "
            f"eps={float(score.get('examples_per_second') or 0.0):.1f} "
            f"elapsed={float(score.get('elapsed_seconds') or 0.0):.1f} "
            f"eta={_format_eta(score.get('eta_seconds'))} "
            f"shards={score.get('shard_count_written', 0)}",
            force=force,
        )

    def _emit_selected(self, *, force: bool = False) -> None:
        selected = dict(self.payload.get("selected_rerun") or {})
        self._emit(
            "phase=selected_rerun "
            f"selected_examples={selected.get('selected_examples_processed', 0)}/"
            f"{selected.get('selected_examples_total', 0)} "
            f"selected_coordinates={selected.get('selected_coordinates_committed', 0)}/"
            f"{selected.get('selected_coordinates_total', 0)} "
            f"reruns_per_second={float(selected.get('reruns_per_second') or 0.0):.1f} "
            f"elapsed={float(selected.get('elapsed_seconds') or 0.0):.1f} "
            f"eta={_format_eta(selected.get('eta_seconds'))}",
            force=force,
        )

    def _emit_corridor(self, *, force: bool = False) -> None:
        corridor = dict(self.payload.get("corridor_export") or {})
        self._emit(
            "phase=corridor_export "
            f"positions={corridor.get('positions_processed', 0)}/"
            f"{corridor.get('positions_total', 0)} "
            f"modes={corridor.get('modes_discovered', 0)} "
            f"fingerprints={corridor.get('fingerprints_discovered', 0)} "
            "assignment_storage_kind="
            f"{corridor.get('assignment_storage_kind')}",
            force=force,
        )

    def _emit(self, line: str, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        if not force and now - self.last_emit_at < 5.0:
            return
        self.last_emit_at = now
        print(line, flush=True)

    def _write(self) -> None:
        self.payload["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


def _validate_required_inputs(
    config: ProductionBuildConfig,
    blockers: list[str],
) -> None:
    if config.selection_integration_policy not in {
        GLOBAL_ONLY_SELECTION_POLICY,
        C6_SELECTION_INTEGRATION_POLICY,
    }:
        blockers.append(
            "unsupported selection_integration_policy: "
            f"{config.selection_integration_policy}"
        )
    if config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY:
        if config.total_selected_exemplar_budget is None:
            blockers.append("C6 requires total_selected_exemplar_budget")
        # Normal C6.2 production derives global ranked supply and passports
        # from this run's score surface.  Supplied files are only fail-closed
        # checkpoints and are compared after Stage 2 has an authority hash.
        for label, path in (
            ("source passports", config.source_passports_path),
            ("global board supply", config.global_board_supply_path),
        ):
            if path is not None and not path.is_file():
                blockers.append(f"{label} override path missing: {path}")
        if config.corridor_feature_jsonl_path is not None:
            blockers.append(
                "C6 derives strict corridor features from the current packed "
                "corridor artifact; --corridor-feature-jsonl is not accepted"
            )
        if config.c4_claims_path is not None or config.c5_selection_path is not None:
            blockers.append(
                "C6 production rebuilds C4/C5 from the current artifact; "
                "external C4/C5 checkpoints are not accepted"
            )
    if (
        config.exemplar_selection_enabled
        and config.target_policy != "corridor_exemplar_v1"
    ):
        blockers.append(
            "selected exemplar delivery requires target_policy='corridor_exemplar_v1'"
        )
    if (
        config.selected_rerun_batch_size is not None
        and config.selected_rerun_batch_size < 1
    ):
        blockers.append("selected_rerun_batch_size must be positive")
    for label, path in (
        ("dataset", config.dataset_path),
        ("corpus manifest", config.corpus_manifest_path),
        ("teacher model provenance", config.teacher_model_provenance_path),
    ):
        if not path.is_file():
            blockers.append(f"{label} path missing: {path}")
    if not blockers:
        corpus_report = validate_corpus_artifact(config.corpus_manifest_path.parent)
        blockers.extend(
            f"corpus manifest invalid: {item}" for item in corpus_report.blockers
        )
        teacher_report = validate_teacher_model_provenance(
            config.teacher_model_provenance_path
        )
        blockers.extend(
            f"teacher model provenance invalid: {item}"
            for item in teacher_report.blockers
        )


def migrate_c6_3_5_1_metadata(
    config: ProductionBuildConfig,
) -> CompatibilityMigrationResult:
    """Upgrade the legacy native payload/index metadata without body writes."""

    output = config.output_dir
    delivery_path = output / "delivery_report.json"
    index_path = output / "selected_exemplars" / "payload_index.json"
    authority_path = output / "c6" / "authority_manifest.json"
    try:
        delivery = read_json_object(delivery_path)
        payload_index = read_json_object(index_path)
        authority = read_json_object(authority_path)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return CompatibilityMigrationResult(False)

    shard_paths = sorted(
        (output / "selected_exemplars").glob("selected-exemplars-*.json")
    )
    if not _is_legacy_native_c6_signature(delivery, payload_index, shard_paths):
        return CompatibilityMigrationResult(False)

    from_schema = "pre_c6_3_4_native_streamed_v1"
    reasons: list[str] = []
    expected_authority = authority.get("score_pass_authority_hash")
    if not expected_authority:
        reasons.append("score_pass_authority_missing")
    selection_hash = authority.get("selection_integration_config_hash")
    if not selection_hash:
        reasons.append("selection_integration_config_hash_missing")
    if delivery.get("delivery_authority_hash") not in {None, expected_authority}:
        reasons.append("delivery_authority_mismatch")
    if delivery.get("score_pass_authority_hash") not in {None, expected_authority}:
        reasons.append("delivery_score_pass_authority_mismatch")
    if delivery.get("selection_integration_config_hash") not in {
        None,
        selection_hash,
    }:
        reasons.append("delivery_selection_config_mismatch")
    if reasons:
        return _failed_compatibility_migration(from_schema, reasons)

    shard_by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
    for path in shard_paths:
        try:
            shard = read_json_object(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"payload_shard_invalid:{path.name}:{exc}")
            continue
        items = shard.get("selected_exemplars")
        if (
            shard.get("delivery_authority_hash") != expected_authority
            or shard.get("payload_hash") != _native_payload_hash_for_probe(shard)
            or not isinstance(items, list)
            or len(items) != 1
        ):
            reasons.append(f"payload_shard_authority_or_hash_invalid:{path.name}")
            continue
        item = items[0]
        coordinate = _probe_coordinate(item)
        if coordinate is None or coordinate in shard_by_coordinate:
            reasons.append(f"payload_shard_coordinate_invalid:{path.name}")
            continue
        if item.get("delivery_authority_hash") != expected_authority:
            reasons.append(f"payload_record_authority_mismatch:{path.name}")
            continue
        shard_by_coordinate[coordinate] = {
            "path": path,
            "payload_hash": shard["payload_hash"],
            "payload_ref": item.get("payload_ref"),
        }

    indexed_records = payload_index.get("selected_exemplars")
    if not isinstance(indexed_records, list):
        reasons.append("payload_index_incomplete")
        indexed_records = []
    updated_index_records: list[dict[str, Any]] = []
    backfilled = 0
    seen_index_coordinates: set[tuple[str, int]] = set()
    for raw_record in indexed_records:
        if not isinstance(raw_record, dict):
            reasons.append("payload_index_record_invalid")
            continue
        record = dict(raw_record)
        coordinate = _probe_coordinate(record)
        if coordinate is None or coordinate in seen_index_coordinates:
            reasons.append("payload_index_coordinate_invalid")
            continue
        seen_index_coordinates.add(coordinate)
        shard_info = shard_by_coordinate.get(coordinate)
        if shard_info is None:
            reasons.append(f"payload_index_coordinate_missing:{coordinate}")
            continue
        shard_hash = shard_info["payload_hash"]
        if record.get("payload_hash") is None:
            record["payload_hash"] = shard_hash
            backfilled += 1
        elif record.get("payload_hash") != shard_hash:
            reasons.append(f"payload_index_hash_mismatch:{coordinate}")
        if record.get("payload_ref") != shard_info["payload_ref"]:
            reasons.append(f"payload_index_ref_mismatch:{coordinate}")
        if record.get("delivery_authority_hash") not in {None, expected_authority}:
            reasons.append(f"payload_index_authority_mismatch:{coordinate}")
        updated_index_records.append(record)

    if len(shard_by_coordinate) != len(indexed_records) or (
        set(shard_by_coordinate) != seen_index_coordinates
    ):
        reasons.append("payload_coordinate_set_mismatch")
    if reasons:
        return _failed_compatibility_migration(from_schema, reasons)

    updated_index = dict(payload_index)
    updated_index["selected_exemplars"] = updated_index_records
    migration_metadata = {
        "schema_version": "c6_metadata_compatibility_migration_v1",
        "applied": True,
        "from": from_schema,
        "payload_index_hashes_backfilled": backfilled,
        "payload_bodies_modified": False,
        "teacher_work_performed": False,
    }
    updated_authority = dict(authority)
    updated_authority["payload_index_sha256"] = _hash_json_file_after_write(
        index_path, updated_index
    )
    updated_authority["metadata_compatibility_migration"] = migration_metadata
    updated_delivery = dict(delivery)
    updated_delivery.update(
        {
            "delivery_authority_hash": expected_authority,
            "score_pass_authority_hash": expected_authority,
            "selection_integration_config_hash": selection_hash,
            "metadata_compatibility_migration": migration_metadata,
            "compatibility_migration_applied": True,
            "compatibility_migration_from": from_schema,
            "payload_index_hashes_backfilled": backfilled,
            "payload_bodies_modified": False,
            "teacher_work_performed": False,
        }
    )

    try:
        _write_json_atomic_metadata(index_path, updated_index)
        _write_json_atomic_metadata(authority_path, updated_authority)
        updated_delivery["authority_manifest_sha256"] = _file_sha256(authority_path)
        updated_delivery["payload_index_sha256"] = _file_sha256(index_path)
        _write_json_atomic_metadata(delivery_path, updated_delivery)
    except (OSError, TypeError, ValueError) as exc:
        return _failed_compatibility_migration(
            from_schema,
            [f"metadata_migration_write_failed:{exc}"],
        )
    return CompatibilityMigrationResult(
        applicable=True,
        applied=True,
        from_schema=from_schema,
        payload_index_hashes_backfilled=backfilled,
    )


def _is_legacy_native_c6_signature(
    delivery: Mapping[str, Any],
    payload_index: Mapping[str, Any],
    shard_paths: list[Path],
) -> bool:
    """Use the positive legacy envelope/index shape, not missing metadata alone."""

    indexed_records = payload_index.get("selected_exemplars")
    index_hashes_missing = isinstance(indexed_records, list) and any(
        isinstance(record, dict) and "payload_hash" not in record
        for record in indexed_records
    )
    delivery_metadata_missing = any(
        field not in delivery
        for field in (
            "delivery_authority_hash",
            "score_pass_authority_hash",
            "selection_integration_config_hash",
        )
    )
    return bool(
        delivery.get("execution_mode") == "native_c6_path_b_v1"
        and delivery.get("delivery_path") == "two_pass_rerun_selected"
        and payload_index.get("schema_version") == "selected_exemplar_payload_index_v1"
        and payload_index.get("storage_kind") == "one_record_json_shards_v1"
        and shard_paths
        and delivery.get("metadata_compatibility_migration") is None
        and (index_hashes_missing or delivery_metadata_missing)
        and all(
            path.name.startswith("selected-exemplars-")
            and path.suffix == ".json"
            and _has_native_shard_structure(path)
            for path in shard_paths
        )
    )


def _has_native_shard_structure(path: Path) -> bool:
    try:
        shard = read_json_object(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        shard.get("schema_version") == "selected_exemplar_payload_shard_v1"
        and "delivery_authority_hash" in shard
        and "record_index" in shard
        and "payload_hash" in shard
        and isinstance(shard.get("selected_exemplars"), list)
    )


def _failed_compatibility_migration(
    from_schema: str,
    reasons: list[str],
) -> CompatibilityMigrationResult:
    return CompatibilityMigrationResult(
        applicable=True,
        applied=False,
        reasons=tuple(dict.fromkeys(reasons)),
        from_schema=from_schema,
    )


def _hash_json_file_after_write(path: Path, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _write_json_atomic_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.compatibility.tmp")
    try:
        write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_c6_finalization_only_resume(
    config: ProductionBuildConfig,
) -> FinalizationResumeEligibility:
    """Check a completed native delivery before any accelerator preflight."""

    reasons: list[str] = []
    if not config.resume:
        reasons.append("resume_not_requested")
    if config.selection_integration_policy != C6_SELECTION_INTEGRATION_POLICY:
        reasons.append("selection_integration_policy_not_c6")
    if config.target_policy != "corridor_exemplar_v1":
        reasons.append("target_policy_not_corridor_exemplar_v1")
    if not config.exemplar_selection_enabled:
        reasons.append("exemplar_selection_disabled")
    if config.exemplar_delivery_path != "two_pass_rerun_selected":
        reasons.append("delivery_path_not_two_pass_rerun_selected")
    if reasons:
        return FinalizationResumeEligibility(False, tuple(reasons))

    def require_file(path: Path, reason: str) -> None:
        if not path.is_file():
            reasons.append(f"{reason}: {path}")

    output = config.output_dir
    manifest_path = _run_manifest_path(config)
    for path, reason in (
        (output / "metadata.json", "metadata_missing"),
        (manifest_path, "run_manifest_missing"),
        (output / "emission_config.json", "emission_config_missing"),
        (output / "teacher_manifest.json", "teacher_manifest_missing"),
        (output / "corridors" / "corridor_modes.json", "corridor_modes_missing"),
        (
            output / "corridors" / "mode_assignments.json",
            "corridor_assignments_missing",
        ),
        (output / "c6" / "authority_manifest.json", "authority_manifest_missing"),
        (
            output / "c6" / "selection_budget_diagnostics.json",
            "selection_budget_diagnostics_missing",
        ),
        (
            output / "c6" / "coverage-plan" / "coverage_plan.json",
            "coverage_plan_missing",
        ),
        (
            output / "c6" / "claims" / "claim_manifest.json",
            "claim_artifact_missing",
        ),
        (
            output / "c6" / "multi-role-selection" / "manifest.json",
            "selected_coordinate_artifact_missing",
        ),
        (output / "delivery_report.json", "delivery_report_missing"),
        (
            output / "leaderboards" / "selected_exemplars.json",
            "selected_records_missing",
        ),
        (
            output / "selected_exemplars" / "payload_index.json",
            "payload_index_missing",
        ),
    ):
        require_file(path, reason)
    if reasons:
        return FinalizationResumeEligibility(False, tuple(reasons))

    try:
        run_manifest = read_json_object(manifest_path)
        emission = read_json_object(output / "emission_config.json")
        teacher_manifest = read_json_object(output / "teacher_manifest.json")
        authority = read_json_object(output / "c6" / "authority_manifest.json")
        delivery = read_json_object(output / "delivery_report.json")
        selected_doc = read_json_object(
            output / "leaderboards" / "selected_exemplars.json"
        )
        payload_index = read_json_object(
            output / "selected_exemplars" / "payload_index.json"
        )
        budget = read_json_object(output / "c6" / "selection_budget_diagnostics.json")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return FinalizationResumeEligibility(False, (f"artifact_json_invalid: {exc}",))

    if run_manifest.get("status") != "complete":
        reasons.append("score_pass_incomplete")
    if run_manifest.get("num_examples_completed") != run_manifest.get(
        "num_examples_planned"
    ):
        reasons.append("score_pass_example_checkpoint_incomplete")
    if run_manifest.get("num_shards_completed") != run_manifest.get(
        "num_shards_planned"
    ):
        reasons.append("score_pass_shard_checkpoint_incomplete")
    if delivery.get("status") != "pass":
        reasons.append("delivery_report_missing_or_failed")
    if delivery.get("execution_mode") != "native_c6_path_b_v1":
        reasons.append("delivery_not_native_c6_path_b")

    selected_records = selected_doc.get("selected_exemplars")
    indexed_records = payload_index.get("selected_exemplars")
    if not isinstance(selected_records, list):
        reasons.append("selected_records_invalid")
        selected_records = []
    if not isinstance(indexed_records, list):
        reasons.append("payload_index_incomplete")
        indexed_records = []
    expected_count = len(selected_records)
    if config.total_selected_exemplar_budget is not None and (
        expected_count != config.total_selected_exemplar_budget
    ):
        reasons.append(
            "selected_coordinate_count_mismatch: "
            f"stored={expected_count} requested={config.total_selected_exemplar_budget}"
        )
    if expected_count == 0:
        reasons.append("selected_coordinate_count_zero")
    if len(indexed_records) != expected_count:
        reasons.append(
            "payload_index_count_mismatch: "
            f"stored={len(indexed_records)} expected={expected_count}"
        )

    selected_coordinates = _probe_coordinates(selected_records)
    indexed_coordinates = _probe_coordinates(indexed_records)
    if len(selected_coordinates) != expected_count:
        reasons.append("selected_coordinate_duplicates")
    if len(indexed_coordinates) != len(indexed_records):
        reasons.append("payload_index_duplicate_coordinates")
    if selected_coordinates != indexed_coordinates:
        reasons.append("payload_coordinate_set_mismatch")

    authority_hash = authority.get("score_pass_authority_hash")
    if not authority_hash:
        reasons.append("score_pass_authority_missing")
    if delivery.get("delivery_authority_hash") != authority_hash:
        reasons.append("delivery_authority_mismatch")
    if authority.get("score_pass_config_hash") != run_manifest.get(
        "emission_config_hash"
    ):
        reasons.append("score_pass_config_hash_mismatch")
    if authority.get("score_pass_resume_hash") != run_manifest.get(
        "resume_config_hash"
    ):
        reasons.append("score_pass_resume_hash_mismatch")
    if authority.get("target_store_metadata_sha256") != _file_sha256(
        output / "metadata.json"
    ):
        reasons.append("target_store_metadata_hash_mismatch")
    if authority.get("delivery_path") != config.exemplar_delivery_path:
        reasons.append("authority_delivery_path_mismatch")
    if authority.get(
        "selection_integration_config_hash"
    ) != _selection_integration_hash(config):
        reasons.append("selection_config_mismatch")
    if emission.get("selection_integration_config_hash") != _selection_integration_hash(
        config
    ):
        reasons.append("emission_selection_config_hash_mismatch")
    if budget.get("total_budget_requested") != config.total_selected_exemplar_budget:
        reasons.append("selection_budget_mismatch")

    _probe_configuration_bindings(
        config,
        run_manifest,
        emission,
        teacher_manifest,
        indexed_records,
        reasons,
    )
    _probe_authority_files(output, authority, reasons)
    _probe_payload_transaction(
        output,
        expected_count=expected_count,
        selected_records=selected_records,
        indexed_records=indexed_records,
        expected_authority=authority_hash,
        reasons=reasons,
    )
    return FinalizationResumeEligibility(not reasons, tuple(reasons))


def _probe_configuration_bindings(
    config: ProductionBuildConfig,
    run_manifest: Mapping[str, Any],
    emission: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    indexed_records: list[Any],
    reasons: list[str],
) -> None:
    expected = {
        "teacher_backend": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "dynamic_top_k_min": config.dynamic_top_k_min,
        "dynamic_top_k_max": config.dynamic_top_k_max,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "selection_integration_policy": config.selection_integration_policy,
    }
    stored = {
        **{
            key: run_manifest.get(key)
            for key in ("teacher_backend", "runtime_mode", "target_policy")
        },
        **{
            key: emission.get(key)
            for key in (
                "sequence_length",
                "top_k",
                "dynamic_top_k_min",
                "dynamic_top_k_max",
                "dynamic_mass_threshold",
                "selection_integration_policy",
            )
        },
        "vocab_size": teacher_manifest.get("vocab_size"),
        "num_buckets": (
            indexed_records[0].get("num_buckets")
            if indexed_records and isinstance(indexed_records[0], dict)
            else None
        ),
    }
    for key, expected_value in expected.items():
        if stored.get(key) != expected_value:
            reasons.append(
                f"{key}_mismatch: stored={stored.get(key)!r} "
                f"requested={expected_value!r}"
            )
    if teacher_manifest.get("teacher_model_id") != config.teacher_model:
        reasons.append("teacher_model_identity_mismatch")
    if teacher_manifest.get("tokenizer_id") != (
        config.tokenizer_id or config.teacher_model
    ):
        reasons.append("tokenizer_identity_mismatch")
    if run_manifest.get("dataset_hash") != _file_sha256(config.dataset_path):
        reasons.append("dataset_hash_mismatch")
    try:
        current_corpus_hash = corpus_provenance_from_manifest(
            config.corpus_manifest_path
        )["source_corpus_hash"]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        reasons.append(f"corpus_hash_unavailable: {exc}")
    else:
        if run_manifest.get("corpus_hash") != current_corpus_hash:
            reasons.append("corpus_hash_mismatch")
    try:
        provenance = read_json_object(config.teacher_model_provenance_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reasons.append(f"teacher_model_provenance_unavailable: {exc}")
    else:
        for key in (
            "config_hash",
            "tokenizer_hash",
            "weights_hash",
            "model_directory_hash",
        ):
            if run_manifest.get("teacher_model_hashes", {}).get(key) != provenance.get(
                key
            ):
                reasons.append(f"teacher_model_{key}_mismatch")


def _probe_authority_files(
    output: Path,
    authority: Mapping[str, Any],
    reasons: list[str],
) -> None:
    paths = authority.get("paths", {})
    hashes = authority.get("hashes", {})
    for path_key, hash_key in (
        ("selector", "selector_sha256"),
        ("global_board_supply", "global_board_supply_sha256"),
        ("source_passports", "source_passports_manifest_sha256"),
        ("corridor_features", "corridor_features_sha256"),
    ):
        relative = paths.get(path_key)
        expected_hash = hashes.get(hash_key)
        if not relative or not expected_hash:
            reasons.append(f"authority_{path_key}_binding_missing")
            continue
        path = output / str(relative)
        if not path.is_file():
            reasons.append(f"authority_{path_key}_missing")
        elif _file_sha256(path) != expected_hash:
            reasons.append(f"authority_{path_key}_hash_mismatch")


def _probe_payload_transaction(
    output: Path,
    *,
    expected_count: int,
    selected_records: list[Any],
    indexed_records: list[Any],
    expected_authority: Any,
    reasons: list[str],
) -> None:
    selected_by_coordinate = {
        coordinate: record
        for record in selected_records
        if isinstance(record, dict)
        and (coordinate := _probe_coordinate(record)) is not None
    }
    indexed_by_coordinate = {
        coordinate: record
        for record in indexed_records
        if isinstance(record, dict)
        and (coordinate := _probe_coordinate(record)) is not None
    }
    if (
        len(list((output / "selected_exemplars").glob("selected-exemplars-*.json")))
        != expected_count
    ):
        reasons.append("payload_shard_count_mismatch")
    seen: set[tuple[str, int]] = set()
    for path in sorted(
        (output / "selected_exemplars").glob("selected-exemplars-*.json")
    ):
        try:
            shard = read_json_object(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"payload_shard_invalid: {path.name}: {exc}")
            continue
        items = shard.get("selected_exemplars")
        if (
            shard.get("schema_version") != "selected_exemplar_payload_shard_v1"
            or shard.get("delivery_authority_hash") != expected_authority
            or not isinstance(items, list)
            or len(items) != 1
            or shard.get("payload_hash") != _native_payload_hash_for_probe(shard)
        ):
            reasons.append(f"payload_envelope_invalid: {path.name}")
            continue
        item = items[0]
        coordinate = _probe_coordinate(item)
        if coordinate is None or coordinate in seen:
            reasons.append(f"payload_coordinate_set_mismatch: {path.name}")
            continue
        seen.add(coordinate)
        if (
            coordinate not in selected_by_coordinate
            or coordinate not in indexed_by_coordinate
        ):
            reasons.append(f"payload_foreign_coordinate: {path.name}")
            continue
        if item.get("delivery_authority_hash") != expected_authority:
            reasons.append(f"payload_authority_mismatch: {path.name}")
        indexed = indexed_by_coordinate[coordinate]
        selected = selected_by_coordinate[coordinate]
        if item.get("payload_ref") != indexed.get("payload_ref"):
            reasons.append(f"payload_index_ref_mismatch: {path.name}")
        if item.get("payload_ref") != selected.get("payload_ref"):
            reasons.append(f"payload_selection_ref_mismatch: {path.name}")
        if indexed.get("payload_hash") != shard.get("payload_hash"):
            reasons.append(f"payload_index_hash_mismatch: {path.name}")
    if seen != set(selected_by_coordinate):
        reasons.append("payload_coordinate_set_mismatch")


def _probe_coordinate(item: Any) -> tuple[str, int] | None:
    if not isinstance(item, dict):
        return None
    try:
        return str(item["selected_example_id"]), int(item["selected_position"])
    except (KeyError, TypeError, ValueError):
        return None


def _probe_coordinates(items: list[Any]) -> set[tuple[str, int]]:
    return {
        coordinate
        for item in items
        if (coordinate := _probe_coordinate(item)) is not None
    }


def _native_payload_hash_for_probe(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _export_c6_selection_authorities(
    config: ProductionBuildConfig,
) -> dict[str, Path | str | bool]:
    """Compose the preserved fingerprint and global C6 authority exports."""

    fingerprint = _export_c6_fingerprint_selection_authority(config)
    return _export_c6_global_authority(config, fingerprint)


def _export_c6_fingerprint_selection_authority(
    config: ProductionBuildConfig,
) -> dict[str, Path | str]:
    """Export selector/features from the provisional score-surface corridor.

    This is the first authority boundary in the canonical native Path-B route.
    It intentionally leaves global-board supply, source passports, and the
    combined authority manifest to the following global-authority operation.
    The compatibility composer above preserves the historical public helper.
    """

    if config.total_selected_exemplar_budget is None:
        raise ValueError("C6 total_selected_exemplar_budget is required")
    c6_root = config.output_dir / "c6"
    c6_root.mkdir(parents=True, exist_ok=True)
    store = TeacherTargetStore.open(config.output_dir)
    examples = load_text_examples(
        config.dataset_path,
        max_examples=store.metadata.num_examples,
    )
    selector_manifest = build_exemplar_selection_manifest(
        store,
        examples=examples,
        batch_size=max(1, config.shard_size_examples),
        capture_mode=_exemplar_capture_mode(config),
        fulfillment_policy=(
            "rerun_selected_capture"
            if config.exemplar_delivery_path == "two_pass_rerun_selected"
            else "select_from_existing_capture"
        ),
        # Capacity is distinct from the C3 allocation.  Retaining it at least
        # as deep as the final budget leaves ranked supply for C4 backfill.
        board_capacity=max(
            config.total_selected_exemplar_budget,
            config.exemplar_leaderboard_capacity,
        ),
        budget_examples=None,
        budget_fraction=None,
        created_at=_now(),
        canonical_score_fields_only=True,
        use_score_pass_fields=True,
        production_global_selector=True,
    )
    selector_path = c6_root / "production_global_selector.json"
    write_json(selector_path, selector_manifest)
    score_pass_authority_hash = _hash_payload(
        {
            "metadata_sha256": _file_sha256(config.output_dir / "metadata.json"),
            "assignment_manifest_sha256": _file_sha256(
                config.output_dir / "corridors" / "mode_assignments.json"
            ),
            "modes_sha256": _file_sha256(
                config.output_dir / "corridors" / "corridor_modes.json"
            ),
            "selector_sha256": _file_sha256(selector_path),
            "selection_integration_config_hash": _selection_integration_hash(config),
        }
    )
    feature_path = export_corridor_candidate_features(
        artifact_dir=config.output_dir,
        output_dir=c6_root / "corridor-features",
    )
    feature_manifest_path = c6_root / "corridor-features" / "manifest.json"
    feature_manifest = read_json_object(feature_manifest_path)
    feature_manifest["score_pass_authority_hash"] = score_pass_authority_hash
    write_json(feature_manifest_path, feature_manifest)
    return {
        "selector_path": selector_path,
        "feature_path": feature_path,
        "score_pass_authority_hash": score_pass_authority_hash,
    }


def _export_c6_global_authority(
    config: ProductionBuildConfig,
    fingerprint: Mapping[str, Path | str],
) -> dict[str, Path | str | bool]:
    """Export global supply/passports after matching fingerprint authority."""

    c6_root = config.output_dir / "c6"
    selector_path = Path(str(fingerprint["selector_path"]))
    feature_path = Path(str(fingerprint["feature_path"]))
    score_pass_authority_hash = str(fingerprint["score_pass_authority_hash"])
    selector_manifest = read_json_object(selector_path)
    global_supply = export_production_global_board_supply(
        selector_manifest,
        source_artifact_id=str(config.output_dir),
        source_artifact_hash=score_pass_authority_hash,
    )
    global_supply["source_provenance"]["score_pass_authority_hash"] = (
        score_pass_authority_hash
    )
    global_path = c6_root / "global-board-supply.json"
    write_json(global_path, global_supply)
    passports_path = export_production_source_passports(
        artifact_dir=config.output_dir,
        output_path=c6_root / "source-passports.json",
        score_pass_authority_hash=score_pass_authority_hash,
    )
    authority_manifest_path = c6_root / "authority_manifest.json"
    run_manifest = read_json_object(config.output_dir / "run_manifest.json")
    authority_manifest = {
        "schema_version": "radjax.c6_selection_authority.v1",
        "score_pass_authority_hash": score_pass_authority_hash,
        "target_store_metadata_sha256": _file_sha256(
            config.output_dir / "metadata.json"
        ),
        "corpus_hash": run_manifest.get("corpus_hash"),
        "score_pass_config_hash": run_manifest.get("emission_config_hash"),
        "score_pass_resume_hash": run_manifest.get("resume_config_hash"),
        "selection_integration_config_hash": _selection_integration_hash(config),
        "delivery_path": config.exemplar_delivery_path,
        "paths": {
            "selector": selector_path.relative_to(config.output_dir).as_posix(),
            "global_board_supply": global_path.relative_to(
                config.output_dir
            ).as_posix(),
            "source_passports": passports_path.relative_to(
                config.output_dir
            ).as_posix(),
            "corridor_features": feature_path.relative_to(config.output_dir).as_posix(),
        },
        "hashes": {
            "selector_sha256": _file_sha256(selector_path),
            "global_board_supply_sha256": _file_sha256(global_path),
            "source_passports_manifest_sha256": _file_sha256(passports_path),
            "corridor_features_sha256": _file_sha256(feature_path),
        },
        "external_authority_override_used": False,
        "production_grade": True,
    }
    write_json(authority_manifest_path, authority_manifest)
    _mark_native_c6_score_pass_artifact(
        config,
        selector_path=selector_path,
        authority_manifest_path=authority_manifest_path,
    )
    external_override_used = _validate_external_c6_overrides(
        config,
        score_pass_authority_hash=score_pass_authority_hash,
    )
    selected_global_path = config.global_board_supply_path or global_path
    selected_passports_path = config.source_passports_path or passports_path
    authority_manifest["authority_paths_used"] = {
        "global_board_supply": str(selected_global_path),
        "source_passports": str(selected_passports_path),
    }
    authority_manifest["external_authority_override_used"] = external_override_used
    write_json(authority_manifest_path, authority_manifest)
    return {
        "feature_path": feature_path,
        "global_board_supply_path": selected_global_path,
        "source_passports_path": selected_passports_path,
        "authority_manifest_path": authority_manifest_path,
        "score_pass_authority_hash": score_pass_authority_hash,
        "external_authority_override_used": external_override_used,
    }


def _mark_native_c6_score_pass_artifact(
    config: ProductionBuildConfig,
    *,
    selector_path: Path,
    authority_manifest_path: Path,
) -> None:
    """Correct the score-pass sidecar once native C6 authority exists."""

    if not _native_c6_path_b_enabled(config):
        return
    path = config.output_dir / "emission_config.json"
    payload = read_json_object(path)
    claims = [
        str(item)
        for item in payload.get("claims_not_made", ())
        if str(item) != "no_production_global_two_pass_selector"
    ]
    payload.update(
        {
            "exemplar_selection_enabled": True,
            "exemplar_selection_manifest": selector_path.relative_to(
                config.output_dir
            ).as_posix(),
            "selection_integration_policy": C6_SELECTION_INTEGRATION_POLICY,
            "native_execution_mode": "native_c6_path_b_v1",
            "selection_authority_manifest": authority_manifest_path.relative_to(
                config.output_dir
            ).as_posix(),
            "claims_not_made": claims,
        }
    )
    write_json(path, payload)


def _validate_external_c6_overrides(
    config: ProductionBuildConfig,
    *,
    score_pass_authority_hash: str,
) -> bool:
    """Fail closed when an optional checkpoint is not tied to this score pass."""

    used = False
    for label, path in (
        ("global board supply", config.global_board_supply_path),
        ("source passports", config.source_passports_path),
    ):
        if path is None:
            continue
        if not path.is_file():
            raise ValueError(f"{label} override path missing: {path}")
        payload = read_json_object(path)
        provenance = payload.get("source_provenance", payload)
        observed = (
            provenance.get("score_pass_authority_hash")
            if isinstance(provenance, Mapping)
            else None
        )
        if observed != score_pass_authority_hash:
            raise ValueError(
                f"{label} override does not match the current score-pass authority hash"
            )
        used = True
    return used


def _prepare_c6_selection(
    config: ProductionBuildConfig,
    authorities: Mapping[str, Path | str | bool],
) -> dict[str, Any]:
    """Run C2-C5 from the internally exported production authorities."""

    if config.total_selected_exemplar_budget is None:
        raise ValueError("C6 total_selected_exemplar_budget is required")
    c6_root = config.output_dir / "c6"
    c6_root.mkdir(parents=True, exist_ok=True)
    c2_summary: dict[str, Any] = {}
    c3_summary: dict[str, Any] = {}
    global_supply: dict[str, Any] | None = None

    feature_path = Path(str(authorities["feature_path"]))
    feature_records = load_candidate_records_jsonl(
        feature_path,
        source_artifact_id=str(feature_path),
    )
    leaderboards = build_corridor_candidate_leaderboards(
        feature_records,
        CorridorLeaderboardPolicy(
            candidate_pool_cap=config.fingerprint_corridor_candidate_pool_cap,
        ),
    )
    c2_path = write_corridor_candidate_leaderboards(
        leaderboards,
        c6_root / "corridor-leaderboards",
        overwrite=True,
    )
    c2_summary = inspect_corridor_candidate_leaderboards(c2_path)
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            corridor_budget_fraction=config.fingerprint_corridor_budget_fraction,
            corridor_budget_max=config.fingerprint_corridor_budget_max,
            corridor_mode_cap=config.fingerprint_corridor_mode_cap,
        ),
        source_leaderboard_provenance=c2_summary,
    )
    c3_path = write_corridor_coverage_plan(
        plan,
        c6_root / "coverage-plan",
        overwrite=True,
    )
    c3_summary = inspect_corridor_coverage_plan(c3_path)
    c3_summary["mode_allocations"] = [
        {
            "mode_id": mode.corridor_mode_id,
            "allocated_slots": mode.allocated_slots,
            "zero_allocation_reason": mode.zero_allocation_reason,
        }
        for mode in plan.modes
    ]
    global_input = load_global_board_input(
        Path(str(authorities["global_board_supply_path"])),
        production_grade=True,
    )
    global_provenance = global_input.source_provenance
    if global_provenance.get("selector_policy") != (
        "multi_leaderboard_exemplar_selector_v1"
    ) or global_provenance.get("selector_schema_version") != (
        "exemplar_selection_manifest_v1"
    ):
        raise ValueError(
            "C6 global board supply must be exported by the production global selector"
        )
    global_supply = global_input.to_dict()
    claims = claim_corridor_then_backfill_global(
        leaderboards,
        plan,
        global_input,
        CorridorGlobalClaimPolicy(
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            # Preserve the complete claim/backfill artifact before enforcing a
            # production budget failure at the Path B teacher boundary.
            require_full_budget=False,
        ),
    )
    write_corridor_global_claim_result(
        claims,
        c6_root / "claims",
        overwrite=True,
    )
    budget_diagnostics = _c6_budget_diagnostics(
        config,
        claims=claims,
        leaderboards=leaderboards,
        plan=plan,
        global_supply=global_supply,
    )
    write_json(c6_root / "selection_budget_diagnostics.json", budget_diagnostics)
    if config.require_full_selected_budget and budget_diagnostics["budget_shortfall"]:
        raise C6BudgetShortfallError(budget_diagnostics)
    source_passports = load_source_passports_for_coordinates(
        Path(str(authorities["source_passports_path"])),
        {
            (coordinate.example_id, coordinate.position)
            for coordinate in claims.selected_coordinates
        },
    )
    selected = build_multi_role_selected_exemplars(
        claims,
        source_passports=source_passports,
    )
    write_multi_role_selection_artifact(
        selected,
        c6_root / "multi-role-selection",
        overwrite=True,
    )
    delivery_path = config.exemplar_delivery_path or "one_pass_pruned_candidate"
    delivery_records = c5_records_for_delivery(
        selected,
        delivery_path=delivery_path,
    )
    return {
        "claims": claims,
        "selected": selected,
        # C6 final validation needs passports for the C5 set, not the whole
        # full-corpus authority stream.
        "source_passports": [
            dict(record.source_passport) for record in selected.records
        ],
        "delivery_records": delivery_records,
        "c2_summary": c2_summary,
        "c3_summary": c3_summary,
        "global_supply": global_supply,
        "budget_diagnostics": budget_diagnostics,
        "authorities": dict(authorities),
    }


def _c6_budget_diagnostics(
    config: ProductionBuildConfig,
    *,
    claims: Any,
    leaderboards: Any,
    plan: Any,
    global_supply: Mapping[str, Any],
) -> dict[str, Any]:
    requested = int(config.total_selected_exemplar_budget or 0)
    final_count = len(claims.selected_coordinates)
    corridor_candidate_entries = [
        (candidate.candidate_id, candidate.position)
        for mode in leaderboards.modes
        for candidate in mode.candidates
    ]
    corridor_candidates = set(corridor_candidate_entries)
    global_candidate_entries = [
        (str(candidate["example_id"]), int(candidate["position"]))
        for board in global_supply.get("boards", [])
        if isinstance(board, Mapping)
        for candidate in board.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]
    global_candidates = set(global_candidate_entries)
    corridor_claim_set = {
        (claim.example_id, claim.position) for claim in claims.corridor_claims
    }
    global_claim_set = {
        (claim.example_id, claim.position) for claim in claims.global_claims
    }
    intersection = corridor_claim_set & global_candidates
    union = corridor_claim_set | global_candidates
    corridor_budget_requested = sum(int(mode.allocated_slots) for mode in plan.modes)
    global_claims = len(claims.global_claims)
    corridor_claims = len(claims.corridor_claims)
    collisions = list(claims.collision_obligations)
    global_examined = sum(
        int(item.get("candidate_count_seen") or 0)
        for item in (claims.summary or {}).get("board_summaries", [])
        if isinstance(item, Mapping)
    )
    shortfall = max(0, requested - final_count)
    if not shortfall:
        reason = None
    elif len(corridor_candidates | global_candidates) < requested:
        reason = "insufficient_eligible_unique_candidates"
    elif global_claims < requested - corridor_claims:
        reason = "global_ranked_supply_exhaustion"
    elif collisions:
        reason = "deduplication_overlap_exhaustion"
    else:
        reason = "fingerprint_corridor_allocation_or_cap_exhaustion"
    return {
        "total_budget_requested": requested,
        "fingerprint_corridor_budget_requested": corridor_budget_requested,
        "fingerprint_corridor_candidates_eligible_unique": len(corridor_candidates),
        "fingerprint_corridor_claims_before_dedup": len(corridor_claim_set),
        "fingerprint_corridor_claims_accepted": corridor_claims,
        "global_supply_exported": len(global_candidates),
        "global_candidates_examined": global_examined,
        "global_claims_accepted": global_claims,
        "cross_role_duplicate_count": len(intersection),
        "accepted_cross_role_overlap": len(corridor_claim_set & global_claim_set),
        "within_role_duplicate_count": (
            len(corridor_candidate_entries)
            - len(corridor_candidates)
            + len(global_candidate_entries)
            - len(global_candidates)
        ),
        "final_unique_selected_count": final_count,
        "budget_shortfall": shortfall,
        "budget_shortfall_reason": reason,
        "global_supply_remaining": max(0, len(global_candidates) - global_examined),
        "fingerprint_corridor_global_intersection_size": len(intersection),
        "fingerprint_corridor_global_jaccard": (
            float(len(intersection)) / float(max(len(union), 1))
        ),
        "accepted_global_rank_depth": max(
            (claim.global_rank for claim in claims.global_claims), default=0
        ),
    }


def _finalize_c6_selection(
    config: ProductionBuildConfig,
    context: dict[str, Any],
    *,
    delivery_report: dict[str, Any] | None,
    audit_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    claims = context["claims"]
    selected = context["selected"]
    legacy: list[Mapping[str, Any]] = []
    payloads: list[Mapping[str, Any]] = []
    selected_path = config.output_dir / "leaderboards" / "selected_exemplars.json"
    payload_path = config.output_dir / "selected_exemplars" / "payload_index.json"
    if selected_path.is_file():
        legacy_payload = read_json_object(selected_path)
        legacy = list(legacy_payload.get("selected_exemplars") or [])
    if payload_path.is_file():
        payload_payload = read_json_object(payload_path)
        payloads = list(payload_payload.get("selected_exemplars") or [])
    else:
        legacy_payload_path = (
            config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
        )
        if legacy_payload_path.is_file():
            payload_payload = read_json_object(legacy_payload_path)
            payloads = list(payload_payload.get("selected_exemplars") or [])
    curriculum_records: list[Mapping[str, Any]] = []
    try:
        curriculum_records = load_curriculum_route_records(config.output_dir)
    except (OSError, TypeError, ValueError) as exc:
        curriculum_records = [{"curriculum_load_error": str(exc)}]
    if delivery_report is None:
        validation = {
            "schema_version": "radjax.c6_integrated_selection_validation.v1",
            "status": "fail",
            "blockers": ["C6 selected delivery did not complete"],
            "warnings": [],
            "selected_unique_count": len(selected.records),
            "selected_obligation_count": selected.summary.get("obligation_count", 0),
            "coordinate_set_authority": "c5",
        }
    else:
        validation = validate_integrated_selection_contract(
            claims,
            selected,
            legacy_records=legacy,
            payload_records=payloads,
            source_passports=context["source_passports"],
            curriculum_records=curriculum_records,
            audit_report=audit_report,
            production_grade=True,
        )
    coverage = build_corridor_coverage_report(
        claims,
        selected,
        c2_summary=context.get("c2_summary"),
        c3_summary=context.get("c3_summary"),
        global_supply=context.get("global_supply"),
        delivery_report=delivery_report,
    )
    coverage["integrated_validation_status"] = validation["status"]
    coverage["integrated_validation_path"] = (
        "reports/c6_integrated_selection_validation.json"
    )
    return validation, coverage


def _backend_config(config: ProductionBuildConfig) -> TeacherBackendConfig:
    tokenizer_id = config.tokenizer_id or config.teacher_model
    return TeacherBackendConfig(
        backend_id=config.teacher_backend,
        runtime_mode=config.runtime_mode,  # type: ignore[arg-type]
        target_policy=config.target_policy,  # type: ignore[arg-type]
        model_id=config.teacher_model,
        tokenizer_id=tokenizer_id,
        sequence_length=config.sequence_length,
        batch_size=1,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        dynamic_top_k_min=config.dynamic_top_k_min,
        dynamic_top_k_max=config.dynamic_top_k_max,
        dynamic_mass_threshold=config.dynamic_mass_threshold,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        local_files_only=True,
        allow_downloads=False,
        fallback_policy="error",
        exemplar_capture_mode=_exemplar_capture_mode(config),
    )


def _streaming_config(
    config: ProductionBuildConfig,
    effective_batch_size: int,
    *,
    progress_callback: Any = None,
) -> BackendTeacherTextbookBuildConfig:
    return BackendTeacherTextbookBuildConfig(
        output_dir=config.output_dir,
        dataset_path=config.dataset_path,
        teacher_backend=config.teacher_backend,
        runtime_mode=config.runtime_mode,
        target_policy=config.target_policy,
        teacher_model_id=config.teacher_model,
        tokenizer_id=config.tokenizer_id or config.teacher_model,
        sequence_length=config.sequence_length,
        batch_size=effective_batch_size,
        max_examples=config.max_examples,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        dynamic_top_k_min=config.dynamic_top_k_min,
        dynamic_top_k_max=config.dynamic_top_k_max,
        dynamic_mass_threshold=config.dynamic_mass_threshold,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        fallback_policy="error",
        exemplar_capture_mode=_exemplar_capture_mode(config),
        local_files_only=True,
        allow_downloads=False,
        overwrite=False,
        corpus_manifest_path=config.corpus_manifest_path,
        teacher_model_provenance_path=config.teacher_model_provenance_path,
        streaming=True,
        resume=config.resume,
        shard_size_examples=config.shard_size_examples,
        progress_log_path=config.progress_log_path,
        run_manifest_path=config.run_manifest_path,
        progress_callback=progress_callback,
        selection_integration_policy=config.selection_integration_policy,
        selection_integration_config_hash=_selection_integration_hash(config),
        exemplar_selection_enabled=_native_c6_path_b_enabled(config),
        native_c6_path_b_execution=_native_c6_path_b_enabled(config),
    )


def _production_report(
    config: ProductionBuildConfig,
    *,
    created_at: str,
    completed_at: str,
    status: str,
    blockers: list[str],
    warnings: list[str],
    doctor_report: dict[str, Any],
    run_plan_path: Path,
    run_plan: dict[str, Any],
    effective_batch_size: int | None,
    already_complete: bool,
    parity_report_path: Path,
    parity_status: str,
    validation_status: str | None = None,
    build_status: str | None = None,
    delivery_report: dict[str, Any] | None = None,
    selected_delivery_failure: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_summary = _artifact_summary(config.output_dir, run_plan)
    report = {
        "schema_version": PRODUCTION_BUILD_REPORT_SCHEMA,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "created_at": created_at,
        "completed_at": completed_at,
        "command": "production-build",
        "output_dir": str(config.output_dir),
        "inputs": _inputs(config),
        "doctor_summary": _doctor_summary(doctor_report),
        "run_plan_path": str(run_plan_path),
        "run_plan_status": run_plan.get("status"),
        "effective_batch_size": effective_batch_size,
        "streaming_build": True,
        "resume_requested": config.resume,
        "already_complete": already_complete,
        "finalization_resume_probe": (
            probe_c6_finalization_only_resume(config).to_dict()
            if config.resume
            else None
        ),
        "run_manifest_path": str(_run_manifest_path(config)),
        "progress_log_path": str(_progress_log_path(config)),
        "production_progress_path": str(_production_progress_path(config)),
        "validation_report_path": str(config.output_dir / "validation_report.json"),
        "validation_status": validation_status or _validation_status(config.output_dir),
        "cover_page_path": str(config.output_dir / "cover_page.json"),
        "delivery_report_path": (
            str(config.output_dir / EXEMPLAR_DELIVERY_REPORT_FILENAME)
            if _selected_exemplar_delivery_enabled(config)
            else None
        ),
        "delivery_path": (
            delivery_report.get("delivery_path")
            if delivery_report is not None
            else config.exemplar_delivery_path
        ),
        "selected_delivery_status": (
            "fail"
            if selected_delivery_failure is not None
            else "pass"
            if delivery_report is not None
            else "not_enabled"
        ),
        "failure_stage": (
            selected_delivery_failure.get("failure_stage")
            if selected_delivery_failure is not None
            else None
        ),
        "selected_delivery_failure": selected_delivery_failure,
        "staging_directory": (
            selected_delivery_failure.get("staging_directory")
            if selected_delivery_failure is not None
            else None
        ),
        "staging_payload_count": (
            selected_delivery_failure.get("staging_payload_count", 0)
            if selected_delivery_failure is not None
            else None
        ),
        "staging_preserved": (
            selected_delivery_failure.get("staging_preserved", False)
            if selected_delivery_failure is not None
            else None
        ),
        "num_examples_scored": (
            delivery_report.get("num_examples_scored")
            if delivery_report is not None
            else None
        ),
        "num_selected_exemplars": (
            delivery_report.get("num_selected_exemplars")
            if delivery_report is not None
            else None
        ),
        "selected_teacher_rerun_count": (
            1
            if delivery_report is not None
            and delivery_report.get("delivery_path") == "two_pass_rerun_selected"
            else 0
            if delivery_report is not None
            else None
        ),
        "selected_teacher_rerun_example_count": (
            delivery_report.get("teacher_rerun_count")
            if delivery_report is not None
            else None
        ),
        "legacy_selected_teacher_rerun_count": (
            0 if _native_c6_path_b_enabled(config) else None
        ),
        "native_c6_selected_teacher_rerun_count": (
            1
            if _native_c6_path_b_enabled(config) and delivery_report is not None
            else 0
            if _native_c6_path_b_enabled(config)
            else None
        ),
        "selected_rerun_batch_size": (
            delivery_report.get("selected_rerun_batch_size")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_batch_count": (
            delivery_report.get("selected_rerun_batch_count")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_examples_per_second": (
            delivery_report.get("selected_rerun_examples_per_second")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_teacher_seconds": (
            delivery_report.get("selected_rerun_teacher_seconds")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_compression_seconds": (
            delivery_report.get("selected_rerun_compression_seconds")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_io_seconds": (
            delivery_report.get("selected_rerun_io_seconds")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_peak_host_memory_bytes": (
            delivery_report.get("selected_rerun_peak_host_memory_bytes")
            if delivery_report is not None
            else None
        ),
        "selected_rerun_peak_device_memory_bytes": (
            delivery_report.get("selected_rerun_peak_device_memory_bytes")
            if delivery_report is not None
            else None
        ),
        "selected_source_example_count": (
            delivery_report.get("selected_source_example_count")
            if delivery_report is not None
            else None
        ),
        "selected_coordinate_count": (
            delivery_report.get("selected_coordinate_count")
            if delivery_report is not None
            else None
        ),
        "requested_source_batch_size": (
            delivery_report.get("requested_source_batch_size")
            if delivery_report is not None
            else None
        ),
        "effective_source_batch_sizes": (
            delivery_report.get("effective_source_batch_sizes")
            if delivery_report is not None
            else None
        ),
        "source_batch_count": (
            delivery_report.get("source_batch_count")
            if delivery_report is not None
            else None
        ),
        "coordinate_compression_batch_count": (
            delivery_report.get("coordinate_compression_batch_count")
            if delivery_report is not None
            else None
        ),
        "selected_row_gather_seconds": (
            delivery_report.get("selected_row_gather_seconds")
            if delivery_report is not None
            else None
        ),
        "payload_write_seconds": (
            delivery_report.get("payload_write_seconds")
            if delivery_report is not None
            else None
        ),
        "selected_board_summary": (
            delivery_report.get("selected_board_summary")
            if delivery_report is not None
            else None
        ),
        "selected_exemplar_payload_retained": (
            delivery_report.get("selected_exemplar_payload_retained")
            if delivery_report is not None
            else None
        ),
        "dynamic_top_k_min": config.dynamic_top_k_min,
        "dynamic_top_k_max": config.dynamic_top_k_max,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "entropy_quantization_step": ENTROPY_PARITY_QUANTIZATION_STEP,
        "entropy_parity_tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
        "long_tail_warning_k": config.long_tail_warning_k,
        "very_long_tail_warning_k": config.very_long_tail_warning_k,
        "perverse_tail_warning_k": config.perverse_tail_warning_k,
        "reject_perverse_exemplars": config.reject_perverse_exemplars,
        "primary_selected_exemplar_budget": (
            config.primary_selected_exemplar_budget
            if config.primary_selected_exemplar_budget is not None
            else config.selected_exemplar_budget
        ),
        "long_tail_side_board_cap": config.long_tail_side_board_cap,
        "perverse_tail_side_board_cap": config.perverse_tail_side_board_cap,
        "include_long_tail_in_primary": config.include_long_tail_in_primary,
        "include_perverse_tail_in_primary": config.include_perverse_tail_in_primary,
        "include_perverse_tail_in_student": config.include_perverse_tail_in_student,
        "long_tail_summary": (
            delivery_report.get("long_tail_summary")
            if delivery_report is not None
            else None
        ),
        "long_tail_observations": (
            delivery_report.get("long_tail_observations")
            if delivery_report is not None
            else None
        ),
        "non_selected_exemplar_payload_retained": (
            delivery_report.get("non_selected_exemplar_payload_retained")
            if delivery_report is not None
            else None
        ),
        "corridor_artifact_built": (
            delivery_report.get("corridor_artifact_built")
            if delivery_report is not None
            else None
        ),
        "corridor_modes_built": (
            delivery_report.get("corridor_modes_built")
            if delivery_report is not None
            else None
        ),
        "corridor_observation_basis": (
            delivery_report.get("corridor_observation_basis")
            if delivery_report is not None
            else None
        ),
        "degraded_corridor_export": (
            delivery_report.get("degraded_corridor_export")
            if delivery_report is not None
            else None
        ),
        "corridor_positions_available": (
            delivery_report.get("corridor_positions_available")
            if delivery_report is not None
            else None
        ),
        "corridor_positions_used": (
            delivery_report.get("corridor_positions_used")
            if delivery_report is not None
            else None
        ),
        "corridor_summary_path": (
            delivery_report.get("corridor_summary_path")
            if delivery_report is not None
            else None
        ),
        "corridor_fingerprints_path": (
            delivery_report.get("corridor_fingerprints_path")
            if delivery_report is not None
            else None
        ),
        "corridor_modes_path": (
            delivery_report.get("corridor_modes_path")
            if delivery_report is not None
            else None
        ),
        "corridor_mode_assignments_path": (
            delivery_report.get("corridor_mode_assignments_path")
            if delivery_report is not None
            else None
        ),
        "corridor_fingerprint_count": (
            delivery_report.get("corridor_fingerprint_count")
            if delivery_report is not None
            else None
        ),
        "corridor_mode_count": (
            delivery_report.get("corridor_mode_count")
            if delivery_report is not None
            else None
        ),
        "corridor_mode_policy": (
            delivery_report.get("corridor_mode_policy")
            if delivery_report is not None
            else None
        ),
        "corridor_max_modes": (
            delivery_report.get("corridor_max_modes")
            if delivery_report is not None
            else None
        ),
        "corridor_tracked_stats": (
            delivery_report.get("corridor_tracked_stats")
            if delivery_report is not None
            else None
        ),
        "corridor_stat_top_k": (
            delivery_report.get("corridor_stat_top_k")
            if delivery_report is not None
            else None
        ),
        "min_corridor_stat_top_k": (
            delivery_report.get("min_corridor_stat_top_k")
            if delivery_report is not None
            else None
        ),
        "corridor_assignment_storage_kind": (
            delivery_report.get("corridor_assignment_storage_kind")
            if delivery_report is not None
            else None
        ),
        "corridor_assignment_count": (
            delivery_report.get("corridor_assignment_count")
            if delivery_report is not None
            else None
        ),
        "selected_exemplars_linked_to_corridor_modes": (
            delivery_report.get("selected_exemplars_linked_to_corridor_modes")
            if delivery_report is not None
            else None
        ),
        "compatibility_migration_applied": (
            delivery_report.get("compatibility_migration_applied")
            if delivery_report is not None
            else None
        ),
        "compatibility_migration_from": (
            delivery_report.get("compatibility_migration_from")
            if delivery_report is not None
            else None
        ),
        "payload_index_hashes_backfilled": (
            delivery_report.get("payload_index_hashes_backfilled")
            if delivery_report is not None
            else 0
        ),
        "payload_bodies_modified": (
            delivery_report.get("payload_bodies_modified")
            if delivery_report is not None
            else False
        ),
        "teacher_work_performed": (
            delivery_report.get("teacher_work_performed")
            if delivery_report is not None
            else None
        ),
        "parity_report_path": str(parity_report_path) if config.parity_left else None,
        "parity_status": parity_status,
        "build_status": build_status,
        "artifact_summary": artifact_summary,
        "claims_not_made": {
            "no_model_download": True,
            "no_network_verification": True,
            "no_silent_cpu_fallback": True,
            "no_multidevice_scheduling": True,
            "no_tpu_jax": True,
            "no_unverified_parity_claim": config.parity_left is None,
        },
    }
    if timing is not None:
        report.update(timing)
    report.update(_c6_report_fields(config.output_dir, config))
    return report


def _inputs(config: ProductionBuildConfig) -> dict[str, Any]:
    return {
        "teacher_model": config.teacher_model,
        "tokenizer_id": config.tokenizer_id or config.teacher_model,
        "dataset": str(config.dataset_path),
        "corpus_manifest": str(config.corpus_manifest_path),
        "teacher_model_provenance": str(config.teacher_model_provenance_path),
        "teacher_backend": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "dynamic_top_k_min": config.dynamic_top_k_min,
        "dynamic_top_k_max": config.dynamic_top_k_max,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "long_tail_warning_k": config.long_tail_warning_k,
        "very_long_tail_warning_k": config.very_long_tail_warning_k,
        "perverse_tail_warning_k": config.perverse_tail_warning_k,
        "reject_perverse_exemplars": config.reject_perverse_exemplars,
        "primary_selected_exemplar_budget": (
            config.primary_selected_exemplar_budget
            if config.primary_selected_exemplar_budget is not None
            else config.selected_exemplar_budget
        ),
        "long_tail_side_board_cap": config.long_tail_side_board_cap,
        "perverse_tail_side_board_cap": config.perverse_tail_side_board_cap,
        "include_long_tail_in_primary": config.include_long_tail_in_primary,
        "include_perverse_tail_in_primary": config.include_perverse_tail_in_primary,
        "include_perverse_tail_in_student": config.include_perverse_tail_in_student,
        "allow_downloads": False,
        "local_files_only": True,
        "exemplar_selection_enabled": config.exemplar_selection_enabled,
        "exemplar_delivery_path": config.exemplar_delivery_path,
        "exemplar_leaderboard_capacity": config.exemplar_leaderboard_capacity,
        "selected_exemplar_budget": config.selected_exemplar_budget,
        "selected_exemplar_fraction": config.selected_exemplar_fraction,
        "retain_unselected_exemplar_payloads": (
            config.retain_unselected_exemplar_payloads
        ),
        "exemplar_score_policy": config.exemplar_score_policy,
        "selected_rerun_batch_size": config.selected_rerun_batch_size,
        "track_delivery_timing": config.track_delivery_timing,
        "progress": config.progress,
        "selection_integration_policy": config.selection_integration_policy,
        "total_selected_exemplar_budget": config.total_selected_exemplar_budget,
        "fingerprint_corridor_budget_fraction": (
            config.fingerprint_corridor_budget_fraction
        ),
        "fingerprint_corridor_budget_max": config.fingerprint_corridor_budget_max,
        "fingerprint_corridor_mode_cap": config.fingerprint_corridor_mode_cap,
        "fingerprint_corridor_candidate_pool_cap": (
            config.fingerprint_corridor_candidate_pool_cap
        ),
        "require_full_selected_budget": config.require_full_selected_budget,
        "corridor_feature_jsonl_path": (
            str(config.corridor_feature_jsonl_path)
            if config.corridor_feature_jsonl_path
            else None
        ),
        "global_board_supply_path": (
            str(config.global_board_supply_path)
            if config.global_board_supply_path
            else None
        ),
        "c4_claims_path": str(config.c4_claims_path) if config.c4_claims_path else None,
        "c5_selection_path": (
            str(config.c5_selection_path) if config.c5_selection_path else None
        ),
        "source_passports_path": (
            str(config.source_passports_path) if config.source_passports_path else None
        ),
        "selection_integration_config_hash": _selection_integration_hash(config),
    }


def _doctor_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "can_emit",
            "failure_stage",
            "failure_reason",
            "dependency_status",
            "torch_available",
            "transformers_available",
            "cuda_available",
            "mps_available",
        )
    }


def _c6_report_fields(
    output_dir: Path,
    config: ProductionBuildConfig,
) -> dict[str, Any]:
    coverage_path = output_dir / "reports" / "fingerprint_corridor_coverage.json"
    validation_path = output_dir / "reports" / "c6_integrated_selection_validation.json"
    coverage = read_json_object(coverage_path) if coverage_path.is_file() else {}
    validation = read_json_object(validation_path) if validation_path.is_file() else {}
    authority_path = output_dir / "c6" / "authority_manifest.json"
    budget_path = output_dir / "c6" / "selection_budget_diagnostics.json"
    authority = read_json_object(authority_path) if authority_path.is_file() else {}
    budget = read_json_object(budget_path) if budget_path.is_file() else {}
    return {
        "selection_integration_policy": config.selection_integration_policy,
        "selection_integration_config_hash": _selection_integration_hash(config),
        "selection_integration_status": (
            validation.get("status")
            if validation
            else "not_enabled"
            if config.selection_integration_policy == GLOBAL_ONLY_SELECTION_POLICY
            else "not_run"
        ),
        "c5_selection_path": (
            str(output_dir / "c6" / "multi-role-selection")
            if config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
            else None
        ),
        "corridor_coverage_report_path": (
            str(coverage_path) if coverage_path.is_file() else None
        ),
        "corridor_coverage_report": coverage or None,
        "c6_integrated_validation": validation or None,
        "c6_authority_manifest_path": str(authority_path) if authority else None,
        "c6_authority_manifest": authority or None,
        "score_pass_authority_hash": authority.get("score_pass_authority_hash"),
        "external_authority_override_used": authority.get(
            "external_authority_override_used"
        ),
        "full_teacher_pass_count": (
            1
            if config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
            else None
        ),
        "selection_budget_diagnostics_path": str(budget_path) if budget else None,
        "selection_budget_diagnostics": budget or None,
    }


def _artifact_summary(output_dir: Path, run_plan: dict[str, Any]) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    metadata = read_json_object(metadata_path) if metadata_path.is_file() else {}
    artifact_estimates = run_plan.get("artifact_estimates", {})
    if not isinstance(artifact_estimates, dict):
        artifact_estimates = {}
    return {
        "num_examples": metadata.get("num_examples"),
        "shard_count": metadata.get("shard_count"),
        "target_type": metadata.get("target_type"),
        "dtype": metadata.get("dtype"),
        "estimated_artifact_bytes": artifact_estimates.get(
            "estimated_total_artifact_bytes"
        ),
        "actual_shard_bytes": _actual_shard_bytes(output_dir),
    }


def _actual_shard_bytes(output_dir: Path) -> int | None:
    shards_dir = output_dir / "shards"
    if not shards_dir.is_dir():
        return None
    return sum(path.stat().st_size for path in shards_dir.glob("shard-*.npz"))


def _selected_exemplar_delivery_enabled(config: ProductionBuildConfig) -> bool:
    return (
        config.exemplar_selection_enabled
        and config.target_policy == "corridor_exemplar_v1"
    )


def _selected_delivery_failure_with_staging(
    config: ProductionBuildConfig,
    c6_context: dict[str, Any] | None,
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    if config.exemplar_delivery_path != "two_pass_rerun_selected":
        return diagnostic
    authority_hash = None
    if c6_context is not None:
        authority_hash = str(
            (c6_context.get("authorities") or {}).get("score_pass_authority_hash") or ""
        )
    failure = dict(diagnostic)
    failure.update(
        selected_delivery_staging_diagnostic(
            config.output_dir,
            delivery_authority_hash=authority_hash,
        )
    )
    return failure


def _exemplar_delivery_config(
    config: ProductionBuildConfig,
    effective_batch_size: int,
    *,
    progress_callback: Any = None,
    authoritative_records: tuple[dict[str, Any], ...] | None = None,
    delivery_authority_hash: str | None = None,
) -> ExemplarDeliveryConfig:
    return ExemplarDeliveryConfig(
        artifact_dir=config.output_dir,
        dataset_path=config.dataset_path,
        delivery_path=config.exemplar_delivery_path or "one_pass_pruned_candidate",
        selection_enabled=config.exemplar_selection_enabled,
        leaderboard_capacity=config.exemplar_leaderboard_capacity,
        selected_exemplar_budget=config.selected_exemplar_budget,
        selected_exemplar_fraction=config.selected_exemplar_fraction,
        retain_unselected_exemplar_payloads=config.retain_unselected_exemplar_payloads,
        score_policy=config.exemplar_score_policy,
        sequence_length=config.sequence_length,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        max_examples=config.max_examples,
        backend_config=_backend_config(config),
        selected_rerun_batch_size=(
            config.selected_rerun_batch_size or effective_batch_size
        ),
        track_timing=config.track_delivery_timing,
        long_tail_warning_k=config.long_tail_warning_k,
        very_long_tail_warning_k=config.very_long_tail_warning_k,
        perverse_tail_warning_k=config.perverse_tail_warning_k,
        reject_perverse_exemplars=config.reject_perverse_exemplars,
        primary_selected_exemplar_budget=(
            config.primary_selected_exemplar_budget
            if config.primary_selected_exemplar_budget is not None
            else config.selected_exemplar_budget
        ),
        long_tail_side_board_cap=config.long_tail_side_board_cap,
        perverse_tail_side_board_cap=config.perverse_tail_side_board_cap,
        include_long_tail_in_primary=config.include_long_tail_in_primary,
        include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        include_perverse_tail_in_student=config.include_perverse_tail_in_student,
        progress_callback=progress_callback,
        authoritative_selection=authoritative_records is not None,
        authoritative_records=authoritative_records,
        execution_mode=(
            "native_c6_path_b_v1"
            if authoritative_records is not None and _native_c6_path_b_enabled(config)
            else "legacy_delivery_v1"
        ),
        rerun_metrics={},
        delivery_authority_hash=delivery_authority_hash,
    )


def _exemplar_capture_mode(config: ProductionBuildConfig) -> str:
    if config.exemplar_delivery_path == "two_pass_rerun_selected":
        return "two_pass_sparse_exemplar"
    return "one_pass_candidate"


def _native_c6_path_b_enabled(config: ProductionBuildConfig) -> bool:
    return (
        config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
        and config.target_policy == "corridor_exemplar_v1"
        and config.exemplar_selection_enabled
        and config.exemplar_delivery_path == "two_pass_rerun_selected"
    )


def _filter_fulfilled_delivery_warnings(warnings: list[str]) -> list[str]:
    fulfilled = {
        "one-pass candidate capture is not a final corpus-global selector",
        "two-pass sparse exemplar capture may require a rerun/selected pass",
    }
    return [warning for warning in warnings if warning not in fulfilled]


def _production_timing_fields(
    config: ProductionBuildConfig,
    *,
    started_at: str,
    completed_at: str,
    production_wall_seconds: float,
    preflight_wall_seconds: float,
    main_pass_wall_seconds: float,
    validation_wall_seconds: float,
    delivery_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not config.track_delivery_timing:
        return None
    fields: dict[str, Any] = {
        "timing_enabled": True,
        "production_started_at": started_at,
        "production_completed_at": completed_at,
        "production_wall_seconds": production_wall_seconds,
        "preflight_wall_seconds": preflight_wall_seconds,
        "score_or_main_pass_wall_seconds": main_pass_wall_seconds,
        "validation_wall_seconds": validation_wall_seconds,
    }
    if delivery_report is not None:
        for key in (
            "selection_wall_seconds",
            "selected_payload_materialization_wall_seconds",
            "pruning_wall_seconds",
            "teacher_rerun_wall_seconds",
            "teacher_rerun_examples_per_second",
            "path_a_wall_seconds",
            "path_b_wall_seconds",
            "examples_per_second",
            "selected_payloads_per_second",
        ):
            if key in delivery_report:
                fields[key] = delivery_report[key]
    return fields


def _planned_example_count(plan: dict[str, Any]) -> int | None:
    artifact_estimates = plan.get("artifact_estimates", {})
    if not isinstance(artifact_estimates, dict):
        return None
    value = artifact_estimates.get("num_examples_effective")
    return int(value) if value is not None else None


def _planned_shard_count(
    config: ProductionBuildConfig,
    plan: dict[str, Any],
) -> int | None:
    examples = _planned_example_count(plan)
    if examples is None:
        return None
    shard_size = max(1, int(config.shard_size_examples))
    return (examples + shard_size - 1) // shard_size


def _rate(count: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return round(float(count) / elapsed, 3)


def _eta(remaining: int, rate: float) -> float | None:
    if remaining <= 0:
        return 0.0
    if rate <= 0:
        return None
    return round(float(remaining) / rate, 3)


def _format_eta(value: object) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.1f}"


def _elapsed(started_at: float) -> float:
    return max(0.0, perf_counter() - started_at)


def _host_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _effective_batch_size(plan: dict[str, Any]) -> int | None:
    resolved = plan.get("resolved_batch_policy", {})
    if not isinstance(resolved, dict):
        return None
    value = resolved.get("effective_gpu_batch_size")
    return int(value) if value is not None else None


def _validation_status(output_dir: Path) -> str | None:
    report_path = output_dir / "validation_report.json"
    if not report_path.is_file():
        return None
    return read_json_object(report_path).get("status")


def _already_complete(config: ProductionBuildConfig) -> bool:
    manifest_path = _run_manifest_path(config)
    if not config.resume or not manifest_path.is_file():
        return False
    return read_json_object(manifest_path).get("status") == "complete"


def _c6_finalization_pending(config: ProductionBuildConfig) -> bool:
    """A completed score pass may still need C2-C5 or selected delivery."""

    if config.selection_integration_policy != C6_SELECTION_INTEGRATION_POLICY:
        return False
    validation_path = (
        config.output_dir / "reports" / "c6_integrated_selection_validation.json"
    )
    if not validation_path.is_file():
        return True
    return read_json_object(validation_path).get("status") != "pass"


def _completed_selected_delivery(config: ProductionBuildConfig) -> bool:
    if not _selected_exemplar_delivery_enabled(config):
        return False
    try:
        report = read_json_object(config.output_dir / "delivery_report.json")
    except (OSError, ValueError):
        return False
    if report.get("status") != "pass":
        return False
    return (
        config.output_dir / "leaderboards" / "selected_exemplars.json"
    ).is_file() and any(
        (config.output_dir / "selected_exemplars").glob("selected-exemplars-*.json")
    )


def _resume_c6_finalization(
    config: ProductionBuildConfig,
    *,
    created_at: str,
    production_started: float,
    report_path: Path,
    parity_report_path: Path,
    progress: Any,
) -> dict[str, Any]:
    """Finish C6 from a complete score/delivery surface without teacher work."""

    blockers: list[str] = []
    warnings: list[str] = []
    output = config.output_dir
    try:
        authority_payload = read_json_object(output / "c6" / "authority_manifest.json")
        paths = authority_payload["paths"]
        authorities = {
            "feature_path": output / str(paths["corridor_features"]),
            "global_board_supply_path": output / str(paths["global_board_supply"]),
            "source_passports_path": output / str(paths["source_passports"]),
            "authority_manifest_path": output / "c6" / "authority_manifest.json",
            "score_pass_authority_hash": authority_payload["score_pass_authority_hash"],
            "external_authority_override_used": authority_payload.get(
                "external_authority_override_used", False
            ),
        }
        progress.stage("c6_finalization_resume")
        context = _prepare_c6_selection(config, authorities)
        delivery_report = read_json_object(output / "delivery_report.json")
        progress.validation_started()
        validation = validate_teacher_textbook(output)
        write_teacher_textbook_validation_report(
            validation,
            output / "validation_report.json",
        )
        if validation.status != "pass":
            blockers.extend(validation.blockers)
        progress.validation_completed(validation.status)
        progress.stage("selected_linkage_audit")
        linkage_audit = audit_selected_linkage(output, strict=True)
        write_selected_linkage_audit(
            linkage_audit,
            output / "selected_linkage_audit.json",
        )
        c6_validation, c6_coverage = _finalize_c6_selection(
            config,
            context,
            delivery_report=delivery_report,
            audit_report=linkage_audit.to_dict(),
        )
        progress.stage("c6_integration_reconciliation")
        audit_payload = linkage_audit.to_dict()
        audit_payload["c6_integration"] = {
            "status": c6_validation["status"],
            "selected_unique_count": c6_validation["selected_unique_count"],
            "selected_obligation_count": c6_validation["selected_obligation_count"],
            "coordinate_set_authority": "c5",
        }
        write_json(output / "selected_linkage_audit.json", audit_payload)
        (output / "reports").mkdir(parents=True, exist_ok=True)
        write_json(
            output / "reports" / "c6_integrated_selection_validation.json",
            c6_validation,
        )
        write_corridor_coverage_report(
            c6_coverage,
            output / "reports" / "fingerprint_corridor_coverage.json",
        )
        if c6_validation["status"] == "fail":
            blockers.extend(str(item) for item in c6_validation["blockers"])
        if linkage_audit.status != "pass":
            blockers.append("selected-linkage audit status is fail")
        if validation.status == "pass" and not blockers:
            progress.stage("cover_page")
            write_cover_page(output)
        existing_report = read_json_object(report_path) if report_path.is_file() else {}
        original_manifest = read_json_object(output / "run_manifest.json")
        run_plan_path = _run_plan_path(config)
        run_plan = (
            read_json_object(run_plan_path)
            if run_plan_path.is_file()
            else {"status": "not_run"}
        )
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail" if blockers else "pass",
            blockers=blockers,
            warnings=warnings,
            doctor_report=existing_report.get("doctor_summary", {}),
            run_plan_path=run_plan_path,
            run_plan=run_plan,
            effective_batch_size=existing_report.get("effective_batch_size"),
            already_complete=True,
            parity_report_path=parity_report_path,
            parity_status="not_run",
            validation_status=validation.status,
            build_status="resumed_finalization",
            delivery_report=delivery_report,
            timing=_production_timing_fields(
                config,
                started_at=created_at,
                completed_at=_now(),
                production_wall_seconds=_elapsed(production_started),
                preflight_wall_seconds=0.0,
                main_pass_wall_seconds=0.0,
                validation_wall_seconds=0.0,
                delivery_report=delivery_report,
            ),
        )
        report["teacher_pass_resumed"] = False
        report["resume_finalization_only"] = True
        report.update(
            {
                "selected_rerun_resumed": False,
                "accelerator_required_for_resume": False,
                "accelerator_probe_status": "skipped_finalization_only",
                "doctor_status": "skipped_finalization_only",
                "run_plan_status": "reused_existing_artifact_plan",
                "finalization_runtime_mode": "cpu",
                "original_teacher_runtime_mode": original_manifest.get(
                    "runtime_mode", config.runtime_mode
                ),
                "original_teacher_backend": original_manifest.get(
                    "teacher_backend", config.teacher_backend
                ),
            }
        )
        return _finalize_production_report(report, report_path, progress)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        blockers.append(f"C6 finalization resume failed: {exc}")
        progress.failure(
            "c6_finalization_resume",
            {
                "failure_stage": "c6_finalization_resume",
                "failure_reason": str(exc),
            },
        )
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report={},
            run_plan_path=_run_plan_path(config),
            run_plan={"status": "not_run"},
            effective_batch_size=None,
            already_complete=True,
            parity_report_path=parity_report_path,
            parity_status="not_run",
            validation_status="not_run",
            build_status="resume_finalization_failed",
        )
        report["teacher_pass_resumed"] = False
        report["resume_finalization_only"] = True
        report.update(
            {
                "selected_rerun_resumed": False,
                "accelerator_required_for_resume": False,
                "accelerator_probe_status": "skipped_finalization_only",
                "doctor_status": "skipped_finalization_only",
                "run_plan_status": "reused_existing_artifact_plan",
                "finalization_runtime_mode": "cpu",
                "original_teacher_runtime_mode": config.runtime_mode,
                "original_teacher_backend": config.teacher_backend,
            }
        )
        return _finalize_production_report(report, report_path, progress)


def _has_existing_artifact(config: ProductionBuildConfig) -> bool:
    output_dir = config.output_dir
    return any(
        path.exists()
        for path in (
            output_dir / "metadata.json",
            _run_manifest_path(config),
            output_dir / "shards",
        )
    )


def _run_plan_path(config: ProductionBuildConfig) -> Path:
    return config.run_plan_path or config.output_dir / "run_plan.json"


def _run_manifest_path(config: ProductionBuildConfig) -> Path:
    return config.run_manifest_path or config.output_dir / "run_manifest.json"


def _progress_log_path(config: ProductionBuildConfig) -> Path:
    return config.progress_log_path or config.output_dir / "progress_log.jsonl"


def _production_progress_path(config: ProductionBuildConfig) -> Path:
    return config.output_dir / PRODUCTION_PROGRESS_FILENAME


def _production_report_path(config: ProductionBuildConfig) -> Path:
    return (
        config.production_report_path
        or config.output_dir / PRODUCTION_BUILD_REPORT_FILENAME
    )


def _parity_report_path(config: ProductionBuildConfig) -> Path:
    return config.parity_report_path or config.output_dir / "parity_report.json"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _selection_integration_hash(config: ProductionBuildConfig) -> str:
    payload = {
        "selection_integration_policy": config.selection_integration_policy,
        "teacher_model": config.teacher_model,
        "tokenizer_id": config.tokenizer_id or config.teacher_model,
        "dataset_path": str(config.dataset_path),
        "corpus_manifest_path": str(config.corpus_manifest_path),
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "dynamic_top_k_min": config.dynamic_top_k_min,
        "dynamic_top_k_max": config.dynamic_top_k_max,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "selected_rerun_batch_size": config.selected_rerun_batch_size,
        "total_selected_exemplar_budget": config.total_selected_exemplar_budget,
        "fingerprint_corridor_budget_fraction": (
            config.fingerprint_corridor_budget_fraction
        ),
        "fingerprint_corridor_budget_max": config.fingerprint_corridor_budget_max,
        "fingerprint_corridor_mode_cap": config.fingerprint_corridor_mode_cap,
        "fingerprint_corridor_candidate_pool_cap": (
            config.fingerprint_corridor_candidate_pool_cap
        ),
        "require_full_selected_budget": config.require_full_selected_budget,
        "c2_schema": "radjax.c2_corridor_candidate_leaderboards.v1",
        "c3_schema": "radjax.c3_corridor_coverage_plan.v1",
        "c4_schema": "radjax.c4_corridor_global_claims.v1",
        "c5_schema": "radjax.multi_role_selected_exemplar.v1",
        "delivery_path": config.exemplar_delivery_path,
    }
    return _hash_payload(payload)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
