"""Preflight validation primitives for production builds.

Terminal-report policy remains in the compatibility facade; this leaf module
owns the actual input and provenance checks and never reaches upward into it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from radjax_tome.builder.c6_integration import (
    C6_SELECTION_INTEGRATION_POLICY,
    GLOBAL_ONLY_SELECTION_POLICY,
)
from radjax_tome.corpora import validate_corpus_artifact
from radjax_tome.provenance import validate_teacher_model_provenance
from radjax_tome.reports import (
    GPURunPlanConfig,
    build_gpu_run_plan,
    build_runtime_doctor_report,
    write_gpu_run_plan,
)


def validate_required_inputs(config: Any, blockers: list[str]) -> None:
    replay_path = getattr(config, "verified_selection_replay_path", None)
    replay_manifest = getattr(config, "verified_selection_bundle_manifest_path", None)
    if (replay_path is None) != (replay_manifest is None):
        blockers.append(
            "verified selection replay requires both replay root and bundle manifest"
        )
    if replay_path is not None and replay_manifest is not None:
        if not replay_path.is_dir() or replay_path.is_symlink():
            blockers.append("verified selection replay root is not a safe directory")
        if not replay_manifest.is_file() or replay_manifest.is_symlink():
            blockers.append("verified selection bundle manifest is not a safe file")
        if config.c4_claims_path is not None or config.c5_selection_path is not None:
            blockers.append(
                "verified selection replay cannot be combined with external "
                "C4/C5 overrides"
            )
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
        for label, path in (
            ("source passports", config.source_passports_path),
            ("global board supply", config.global_board_supply_path),
        ):
            if path is not None and not path.is_file():
                blockers.append(f"{label} override path missing: {path}")
        if config.corridor_feature_jsonl_path is not None:
            blockers.append(
                "C6 derives strict corridor features from the current packed corridor "
                "artifact; --corridor-feature-jsonl is not accepted"
            )
        if config.c4_claims_path is not None or config.c5_selection_path is not None:
            blockers.append(
                "C6 production rebuilds C4/C5 from the current artifact; external "
                "C4/C5 checkpoints are not accepted"
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
    if blockers:
        return
    corpus_report = validate_corpus_artifact(config.corpus_manifest_path.parent)
    blockers.extend(
        f"corpus manifest invalid: {item}" for item in corpus_report.blockers
    )
    teacher_report = validate_teacher_model_provenance(
        config.teacher_model_provenance_path
    )
    blockers.extend(
        f"teacher model provenance invalid: {item}" for item in teacher_report.blockers
    )


@dataclass(frozen=True)
class PreflightOperations:
    """Compatibility/report policy injected by the public production facade."""

    report: Callable[..., dict[str, Any]]
    record_terminal_report: Callable[[Any, dict[str, Any]], None]
    terminal_stage_failure: Callable[[Any, str], Any]
    stage_success: Callable[..., Any]
    now: Callable[[], str]
    migrate_metadata: Callable[[Any], Any]
    probe_finalization_resume: Callable[[Any], Any]
    has_existing_artifact: Callable[[Any], bool]
    finalization_pending: Callable[[Any], bool]
    resume_finalization: Callable[..., dict[str, Any]]
    backend_config: Callable[[Any], Any]
    effective_batch_size: Callable[[dict[str, Any]], int | None]
    runtime_doctor: Callable[[Any], dict[str, Any]] = build_runtime_doctor_report
    build_plan: Callable[[Any], dict[str, Any]] = build_gpu_run_plan
    write_plan: Callable[[dict[str, Any], Any], None] = write_gpu_run_plan


@dataclass(frozen=True)
class ProductionPreflightAssessment:
    """Mutation-free destination decision used before production execution."""

    status: str
    destination: Path
    destination_state: str
    action: str
    blockers: tuple[str, ...] = ()


def assess_production_preflight(
    output_dir: Path, *, resume: bool = False, overwrite: bool = False
) -> ProductionPreflightAssessment:
    """Classify a destination without creating, deleting, or writing anything."""
    if resume and overwrite:
        return ProductionPreflightAssessment(
            "fail",
            output_dir,
            "invalid",
            "reject",
            ("resume and overwrite are mutually exclusive",),
        )
    candidate = output_dir if output_dir.is_absolute() else output_dir.absolute()
    if candidate in {Path("/"), Path.home(), Path.cwd()} or candidate.is_symlink():
        return ProductionPreflightAssessment(
            "fail", candidate, "unsafe", "reject", ("destination is unsafe",)
        )
    if not candidate.exists():
        state = "missing"
    elif candidate.is_file() or not candidate.is_dir():
        state = "special"
    elif any(candidate.iterdir()):
        state = "nonempty_directory"
    else:
        state = "empty_directory"
    if state == "special":
        return ProductionPreflightAssessment(
            "fail", candidate, state, "reject", ("destination is not a directory",)
        )
    if state == "missing":
        return ProductionPreflightAssessment(
            "fail" if resume else "pass",
            candidate,
            state,
            "reject" if resume else "create",
            ("cannot resume a missing destination",) if resume else (),
        )
    if state == "empty_directory":
        return ProductionPreflightAssessment(
            "fail" if resume else "pass",
            candidate,
            state,
            "reject" if resume else "use",
            ("cannot resume an empty destination",) if resume else (),
        )
    markers = ("run_manifest.json", "production_build_report.json", "metadata.json")
    if not any((candidate / marker).is_file() for marker in markers):
        return ProductionPreflightAssessment(
            "fail",
            candidate,
            state,
            "reject",
            ("destination contains no canonical Tome ownership marker",),
        )
    if not (resume or overwrite):
        return ProductionPreflightAssessment(
            "fail",
            candidate,
            state,
            "reject",
            ("destination contains existing entries",),
        )
    return ProductionPreflightAssessment(
        "pass", candidate, state, "resume" if resume else "replace"
    )


def _failure_report(
    state: Any,
    operations: PreflightOperations,
    *,
    doctor_report: dict[str, Any],
    run_plan: dict[str, Any],
    effective_batch_size: int | None,
    validation_status: str | None = None,
    build_status: str | None = None,
) -> Any:
    report = operations.report(
        state.config,
        created_at=state.created_at,
        completed_at=operations.now(),
        status="fail",
        blockers=state.blockers,
        warnings=state.warnings,
        doctor_report=doctor_report,
        run_plan_path=state.run_plan_path,
        run_plan=run_plan,
        effective_batch_size=effective_batch_size,
        already_complete=state.already_complete,
        parity_report_path=state.parity_report_path,
        parity_status="not_run",
        validation_status=validation_status,
        build_status=build_status,
    )
    operations.record_terminal_report(state, report)
    return operations.terminal_stage_failure(state, "preflight")


def run_preflight(state: Any, *, operations: PreflightOperations) -> Any:
    """Run the historical preflight body with façade policy injected.

    This owns validation, plan construction, canonical resume resolution, and
    overwrite gating.  It deliberately does not import the public façade.
    """

    from radjax_tome.builder.native_path_b.api import resolve_canonical_path_b_config
    from radjax_tome.builder.native_path_b.resume import resolve_native_path_b_resume
    from radjax_tome.builder.teacher_textbook import validate_teacher_textbook
    from radjax_tome.io.json import read_json_object

    config = state.config
    validate_required_inputs(config, state.blockers)
    state.progress.memory_checkpoint("preflight_complete")
    if state.blockers:
        return _failure_report(
            state,
            operations,
            doctor_report={},
            run_plan={},
            effective_batch_size=None,
        )
    c6_resume_requested = (
        config.resume
        and config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
        and config.target_policy == "corridor_exemplar_v1"
        and config.exemplar_selection_enabled
        and config.exemplar_delivery_path == "two_pass_rerun_selected"
    )
    if c6_resume_requested:
        state.progress.stage("compatibility_migration")
        migration = operations.migrate_metadata(config)
        if migration.applicable and not migration.applied:
            state.blockers.append(
                "C6.3.5.1 metadata compatibility migration failed: "
                + "; ".join(migration.reasons)
            )
            result = _failure_report(
                state,
                operations,
                doctor_report={},
                run_plan={"status": "not_run"},
                effective_batch_size=None,
            )
            if state.terminal_report is not None:
                state.terminal_report["compatibility_migration"] = migration.to_dict()
            return result
        canonical_config = resolve_canonical_path_b_config(config)
        state.native_resume_resolution = resolve_native_path_b_resume(
            config.output_dir,
            config=canonical_config,
            run_manifest_path=config.run_manifest_path,
        )
        if state.native_resume_resolution.complete:
            operations.record_terminal_report(
                state, read_json_object(state.report_path)
            )
            return operations.terminal_stage_failure(state, "preflight")
    finalization_probe = operations.probe_finalization_resume(config)
    output_has_artifact = operations.has_existing_artifact(config)
    if output_has_artifact and not (config.resume or config.overwrite):
        state.blockers.append("output exists; use --resume or --overwrite")
        return _failure_report(
            state, operations, doctor_report={}, run_plan={}, effective_batch_size=None
        )
    if (
        state.already_complete
        and not operations.finalization_pending(config)
        and not c6_resume_requested
    ):
        validation = validate_teacher_textbook(config.output_dir)
        report = operations.report(
            config,
            created_at=state.created_at,
            completed_at=operations.now(),
            status="pass" if validation.status == "pass" else "fail",
            blockers=[] if validation.status == "pass" else list(validation.blockers),
            warnings=[] if validation.status == "pass" else list(validation.warnings),
            doctor_report={},
            run_plan_path=state.run_plan_path,
            run_plan={"status": "not_run"},
            effective_batch_size=None,
            already_complete=True,
            parity_report_path=state.parity_report_path,
            parity_status="not_run",
            validation_status=validation.status,
            build_status="already_complete"
            if validation.status == "pass"
            else "already_complete_invalid",
        )
        operations.record_terminal_report(state, report)
        return operations.terminal_stage_failure(state, "preflight")
    if finalization_probe.eligible and (
        not c6_resume_requested
        or state.native_resume_resolution is None
        or state.native_resume_resolution.stage
        in {"validation_linkage", "reconciliation_cover", "final_reporting"}
    ):
        state.terminal_report = operations.resume_finalization(
            config,
            created_at=state.created_at,
            production_started=state.production_started,
            report_path=state.report_path,
            parity_report_path=state.parity_report_path,
            progress=state.progress,
        )
        return operations.terminal_stage_failure(state, "preflight")
    if output_has_artifact and config.overwrite:
        shutil.rmtree(config.output_dir)
    backend = operations.backend_config(config)
    doctor_report = operations.runtime_doctor(backend)
    plan = operations.build_plan(
        GPURunPlanConfig(
            backend_config=backend,
            dataset_path=config.dataset_path,
            corpus_manifest_path=config.corpus_manifest_path,
            teacher_model_provenance_path=config.teacher_model_provenance_path,
            max_examples=config.max_examples,
            strict_provenance=config.strict_provenance,
            max_artifact_bytes=config.max_artifact_bytes,
            fail_on_warnings=config.fail_on_plan_warnings,
            selection_integration_policy=config.selection_integration_policy,
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            fingerprint_corridor_budget_fraction=config.fingerprint_corridor_budget_fraction,
            fingerprint_corridor_budget_max=config.fingerprint_corridor_budget_max,
            fingerprint_corridor_mode_cap=config.fingerprint_corridor_mode_cap,
            fingerprint_corridor_candidate_pool_cap=config.fingerprint_corridor_candidate_pool_cap,
            require_full_selected_budget=config.require_full_selected_budget,
        )
    )
    operations.write_plan(plan, state.run_plan_path)
    state.warnings.extend(str(item) for item in plan.get("warnings", ()))
    if str(plan.get("status")) == "fail":
        state.blockers.extend(str(item) for item in plan.get("blockers", ()))
    if str(plan.get("status")) == "warn" and config.no_build_if_plan_warn:
        state.blockers.append(
            "run plan status is warn and no_build_if_plan_warn is enabled"
        )
    effective_batch_size = operations.effective_batch_size(plan)
    if effective_batch_size is None:
        state.blockers.append("run plan did not select an effective batch size")
    if state.blockers:
        return _failure_report(
            state,
            operations,
            doctor_report=doctor_report,
            run_plan=plan,
            effective_batch_size=effective_batch_size,
        )
    state.doctor_report = doctor_report
    state.plan = plan
    state.effective_batch_size = effective_batch_size
    state.preflight_wall_seconds = perf_counter() - state.preflight_started
    return operations.stage_success(state, "preflight", paths=(state.run_plan_path,))
