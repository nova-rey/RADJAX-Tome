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
    EXEMPLAR_DELIVERY_REPORT_FILENAME,
    ExemplarDeliveryConfig,
    SelectedExemplarDeliveryError,
    materialize_selected_exemplar_delivery,
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
from radjax_tome.corpora import validate_corpus_artifact
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


def build_production_gpu_tome(config: ProductionBuildConfig) -> dict[str, Any]:
    created_at = _now()
    production_started = perf_counter()
    preflight_started = perf_counter()
    report_path = _production_report_path(config)
    run_plan_path = _run_plan_path(config)
    parity_report_path = _parity_report_path(config)
    progress = _ProductionProgressReporter(
        enabled=config.progress,
        output_dir=config.output_dir,
        path=_production_progress_path(config),
    )
    progress.start()
    blockers: list[str] = []
    warnings: list[str] = []
    c6_context: dict[str, Any] | None = None
    already_complete = _already_complete(config)

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
        return _finalize_production_report(report, report_path, progress)

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
        return _finalize_production_report(report, report_path, progress)
    if already_complete and not _c6_finalization_pending(config):
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
            return _finalize_production_report(report, report_path, progress)
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
        return _finalize_production_report(report, report_path, progress)
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
        return _finalize_production_report(report, report_path, progress)

    preflight_wall_seconds = _elapsed(preflight_started)
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
        return _finalize_production_report(report, report_path, progress)
    main_pass_wall_seconds = _elapsed(main_pass_started)
    progress.memory_checkpoint("score_pass_complete")

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
                )
            )
        except SelectedExemplarDeliveryError as exc:
            selected_delivery_failure = exc.diagnostic
            blockers.append(str(exc))
        except Exception as exc:
            selected_delivery_failure = {
                "failure_stage": "selected_exemplar_delivery",
                "failure_reason": str(exc),
                "delivery_path": config.exemplar_delivery_path,
            }
            blockers.append(str(exc))
    progress.validation_started()
    validation_started = perf_counter()
    validation = validate_teacher_textbook(config.output_dir)
    write_teacher_textbook_validation_report(
        validation,
        config.output_dir / "validation_report.json",
    )
    validation_wall_seconds = _elapsed(validation_started)
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
            config.output_dir / "reports" / "c6_integrated_selection_validation.json",
            c6_validation,
        )
        if c6_validation["status"] == "fail":
            blockers.extend(c6_validation["blockers"])
        write_corridor_coverage_report(
            c6_coverage,
            config.output_dir / "reports" / "fingerprint_corridor_coverage.json",
        )
    write_cover_page(config.output_dir)
    parity_status = "not_run"
    if config.parity_left is not None and validation.status == "pass":
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

    if validation.status != "pass":
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
        validation_status=validation.status,
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


def _export_c6_selection_authorities(
    config: ProductionBuildConfig,
) -> dict[str, Path | str | bool]:
    """Derive every C6 authority from the completed Stage 1 score surface."""

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
    feature_path = export_corridor_candidate_features(
        artifact_dir=config.output_dir,
        output_dir=c6_root / "corridor-features",
    )
    feature_manifest_path = c6_root / "corridor-features" / "manifest.json"
    feature_manifest = read_json_object(feature_manifest_path)
    feature_manifest["score_pass_authority_hash"] = score_pass_authority_hash
    write_json(feature_manifest_path, feature_manifest)
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
    global_supply: Mapping[str, Any],
) -> dict[str, Any]:
    requested = int(config.total_selected_exemplar_budget or 0)
    final_count = len(claims.selected_coordinates)
    corridor_candidates = sum(len(mode.candidates) for mode in leaderboards.modes)
    global_candidates = sum(
        len(board.get("candidates") or [])
        for board in global_supply.get("boards", [])
        if isinstance(board, Mapping)
    )
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
    elif corridor_candidates + global_candidates < requested:
        reason = "insufficient_eligible_unique_candidates"
    elif global_claims < requested - corridor_claims:
        reason = "global_ranked_supply_exhaustion"
    elif collisions:
        reason = "deduplication_overlap_exhaustion"
    else:
        reason = "fingerprint_corridor_allocation_or_cap_exhaustion"
    return {
        "total_budget_requested": requested,
        "fingerprint_corridor_budget_requested": len(claims.corridor_claims),
        "fingerprint_corridor_candidates_eligible_unique": corridor_candidates,
        "fingerprint_corridor_claims_before_dedup": corridor_claims,
        "fingerprint_corridor_claims_accepted": corridor_claims,
        "global_supply_exported": global_candidates,
        "global_candidates_examined": global_examined,
        "global_claims_accepted": global_claims,
        "cross_role_duplicate_count": len(collisions),
        "within_role_duplicate_count": 0,
        "final_unique_selected_count": final_count,
        "budget_shortfall": shortfall,
        "budget_shortfall_reason": reason,
        "global_supply_remaining": max(0, global_candidates - global_examined),
        "fingerprint_corridor_global_intersection_size": len(collisions),
        "fingerprint_corridor_global_jaccard": (
            float(len(collisions)) / float(max(final_count, 1))
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
    payload_path = (
        config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )
    if selected_path.is_file():
        legacy_payload = read_json_object(selected_path)
        legacy = list(legacy_payload.get("selected_exemplars") or [])
    if payload_path.is_file():
        payload_payload = read_json_object(payload_path)
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


def _exemplar_delivery_config(
    config: ProductionBuildConfig,
    effective_batch_size: int,
    *,
    progress_callback: Any = None,
    authoritative_records: tuple[dict[str, Any], ...] | None = None,
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
        selected_rerun_batch_size=effective_batch_size,
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
