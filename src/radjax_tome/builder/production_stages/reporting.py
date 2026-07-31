"""Pure production-report support owned outside the orchestration facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from radjax_tome.reports import (
    TomeParityConfig,
    compare_tome_artifacts,
    write_tome_parity_report,
)


def filter_fulfilled_delivery_warnings(warnings: list[str]) -> list[str]:
    return [
        warning
        for warning in warnings
        if warning != "selected exemplar delivery fulfilled"
    ]


def production_timing_fields(
    *,
    started_at: str,
    completed_at: str,
    production_wall_seconds: float,
    preflight_wall_seconds: float | None,
    main_pass_wall_seconds: float | None,
    validation_wall_seconds: float | None,
    delivery_report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "production_wall_seconds": production_wall_seconds,
        "preflight_wall_seconds": preflight_wall_seconds,
        "main_pass_wall_seconds": main_pass_wall_seconds,
        "validation_wall_seconds": validation_wall_seconds,
        "selected_delivery_wall_seconds": (delivery_report or {}).get("wall_seconds"),
    }


def render_production_build_summary(report: dict[str, Any]) -> list[str]:
    return [
        f"production status: {report.get('status')}",
        f"output: {report.get('output_dir')}",
        f"selected exemplars: {report.get('num_selected_exemplars')}",
    ]


@dataclass(frozen=True)
class FinalReportingOperations:
    """Facade compatibility hooks; no stage imports the façade."""

    report: Callable[..., dict[str, Any]]
    finalize_report: Callable[[dict[str, Any], Any, Any], dict[str, Any]]
    timing_fields: Callable[..., dict[str, Any]]
    now: Callable[[], str]
    filter_warnings: Callable[[list[str]], list[str]] = (
        filter_fulfilled_delivery_warnings
    )


def native_final_reporting_operation(
    state: Any, inputs: Any, *, operations: FinalReportingOperations
) -> Any:
    """Render terminal production evidence after completed M4 proof."""

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
        state.warnings = operations.filter_warnings(state.warnings)
    status = "fail" if state.blockers else "warn" if state.warnings else "pass"
    report = operations.report(
        config,
        created_at=state.created_at,
        completed_at=operations.now(),
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
        timing=operations.timing_fields(
            config,
            started_at=state.created_at,
            completed_at=operations.now(),
            production_wall_seconds=perf_counter() - state.production_started,
            preflight_wall_seconds=state.preflight_wall_seconds,
            main_pass_wall_seconds=state.main_pass_wall_seconds,
            validation_wall_seconds=float(value["validation_wall_seconds"]),
            delivery_report=delivery_report,
        ),
    )
    state.terminal_report = operations.finalize_report(
        report, state.report_path, state.progress
    )
    from radjax_tome.builder.production_stages.evidence import native_file_evidence

    evidence = native_file_evidence(
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
