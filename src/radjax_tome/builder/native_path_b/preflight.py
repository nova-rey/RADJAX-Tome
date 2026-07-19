"""Callback-driven native Path-B preflight stage adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import (
    StageFailure,
    StageResult,
)

PreflightValueT = TypeVar("PreflightValueT")
PreflightOperation = Callable[[CanonicalPathBConfig], StageResult[PreflightValueT]]

PREFLIGHT_STAGE = "preflight"


def run_preflight_stage(
    config: CanonicalPathBConfig,
    *,
    operation: PreflightOperation[PreflightValueT],
) -> StageResult[PreflightValueT]:
    """Run the injected existing preflight operation as the first native stage.

    The callback owns all existing planner, doctor, validation, resume, and
    report behavior.  This adapter only normalizes callback failures and
    requires explicit typed evidence; it has no filesystem side effects.
    """

    try:
        result = operation(config)
    except Exception as exc:
        return _failed_preflight(
            "preflight_operation_failed",
            (str(exc),),
            remediation="inspect the existing preflight operation failure",
        )
    return _validated_stage_result(result, expected_stage=PREFLIGHT_STAGE)


def _validated_stage_result(
    result: object,
    *,
    expected_stage: str,
) -> StageResult[PreflightValueT]:
    if not isinstance(result, StageResult):
        return _failed_preflight(
            "preflight_operation_returned_invalid_result",
            ("preflight callback must return StageResult",),
            remediation="return typed preflight value and StageEvidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    if evidence is None or evidence.stage != expected_stage:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_preflight(
            "preflight_evidence_stage_mismatch",
            (
                "preflight callback must return evidence with "
                f"stage={expected_stage!r}; got {actual_stage!r}",
            ),
            remediation="bind the existing preflight evidence to the preflight stage",
        )
    return result


def _failed_preflight(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[PreflightValueT]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=PREFLIGHT_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
