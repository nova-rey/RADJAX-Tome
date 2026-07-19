"""Callback-driven native Path-B streaming score-pass stage adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import StageFailure, StageResult

PreflightValueT = TypeVar("PreflightValueT")
ScorePassValueT = TypeVar("ScorePassValueT")
ScorePassOperation = Callable[
    [CanonicalPathBConfig, PreflightValueT], StageResult[ScorePassValueT]
]

SCORE_PASS_STAGE = "score_pass"


def run_score_pass_stage(
    config: CanonicalPathBConfig,
    preflight: StageResult[PreflightValueT],
    *,
    operation: ScorePassOperation[PreflightValueT, ScorePassValueT],
    propagate_exceptions: bool = False,
) -> StageResult[ScorePassValueT]:
    """Run the injected existing streaming score pass after typed preflight."""

    if preflight.status != "pass" or preflight.value is None:
        return _failed_score_pass(
            "preflight_required",
            ("score pass cannot run until typed preflight evidence passes",),
            remediation="resolve the preflight failure before starting the score pass",
        )
    try:
        result = operation(config, preflight.value)
    except Exception as exc:
        if propagate_exceptions:
            raise
        return _failed_score_pass(
            "score_pass_operation_failed",
            (str(exc),),
            remediation="inspect the existing streaming score-pass failure",
        )
    if not isinstance(result, StageResult):
        return _failed_score_pass(
            "score_pass_operation_returned_invalid_result",
            ("score-pass callback must return StageResult",),
            remediation="return typed score-pass value and StageEvidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    if evidence is None or evidence.stage != SCORE_PASS_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_score_pass(
            "score_pass_evidence_stage_mismatch",
            (
                "score-pass callback must return evidence with "
                f"stage={SCORE_PASS_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind the existing score-pass evidence to the score_pass stage",
        )
    return result


def _failed_score_pass(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[ScorePassValueT]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=SCORE_PASS_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
