"""Callback-driven early score-surface corridor stage adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import (
    ScoreSurfaceCorridorEvidence,
    StageFailure,
    StageResult,
)

ScorePassValueT = TypeVar("ScorePassValueT")
EarlyCorridorOperation = Callable[
    [CanonicalPathBConfig, ScorePassValueT],
    StageResult[ScoreSurfaceCorridorEvidence],
]

SCORE_SURFACE_CORRIDOR_STAGE = "score_surface_corridor_materialization"


def run_score_surface_corridor_stage(
    config: CanonicalPathBConfig,
    score_pass: StageResult[ScorePassValueT] | None,
    *,
    operation: EarlyCorridorOperation[ScorePassValueT],
) -> StageResult[ScoreSurfaceCorridorEvidence]:
    """Materialize only provisional corridor evidence after the score pass.

    The injected operation owns the existing corridor builder and its artifact
    writes.  This adapter verifies that its returned proof remains provisional
    and therefore cannot claim selected-artifact finalization.
    """

    if score_pass is None or score_pass.status != "pass" or score_pass.value is None:
        return _failed_early_corridor(
            "score_pass_required",
            ("early corridor materialization requires a passing score pass",),
            remediation="complete the streaming score pass before corridor export",
        )
    try:
        result = operation(config, score_pass.value)
    except Exception as exc:
        return _failed_early_corridor(
            "score_surface_corridor_operation_failed",
            (str(exc),),
            remediation="inspect the existing early corridor export failure",
        )
    if not isinstance(result, StageResult):
        return _failed_early_corridor(
            "score_surface_corridor_operation_returned_invalid_result",
            ("early corridor callback must return StageResult",),
            remediation="return typed provisional corridor evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != SCORE_SURFACE_CORRIDOR_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_early_corridor(
            "score_surface_corridor_evidence_stage_mismatch",
            (
                "early corridor callback must return evidence with "
                f"stage={SCORE_SURFACE_CORRIDOR_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind corridor evidence to the provisional early stage",
        )
    if not isinstance(value, ScoreSurfaceCorridorEvidence):
        return _failed_early_corridor(
            "score_surface_corridor_value_invalid",
            ("early corridor callback must return ScoreSurfaceCorridorEvidence",),
            remediation="derive the provisional evidence from existing corridor JSON",
        )
    if (
        value.stage_evidence != evidence
        or value.selected_exemplar_count != 0
        or value.selected_exemplars_linked
    ):
        return _failed_early_corridor(
            "score_surface_corridor_not_provisional",
            (
                "early corridor evidence must retain zero selected exemplars "
                "and no selected-link claim",
            ),
            remediation=(
                "do not use selected-artifact corridor evidence before delivery"
            ),
        )
    return result


def _failed_early_corridor(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[ScoreSurfaceCorridorEvidence]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=SCORE_SURFACE_CORRIDOR_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
