"""Callback-driven selected rerun and late corridor finalization adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import (
    SelectedArtifactCorridorEvidence,
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.selection import (
    C5_STAGE,
    INTEGRATED_SELECTION_STAGE,
    IntegratedSelectionHandoff,
)

SelectionValueT = TypeVar("SelectionValueT")
SelectedRerunValueT = TypeVar("SelectedRerunValueT")

SELECTED_RERUN_STAGE = "selected_delivery_rerun"
SELECTED_ARTIFACT_CORRIDOR_STAGE = "selected_artifact_corridor_finalization"


@dataclass(frozen=True)
class SelectedDeliveryInputs(Generic[SelectionValueT]):
    """C5 selected coordinates and proof consumed by the selected rerun."""

    selection: SelectionValueT
    selection_evidence: StageEvidence
    c5_evidence: StageEvidence


@dataclass(frozen=True)
class SelectedRerunHandoff(Generic[SelectedRerunValueT]):
    """In-memory selected-rerun result and its evidence for late finalization."""

    value: SelectedRerunValueT
    stage_evidence: StageEvidence

    def __post_init__(self) -> None:
        if self.stage_evidence.stage != SELECTED_RERUN_STAGE:
            raise ValueError(
                f"selected rerun handoff must describe stage={SELECTED_RERUN_STAGE!r}"
            )


SelectedRerunOperation = Callable[
    [CanonicalPathBConfig, SelectedDeliveryInputs[SelectionValueT]],
    StageResult[SelectedRerunHandoff[SelectedRerunValueT]],
]


@dataclass(frozen=True)
class LateCorridorInputs(Generic[SelectionValueT, SelectedRerunValueT]):
    """C5 and rerun evidence consumed by selected-artifact finalization."""

    selection: SelectionValueT
    selection_evidence: StageEvidence
    c5_evidence: StageEvidence
    selected_rerun: SelectedRerunValueT
    selected_rerun_evidence: StageEvidence


LateCorridorOperation = Callable[
    [
        CanonicalPathBConfig,
        LateCorridorInputs[SelectionValueT, SelectedRerunValueT],
    ],
    StageResult[SelectedArtifactCorridorEvidence],
]


def run_selected_delivery_rerun_stage(
    config: CanonicalPathBConfig,
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    ),
    *,
    operation: SelectedRerunOperation[SelectionValueT, SelectedRerunValueT],
) -> StageResult[SelectedRerunHandoff[SelectedRerunValueT]]:
    """Run selected-only delivery from passing C5 selection evidence."""

    delivery_inputs = _selected_delivery_inputs(integrated_selection)
    if isinstance(delivery_inputs, StageFailure):
        return _failed_rerun(
            delivery_inputs.reason,
            delivery_inputs.blockers,
            remediation=delivery_inputs.remediation or "complete C2-C5 selection first",
        )
    try:
        result = operation(config, delivery_inputs)
    except Exception as exc:
        return _failed_rerun(
            "selected_delivery_rerun_operation_failed",
            (str(exc),),
            remediation="inspect the existing selected rerun failure",
        )
    if not isinstance(result, StageResult):
        return _failed_rerun(
            "selected_delivery_rerun_operation_returned_invalid_result",
            ("selected rerun callback must return StageResult",),
            remediation="return typed selected-rerun value and evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != SELECTED_RERUN_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_rerun(
            "selected_delivery_rerun_evidence_stage_mismatch",
            (
                "selected rerun callback must return evidence with "
                f"stage={SELECTED_RERUN_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind selected rerun output to its explicit stage",
        )
    if not isinstance(value, SelectedRerunHandoff):
        return _failed_rerun(
            "selected_delivery_rerun_value_invalid",
            ("selected rerun callback must return SelectedRerunHandoff",),
            remediation="retain selected rerun evidence for late finalization",
        )
    if value.stage_evidence != evidence:
        return _failed_rerun(
            "selected_delivery_rerun_handoff_evidence_mismatch",
            ("selected rerun handoff evidence must match result evidence",),
            remediation="return one consistent selected rerun evidence object",
        )
    return result


def run_selected_artifact_corridor_finalization_stage(
    config: CanonicalPathBConfig,
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    ),
    selected_rerun: StageResult[SelectedRerunHandoff[SelectedRerunValueT]] | None,
    *,
    operation: LateCorridorOperation[SelectionValueT, SelectedRerunValueT],
) -> StageResult[SelectedArtifactCorridorEvidence]:
    """Finalize public corridors after selected rerun, never from early proof."""

    delivery_inputs = _selected_delivery_inputs(integrated_selection)
    if isinstance(delivery_inputs, StageFailure):
        return _failed_late_corridor(
            delivery_inputs.reason,
            delivery_inputs.blockers,
            remediation=delivery_inputs.remediation or "complete C2-C5 selection first",
        )
    if (
        selected_rerun is None
        or selected_rerun.status != "pass"
        or selected_rerun.value is None
        or selected_rerun.evidence is None
        or selected_rerun.evidence.stage != SELECTED_RERUN_STAGE
        or selected_rerun.value.stage_evidence != selected_rerun.evidence
    ):
        return _failed_late_corridor(
            "selected_delivery_rerun_required",
            ("late corridor finalization requires passing selected rerun evidence",),
            remediation="complete selected delivery rerun before late corridor export",
        )
    inputs = LateCorridorInputs(
        selection=delivery_inputs.selection,
        selection_evidence=delivery_inputs.selection_evidence,
        c5_evidence=delivery_inputs.c5_evidence,
        selected_rerun=selected_rerun.value.value,
        selected_rerun_evidence=selected_rerun.evidence,
    )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_late_corridor(
            "selected_artifact_corridor_operation_failed",
            (str(exc),),
            remediation="inspect the existing late corridor finalization failure",
        )
    if not isinstance(result, StageResult):
        return _failed_late_corridor(
            "selected_artifact_corridor_operation_returned_invalid_result",
            ("late corridor callback must return StageResult",),
            remediation="return selected-linked final corridor evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != SELECTED_ARTIFACT_CORRIDOR_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_late_corridor(
            "selected_artifact_corridor_evidence_stage_mismatch",
            (
                "late corridor callback must return evidence with "
                f"stage={SELECTED_ARTIFACT_CORRIDOR_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind late selected-artifact evidence to its explicit stage",
        )
    if not isinstance(value, SelectedArtifactCorridorEvidence):
        return _failed_late_corridor(
            "selected_artifact_corridor_value_invalid",
            ("late corridor callback must return SelectedArtifactCorridorEvidence",),
            remediation="derive selected-linked evidence from finalized corridors",
        )
    if (
        value.stage_evidence != evidence
        or value.selected_exemplar_count < 1
        or not value.selected_exemplars_linked
    ):
        return _failed_late_corridor(
            "selected_artifact_corridor_not_selected_linked",
            (
                "late corridor evidence must retain selected exemplars and "
                "selected linkage",
            ),
            remediation="do not present provisional early corridor evidence as final",
        )
    return result


def _selected_delivery_inputs(
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    ),
) -> SelectedDeliveryInputs[SelectionValueT] | StageFailure:
    if (
        integrated_selection is None
        or integrated_selection.status != "pass"
        or integrated_selection.value is None
        or integrated_selection.evidence is None
        or integrated_selection.evidence.stage != INTEGRATED_SELECTION_STAGE
        or integrated_selection.value.stage_evidence != integrated_selection.evidence
        or integrated_selection.value.c5_evidence.stage != C5_STAGE
    ):
        return StageFailure(
            stage=SELECTED_RERUN_STAGE,
            reason="integrated_selection_required",
            blockers=("selected delivery requires passing C5 selection evidence",),
            resumable=True,
            remediation="complete integrated C2-C5 selection before delivery",
        )
    return SelectedDeliveryInputs(
        selection=integrated_selection.value.value,
        selection_evidence=integrated_selection.evidence,
        c5_evidence=integrated_selection.value.c5_evidence,
    )


def _failed_rerun(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[SelectedRerunHandoff[SelectedRerunValueT]]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=SELECTED_RERUN_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )


def _failed_late_corridor(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[SelectedArtifactCorridorEvidence]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=SELECTED_ARTIFACT_CORRIDOR_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
