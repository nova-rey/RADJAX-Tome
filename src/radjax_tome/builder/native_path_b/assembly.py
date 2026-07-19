"""Callback-driven artifact assembly adapter after final selected corridors."""

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
from radjax_tome.builder.native_path_b.delivery import (
    SELECTED_ARTIFACT_CORRIDOR_STAGE,
    SELECTED_RERUN_STAGE,
    SelectedRerunHandoff,
)
from radjax_tome.builder.native_path_b.selection import (
    C5_STAGE,
    INTEGRATED_SELECTION_STAGE,
    IntegratedSelectionHandoff,
)

SelectionValueT = TypeVar("SelectionValueT")
SelectedRerunValueT = TypeVar("SelectedRerunValueT")
AssemblyValueT = TypeVar("AssemblyValueT")

ARTIFACT_ASSEMBLY_STAGE = "artifact_assembly"


@dataclass(frozen=True)
class ArtifactAssemblyInputs(Generic[SelectionValueT, SelectedRerunValueT]):
    """Selected delivery and final corridor proof consumed by assembly."""

    selection: SelectionValueT
    selection_evidence: StageEvidence
    c5_evidence: StageEvidence
    selected_rerun: SelectedRerunValueT
    selected_rerun_evidence: StageEvidence
    final_corridor: SelectedArtifactCorridorEvidence
    final_corridor_evidence: StageEvidence


@dataclass(frozen=True)
class ArtifactAssemblyHandoff(Generic[AssemblyValueT]):
    """Typed assembled artifact handoff for later validation and reporting."""

    value: AssemblyValueT
    stage_evidence: StageEvidence

    def __post_init__(self) -> None:
        if self.stage_evidence.stage != ARTIFACT_ASSEMBLY_STAGE:
            raise ValueError(
                "artifact assembly handoff must describe "
                f"stage={ARTIFACT_ASSEMBLY_STAGE!r}"
            )


ArtifactAssemblyOperation = Callable[
    [
        CanonicalPathBConfig,
        ArtifactAssemblyInputs[SelectionValueT, SelectedRerunValueT],
    ],
    StageResult[ArtifactAssemblyHandoff[AssemblyValueT]],
]


def run_artifact_assembly_stage(
    config: CanonicalPathBConfig,
    *,
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    ),
    selected_rerun: StageResult[SelectedRerunHandoff[SelectedRerunValueT]] | None,
    final_corridor: StageResult[SelectedArtifactCorridorEvidence] | None,
    operation: ArtifactAssemblyOperation[
        SelectionValueT,
        SelectedRerunValueT,
        AssemblyValueT,
    ],
) -> StageResult[ArtifactAssemblyHandoff[AssemblyValueT]]:
    """Assemble only after passing C5, rerun, and final selected corridor proof."""

    inputs = _assembly_inputs(integrated_selection, selected_rerun, final_corridor)
    if isinstance(inputs, StageFailure):
        return _failed_assembly(
            inputs.reason,
            inputs.blockers,
            remediation=(
                inputs.remediation or "complete delivery and final corridor stages"
            ),
        )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_assembly(
            "artifact_assembly_operation_failed",
            (str(exc),),
            remediation="inspect the existing artifact assembly failure",
        )
    if not isinstance(result, StageResult):
        return _failed_assembly(
            "artifact_assembly_operation_returned_invalid_result",
            ("artifact assembly callback must return StageResult",),
            remediation="return typed assembly handoff and evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != ARTIFACT_ASSEMBLY_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_assembly(
            "artifact_assembly_evidence_stage_mismatch",
            (
                "artifact assembly callback must return evidence with "
                f"stage={ARTIFACT_ASSEMBLY_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind assembled artifact evidence to its explicit stage",
        )
    if not isinstance(value, ArtifactAssemblyHandoff):
        return _failed_assembly(
            "artifact_assembly_value_invalid",
            ("artifact assembly callback must return ArtifactAssemblyHandoff",),
            remediation="retain assembly evidence for validation and reporting",
        )
    if value.stage_evidence != evidence:
        return _failed_assembly(
            "artifact_assembly_handoff_evidence_mismatch",
            ("assembly handoff evidence must match result evidence",),
            remediation="return one consistent artifact assembly evidence object",
        )
    return result


def _assembly_inputs(
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    ),
    selected_rerun: StageResult[SelectedRerunHandoff[SelectedRerunValueT]] | None,
    final_corridor: StageResult[SelectedArtifactCorridorEvidence] | None,
) -> ArtifactAssemblyInputs[SelectionValueT, SelectedRerunValueT] | StageFailure:
    if (
        integrated_selection is None
        or integrated_selection.status != "pass"
        or integrated_selection.value is None
        or integrated_selection.evidence is None
        or integrated_selection.evidence.stage != INTEGRATED_SELECTION_STAGE
        or integrated_selection.value.c5_evidence.stage != C5_STAGE
    ):
        return _assembly_failure("integrated_selection_required")
    if (
        selected_rerun is None
        or selected_rerun.status != "pass"
        or selected_rerun.value is None
        or selected_rerun.evidence is None
        or selected_rerun.evidence.stage != SELECTED_RERUN_STAGE
        or selected_rerun.value.stage_evidence != selected_rerun.evidence
    ):
        return _assembly_failure("selected_delivery_rerun_required")
    if (
        final_corridor is None
        or final_corridor.status != "pass"
        or final_corridor.value is None
        or final_corridor.evidence is None
        or final_corridor.evidence.stage != SELECTED_ARTIFACT_CORRIDOR_STAGE
        or final_corridor.value.stage_evidence != final_corridor.evidence
        or final_corridor.value.selected_exemplar_count < 1
        or not final_corridor.value.selected_exemplars_linked
    ):
        return _assembly_failure("selected_artifact_corridor_required")
    return ArtifactAssemblyInputs(
        selection=integrated_selection.value.value,
        selection_evidence=integrated_selection.evidence,
        c5_evidence=integrated_selection.value.c5_evidence,
        selected_rerun=selected_rerun.value.value,
        selected_rerun_evidence=selected_rerun.evidence,
        final_corridor=final_corridor.value,
        final_corridor_evidence=final_corridor.evidence,
    )


def _assembly_failure(reason: str) -> StageFailure:
    return StageFailure(
        stage=ARTIFACT_ASSEMBLY_STAGE,
        reason=reason,
        blockers=("artifact assembly requires all preceding selected delivery proof",),
        resumable=True,
        remediation="complete the missing upstream selected delivery stage",
    )


def _failed_assembly(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[ArtifactAssemblyHandoff[AssemblyValueT]]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=ARTIFACT_ASSEMBLY_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
