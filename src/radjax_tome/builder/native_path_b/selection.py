"""Callback-driven C2-C5 integrated selection stage adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.authorities import (
    FINGERPRINT_AUTHORITY_STAGE,
    GLOBAL_AUTHORITY_STAGE,
)
from radjax_tome.builder.native_path_b.contracts import (
    StageEvidence,
    StageFailure,
    StageResult,
)

FingerprintAuthorityValueT = TypeVar("FingerprintAuthorityValueT")
GlobalAuthorityValueT = TypeVar("GlobalAuthorityValueT")
SelectionValueT = TypeVar("SelectionValueT")

INTEGRATED_SELECTION_STAGE = "integrated_selection"
C2_STAGE = "c2_corridor_candidate_leaderboards"
C3_STAGE = "c3_corridor_coverage_plan"
C4_STAGE = "c4_corridor_global_claims"
C5_STAGE = "c5_multi_role_selection"


@dataclass(frozen=True)
class SelectionAuthorityInputs(
    Generic[FingerprintAuthorityValueT, GlobalAuthorityValueT]
):
    """Authority values plus their existing proof consumed by C2-C5."""

    fingerprint_value: FingerprintAuthorityValueT
    fingerprint_evidence: StageEvidence
    global_value: GlobalAuthorityValueT
    global_evidence: StageEvidence


@dataclass(frozen=True)
class IntegratedSelectionHandoff(Generic[SelectionValueT]):
    """Typed C2/C3/C4/C5 handoff for delivery, with no new artifact schema."""

    value: SelectionValueT
    stage_evidence: StageEvidence
    c2_evidence: StageEvidence
    c3_evidence: StageEvidence
    c4_evidence: StageEvidence
    c5_evidence: StageEvidence

    def __post_init__(self) -> None:
        expected_stages = (
            ("stage_evidence", self.stage_evidence, INTEGRATED_SELECTION_STAGE),
            ("c2_evidence", self.c2_evidence, C2_STAGE),
            ("c3_evidence", self.c3_evidence, C3_STAGE),
            ("c4_evidence", self.c4_evidence, C4_STAGE),
            ("c5_evidence", self.c5_evidence, C5_STAGE),
        )
        for name, evidence, expected_stage in expected_stages:
            if evidence.stage != expected_stage:
                raise ValueError(
                    f"{name} must describe stage={expected_stage!r}; "
                    f"got {evidence.stage!r}"
                )


IntegratedSelectionOperation = Callable[
    [
        CanonicalPathBConfig,
        SelectionAuthorityInputs[
            FingerprintAuthorityValueT,
            GlobalAuthorityValueT,
        ],
    ],
    StageResult[IntegratedSelectionHandoff[SelectionValueT]],
]


def run_integrated_selection_stage(
    config: CanonicalPathBConfig,
    *,
    fingerprint_authority: StageResult[FingerprintAuthorityValueT] | None,
    global_authority: StageResult[GlobalAuthorityValueT] | None,
    operation: IntegratedSelectionOperation[
        FingerprintAuthorityValueT,
        GlobalAuthorityValueT,
        SelectionValueT,
    ],
) -> StageResult[IntegratedSelectionHandoff[SelectionValueT]]:
    """Run existing C2-C5 selection only from passing authority proof."""

    inputs = _authority_inputs(fingerprint_authority, global_authority)
    if isinstance(inputs, StageFailure):
        return _failed_selection(
            inputs.reason,
            inputs.blockers,
            remediation=inputs.remediation or "complete authority export before C2-C5",
        )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_selection(
            "integrated_selection_operation_failed",
            (str(exc),),
            remediation="inspect the existing C2-C5 selection failure",
        )
    if not isinstance(result, StageResult):
        return _failed_selection(
            "integrated_selection_operation_returned_invalid_result",
            ("integrated selection callback must return StageResult",),
            remediation="return typed C2-C5 handoff and StageEvidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != INTEGRATED_SELECTION_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_selection(
            "integrated_selection_evidence_stage_mismatch",
            (
                "integrated selection callback must return evidence with "
                f"stage={INTEGRATED_SELECTION_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind C2-C5 result to the integrated selection stage",
        )
    if not isinstance(value, IntegratedSelectionHandoff):
        return _failed_selection(
            "integrated_selection_value_invalid",
            ("integrated selection callback must return IntegratedSelectionHandoff",),
            remediation="retain C2/C3/C4/C5 evidence in the selection handoff",
        )
    if value.stage_evidence != evidence:
        return _failed_selection(
            "integrated_selection_handoff_evidence_mismatch",
            ("selection handoff stage evidence must match the result evidence",),
            remediation="return one consistent integrated selection evidence object",
        )
    return result


def _authority_inputs(
    fingerprint_authority: StageResult[FingerprintAuthorityValueT] | None,
    global_authority: StageResult[GlobalAuthorityValueT] | None,
) -> (
    SelectionAuthorityInputs[FingerprintAuthorityValueT, GlobalAuthorityValueT]
    | StageFailure
):
    if (
        fingerprint_authority is None
        or fingerprint_authority.status != "pass"
        or fingerprint_authority.value is None
        or fingerprint_authority.evidence is None
        or fingerprint_authority.evidence.stage != FINGERPRINT_AUTHORITY_STAGE
    ):
        return StageFailure(
            stage=INTEGRATED_SELECTION_STAGE,
            reason="fingerprint_authority_required",
            blockers=("C2-C5 requires passing fingerprint authority evidence",),
            resumable=True,
            remediation="complete fingerprint authority export before selection",
        )
    if (
        global_authority is None
        or global_authority.status != "pass"
        or global_authority.value is None
        or global_authority.evidence is None
        or global_authority.evidence.stage != GLOBAL_AUTHORITY_STAGE
    ):
        return StageFailure(
            stage=INTEGRATED_SELECTION_STAGE,
            reason="global_authority_required",
            blockers=("C2-C5 requires passing global authority evidence",),
            resumable=True,
            remediation="complete global authority export before selection",
        )
    return SelectionAuthorityInputs(
        fingerprint_value=fingerprint_authority.value,
        fingerprint_evidence=fingerprint_authority.evidence,
        global_value=global_authority.value,
        global_evidence=global_authority.evidence,
    )


def _failed_selection(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[IntegratedSelectionHandoff[SelectionValueT]]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=INTEGRATED_SELECTION_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
