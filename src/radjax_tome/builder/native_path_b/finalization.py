"""Callback-driven terminal reporting adapter after reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.assembly import (
    ARTIFACT_ASSEMBLY_STAGE,
    ArtifactAssemblyHandoff,
)
from radjax_tome.builder.native_path_b.contracts import (
    NativePathBRunResult,
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.verification import (
    RECONCILIATION_COVER_STAGE,
    VALIDATION_LINKAGE_STAGE,
    ReconciliationCoverHandoff,
    ValidationLinkageHandoff,
)

AssemblyValueT = TypeVar("AssemblyValueT")
ValidationValueT = TypeVar("ValidationValueT")
ReconciliationValueT = TypeVar("ReconciliationValueT")

FINAL_REPORTING_STAGE = "final_reporting"


@dataclass(frozen=True)
class FinalReportingInputs(
    Generic[AssemblyValueT, ValidationValueT, ReconciliationValueT]
):
    """Completed assembly and verification proof consumed by final reporting."""

    assembly: AssemblyValueT
    assembly_evidence: StageEvidence
    validation: ValidationValueT
    validation_evidence: StageEvidence
    reconciliation: ReconciliationValueT
    reconciliation_evidence: StageEvidence


FinalReportingOperation = Callable[
    [
        CanonicalPathBConfig,
        FinalReportingInputs[
            AssemblyValueT,
            ValidationValueT,
            ReconciliationValueT,
        ],
    ],
    NativePathBRunResult,
]


def run_final_reporting_stage(
    config: CanonicalPathBConfig,
    *,
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
    validation: StageResult[ValidationLinkageHandoff[ValidationValueT]] | None,
    reconciliation: StageResult[ReconciliationCoverHandoff[ReconciliationValueT]]
    | None,
    operation: FinalReportingOperation[
        AssemblyValueT,
        ValidationValueT,
        ReconciliationValueT,
    ],
) -> NativePathBRunResult:
    """Render terminal result from completed proof without changing callbacks."""

    inputs = _final_reporting_inputs(assembly, validation, reconciliation)
    if isinstance(inputs, StageFailure):
        return _failed_terminal(
            inputs.reason,
            inputs.blockers,
            remediation=inputs.remediation or "complete preceding finalization stages",
        )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_terminal(
            "final_reporting_operation_failed",
            (str(exc),),
            remediation="inspect the existing final reporting failure",
        )
    if not isinstance(result, NativePathBRunResult):
        return _failed_terminal(
            "final_reporting_operation_returned_invalid_result",
            ("final reporting callback must return NativePathBRunResult",),
            remediation="return the existing terminal reporting result",
        )
    if result.status == "fail":
        return result
    if result.evidence is None or result.evidence.stage != FINAL_REPORTING_STAGE:
        actual_stage = None if result.evidence is None else result.evidence.stage
        return _failed_terminal(
            "final_reporting_evidence_stage_mismatch",
            (
                "final reporting callback must return evidence with "
                f"stage={FINAL_REPORTING_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind terminal result to final reporting evidence",
        )
    return result


def _final_reporting_inputs(
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
    validation: StageResult[ValidationLinkageHandoff[ValidationValueT]] | None,
    reconciliation: StageResult[ReconciliationCoverHandoff[ReconciliationValueT]]
    | None,
) -> (
    FinalReportingInputs[AssemblyValueT, ValidationValueT, ReconciliationValueT]
    | StageFailure
):
    if (
        assembly is None
        or assembly.status != "pass"
        or assembly.value is None
        or assembly.evidence is None
        or assembly.evidence.stage != ARTIFACT_ASSEMBLY_STAGE
        or assembly.value.stage_evidence != assembly.evidence
    ):
        return _finalization_failure("artifact_assembly_required")
    if (
        validation is None
        or validation.status != "pass"
        or validation.value is None
        or validation.evidence is None
        or validation.evidence.stage != VALIDATION_LINKAGE_STAGE
        or validation.value.stage_evidence != validation.evidence
    ):
        return _finalization_failure("validation_linkage_required")
    if (
        reconciliation is None
        or reconciliation.status != "pass"
        or reconciliation.value is None
        or reconciliation.evidence is None
        or reconciliation.evidence.stage != RECONCILIATION_COVER_STAGE
        or reconciliation.value.stage_evidence != reconciliation.evidence
    ):
        return _finalization_failure("reconciliation_cover_required")
    return FinalReportingInputs(
        assembly=assembly.value.value,
        assembly_evidence=assembly.evidence,
        validation=validation.value.value,
        validation_evidence=validation.evidence,
        reconciliation=reconciliation.value.value,
        reconciliation_evidence=reconciliation.evidence,
    )


def _finalization_failure(reason: str) -> StageFailure:
    return StageFailure(
        stage=FINAL_REPORTING_STAGE,
        reason=reason,
        blockers=("final reporting requires all preceding finalization proof",),
        resumable=True,
        remediation="complete the missing preceding finalization stage",
    )


def _failed_terminal(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> NativePathBRunResult:
    return NativePathBRunResult(
        status="fail",
        production_report_path=None,
        validation_report_path=None,
        evidence=None,
        failure=StageFailure(
            stage=FINAL_REPORTING_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
