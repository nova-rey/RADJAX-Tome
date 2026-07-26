"""Callback-driven validation and reconciliation adapters after assembly."""

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
    StageEvidence,
    StageFailure,
    StageResult,
)

AssemblyValueT = TypeVar("AssemblyValueT")
ValidationValueT = TypeVar("ValidationValueT")
ReconciliationValueT = TypeVar("ReconciliationValueT")

VALIDATION_LINKAGE_STAGE = "validation_linkage"
RECONCILIATION_COVER_STAGE = "reconciliation_cover"


@dataclass(frozen=True)
class ValidationLinkageInputs(Generic[AssemblyValueT]):
    """Assembled artifact and proof consumed by validation and linkage."""

    assembly: AssemblyValueT
    assembly_evidence: StageEvidence


@dataclass(frozen=True)
class ValidationLinkageHandoff(Generic[ValidationValueT]):
    """Typed validation outcome retained for reconciliation and reporting."""

    value: ValidationValueT
    stage_evidence: StageEvidence

    def __post_init__(self) -> None:
        if self.stage_evidence.stage != VALIDATION_LINKAGE_STAGE:
            raise ValueError(
                f"validation handoff must describe stage={VALIDATION_LINKAGE_STAGE!r}"
            )


ValidationLinkageOperation = Callable[
    [CanonicalPathBConfig, ValidationLinkageInputs[AssemblyValueT]],
    StageResult[ValidationLinkageHandoff[ValidationValueT]],
]


@dataclass(frozen=True)
class ReconciliationCoverInputs(Generic[AssemblyValueT, ValidationValueT]):
    """Assembled artifact plus validation proof consumed by finalization."""

    assembly: AssemblyValueT
    assembly_evidence: StageEvidence
    validation: ValidationValueT
    validation_evidence: StageEvidence


@dataclass(frozen=True)
class ReconciliationCoverHandoff(Generic[ReconciliationValueT]):
    """Typed reconciliation, coverage, and cover outcome for reporting."""

    value: ReconciliationValueT
    stage_evidence: StageEvidence

    def __post_init__(self) -> None:
        if self.stage_evidence.stage != RECONCILIATION_COVER_STAGE:
            raise ValueError(
                "reconciliation handoff must describe "
                f"stage={RECONCILIATION_COVER_STAGE!r}"
            )


ReconciliationCoverOperation = Callable[
    [CanonicalPathBConfig, ReconciliationCoverInputs[AssemblyValueT, ValidationValueT]],
    StageResult[ReconciliationCoverHandoff[ReconciliationValueT]],
]


def run_validation_linkage_stage(
    config: CanonicalPathBConfig,
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
    *,
    operation: ValidationLinkageOperation[AssemblyValueT, ValidationValueT],
) -> StageResult[ValidationLinkageHandoff[ValidationValueT]]:
    """Run existing validation only from a passing assembled artifact."""

    inputs = _validation_inputs(assembly)
    if isinstance(inputs, StageFailure):
        return _failed_validation(
            inputs.reason,
            inputs.blockers,
            remediation=inputs.remediation or "complete artifact assembly first",
        )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_validation(
            "validation_linkage_operation_failed",
            (str(exc),),
            remediation="inspect the existing validation/linkage failure",
        )
    if not isinstance(result, StageResult):
        return _failed_validation(
            "validation_linkage_operation_returned_invalid_result",
            ("validation/linkage callback must return StageResult",),
            remediation="return typed validation handoff and evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != VALIDATION_LINKAGE_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_validation(
            "validation_linkage_evidence_stage_mismatch",
            (
                "validation/linkage callback must return evidence with "
                f"stage={VALIDATION_LINKAGE_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind validation/linkage output to its explicit stage",
        )
    if not isinstance(value, ValidationLinkageHandoff):
        return _failed_validation(
            "validation_linkage_value_invalid",
            ("validation/linkage callback must return ValidationLinkageHandoff",),
            remediation="retain validation evidence for finalization",
        )
    if value.stage_evidence != evidence:
        return _failed_validation(
            "validation_linkage_handoff_evidence_mismatch",
            ("validation handoff evidence must match result evidence",),
            remediation="return one consistent validation evidence object",
        )
    return result


def run_reconciliation_cover_stage(
    config: CanonicalPathBConfig,
    *,
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
    validation: StageResult[ValidationLinkageHandoff[ValidationValueT]] | None,
    operation: ReconciliationCoverOperation[
        AssemblyValueT,
        ValidationValueT,
        ReconciliationValueT,
    ],
) -> StageResult[ReconciliationCoverHandoff[ReconciliationValueT]]:
    """Run reconciliation and cover only after assembled artifact validation."""

    inputs = _reconciliation_inputs(assembly, validation)
    if isinstance(inputs, StageFailure):
        return _failed_reconciliation(
            inputs.reason,
            inputs.blockers,
            remediation=(
                inputs.remediation or "complete validation/linkage before finalization"
            ),
        )
    try:
        result = operation(config, inputs)
    except Exception as exc:
        return _failed_reconciliation(
            "reconciliation_cover_operation_failed",
            (str(exc),),
            remediation="inspect the existing reconciliation/cover failure",
        )
    if not isinstance(result, StageResult):
        return _failed_reconciliation(
            "reconciliation_cover_operation_returned_invalid_result",
            ("reconciliation/cover callback must return StageResult",),
            remediation="return typed reconciliation handoff and evidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    value = result.value
    if evidence is None or evidence.stage != RECONCILIATION_COVER_STAGE:
        actual_stage = None if evidence is None else evidence.stage
        return _failed_reconciliation(
            "reconciliation_cover_evidence_stage_mismatch",
            (
                "reconciliation/cover callback must return evidence with "
                f"stage={RECONCILIATION_COVER_STAGE!r}; got {actual_stage!r}",
            ),
            remediation="bind reconciliation/cover output to its explicit stage",
        )
    if not isinstance(value, ReconciliationCoverHandoff):
        return _failed_reconciliation(
            "reconciliation_cover_value_invalid",
            ("reconciliation/cover callback must return ReconciliationCoverHandoff",),
            remediation="retain reconciliation proof for final reporting",
        )
    if value.stage_evidence != evidence:
        return _failed_reconciliation(
            "reconciliation_cover_handoff_evidence_mismatch",
            ("reconciliation handoff evidence must match result evidence",),
            remediation="return one consistent reconciliation evidence object",
        )
    return result


def _validation_inputs(
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
) -> ValidationLinkageInputs[AssemblyValueT] | StageFailure:
    if (
        assembly is None
        or assembly.status != "pass"
        or assembly.value is None
        or assembly.evidence is None
        or assembly.evidence.stage != ARTIFACT_ASSEMBLY_STAGE
        or assembly.value.stage_evidence != assembly.evidence
    ):
        return _verification_failure("artifact_assembly_required")
    return ValidationLinkageInputs(
        assembly=assembly.value.value,
        assembly_evidence=assembly.evidence,
    )


def _reconciliation_inputs(
    assembly: StageResult[ArtifactAssemblyHandoff[AssemblyValueT]] | None,
    validation: StageResult[ValidationLinkageHandoff[ValidationValueT]] | None,
) -> ReconciliationCoverInputs[AssemblyValueT, ValidationValueT] | StageFailure:
    validation_inputs = _validation_inputs(assembly)
    if isinstance(validation_inputs, StageFailure):
        return validation_inputs
    if (
        validation is None
        or validation.status != "pass"
        or validation.value is None
        or validation.evidence is None
        or validation.evidence.stage != VALIDATION_LINKAGE_STAGE
        or validation.value.stage_evidence != validation.evidence
    ):
        return _verification_failure("validation_linkage_required")
    return ReconciliationCoverInputs(
        assembly=validation_inputs.assembly,
        assembly_evidence=validation_inputs.assembly_evidence,
        validation=validation.value.value,
        validation_evidence=validation.evidence,
    )


def _verification_failure(reason: str) -> StageFailure:
    return StageFailure(
        stage=VALIDATION_LINKAGE_STAGE,
        reason=reason,
        blockers=("verification requires the preceding assembled artifact proof",),
        resumable=True,
        remediation="complete the missing preceding verification stage",
    )


def _failed_validation(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[ValidationLinkageHandoff[ValidationValueT]]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=VALIDATION_LINKAGE_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )


def _failed_reconciliation(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[ReconciliationCoverHandoff[ReconciliationValueT]]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=RECONCILIATION_COVER_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
