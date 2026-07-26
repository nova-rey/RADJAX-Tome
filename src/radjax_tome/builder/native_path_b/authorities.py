"""Callback-driven native Path-B fingerprint and global authority adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import (
    ScoreSurfaceCorridorEvidence,
    StageFailure,
    StageResult,
)

FingerprintAuthorityValueT = TypeVar("FingerprintAuthorityValueT")
GlobalAuthorityValueT = TypeVar("GlobalAuthorityValueT")
FingerprintAuthorityOperation = Callable[
    [CanonicalPathBConfig, ScoreSurfaceCorridorEvidence],
    StageResult[FingerprintAuthorityValueT],
]
GlobalAuthorityOperation = Callable[
    [CanonicalPathBConfig, ScoreSurfaceCorridorEvidence, FingerprintAuthorityValueT],
    StageResult[GlobalAuthorityValueT],
]

FINGERPRINT_AUTHORITY_STAGE = "fingerprint_corridor_selection_authority_export"
GLOBAL_AUTHORITY_STAGE = "global_authority_export"


def run_fingerprint_selection_authority_stage(
    config: CanonicalPathBConfig,
    early_corridor: StageResult[ScoreSurfaceCorridorEvidence] | None,
    *,
    operation: FingerprintAuthorityOperation[FingerprintAuthorityValueT],
) -> StageResult[FingerprintAuthorityValueT]:
    """Export fingerprint authority only from passing provisional corridor proof."""

    if not _passing_early_corridor(early_corridor):
        return _failed_fingerprint_authority(
            "provisional_corridor_required",
            ("fingerprint authority requires passing provisional corridor evidence",),
            remediation="complete early score-surface corridor materialization first",
        )
    assert early_corridor is not None and early_corridor.value is not None
    try:
        result = operation(config, early_corridor.value)
    except Exception as exc:
        return _failed_fingerprint_authority(
            "fingerprint_authority_operation_failed",
            (str(exc),),
            remediation="inspect the existing fingerprint authority export failure",
        )
    return _validated_authority_result(
        result,
        stage=FINGERPRINT_AUTHORITY_STAGE,
        failure_factory=_failed_fingerprint_authority,
    )


def run_global_authority_stage(
    config: CanonicalPathBConfig,
    early_corridor: StageResult[ScoreSurfaceCorridorEvidence] | None,
    fingerprint_authority: StageResult[FingerprintAuthorityValueT] | None,
    *,
    operation: GlobalAuthorityOperation[
        FingerprintAuthorityValueT,
        GlobalAuthorityValueT,
    ],
) -> StageResult[GlobalAuthorityValueT]:
    """Export global supply after passing early and fingerprint authority stages."""

    if not _passing_early_corridor(early_corridor):
        return _failed_global_authority(
            "provisional_corridor_required",
            ("global authority requires passing provisional corridor evidence",),
            remediation="complete early score-surface corridor materialization first",
        )
    if (
        fingerprint_authority is None
        or fingerprint_authority.status != "pass"
        or fingerprint_authority.value is None
    ):
        return _failed_global_authority(
            "fingerprint_authority_required",
            ("global authority requires passing fingerprint authority evidence",),
            remediation="complete fingerprint authority export before global supply",
        )
    assert early_corridor is not None and early_corridor.value is not None
    try:
        result = operation(config, early_corridor.value, fingerprint_authority.value)
    except Exception as exc:
        return _failed_global_authority(
            "global_authority_operation_failed",
            (str(exc),),
            remediation="inspect the existing global authority export failure",
        )
    return _validated_authority_result(
        result,
        stage=GLOBAL_AUTHORITY_STAGE,
        failure_factory=_failed_global_authority,
    )


def _passing_early_corridor(
    result: StageResult[ScoreSurfaceCorridorEvidence] | None,
) -> bool:
    if result is None or result.status != "pass" or result.value is None:
        return False
    return (
        result.evidence is not None
        and result.evidence.stage == "score_surface_corridor_materialization"
        and result.value.stage_evidence == result.evidence
        and result.value.selected_exemplar_count == 0
        and not result.value.selected_exemplars_linked
    )


def _validated_authority_result(
    result: object,
    *,
    stage: str,
    failure_factory: Callable[..., StageResult[FingerprintAuthorityValueT]],
) -> StageResult[FingerprintAuthorityValueT]:
    if not isinstance(result, StageResult):
        return failure_factory(
            f"{stage}_operation_returned_invalid_result",
            ("authority callback must return StageResult",),
            remediation="return typed authority value and StageEvidence",
        )
    if result.status == "fail":
        return result
    evidence = result.evidence
    if evidence is None or evidence.stage != stage:
        actual_stage = None if evidence is None else evidence.stage
        return failure_factory(
            f"{stage}_evidence_stage_mismatch",
            (
                "authority callback must return evidence with "
                f"stage={stage!r}; got {actual_stage!r}",
            ),
            remediation="bind authority evidence to its explicit native stage",
        )
    return result


def _failed_fingerprint_authority(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[FingerprintAuthorityValueT]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=FINGERPRINT_AUTHORITY_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )


def _failed_global_authority(
    reason: str,
    blockers: tuple[str, ...],
    *,
    remediation: str,
) -> StageResult[GlobalAuthorityValueT]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(
            stage=GLOBAL_AUTHORITY_STAGE,
            reason=reason,
            blockers=blockers,
            resumable=True,
            remediation=remediation,
        ),
    )
