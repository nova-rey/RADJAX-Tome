"""Immutable M4A contracts for native Path-B evidence and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

StageStatus = Literal["pass", "fail"]
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class FileHash:
    """A content hash observed at an immutable artifact path."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class EvidenceCount:
    """A named, non-negative count derived from existing evidence."""

    name: str
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"evidence count must be non-negative: {self.name}")


@dataclass(frozen=True)
class PriorStageProof:
    """Read-only proof retained from a completed upstream stage."""

    stage: str
    paths: tuple[Path, ...]
    hashes: tuple[FileHash, ...]
    counts: tuple[EvidenceCount, ...] = ()


@dataclass(frozen=True)
class StageEvidence:
    """Existing paths, hashes, counts, and upstream proof for one stage."""

    stage: str
    paths: tuple[Path, ...]
    hashes: tuple[FileHash, ...]
    counts: tuple[EvidenceCount, ...]
    prior_stage_proof: PriorStageProof | None = None


@dataclass(frozen=True)
class EvidenceDiagnostic:
    """A stable key/value detail recorded with a typed stage failure."""

    name: str
    value: str


@dataclass(frozen=True)
class StageFailure:
    """A fail-closed stage result with actionable remediation."""

    stage: str
    reason: str
    blockers: tuple[str, ...]
    diagnostics: tuple[EvidenceDiagnostic, ...] = ()
    resumable: bool = False
    remediation: str | None = None


@dataclass(frozen=True)
class StageResult(Generic[ResultT]):
    """Typed stage outcome without a new persistent representation."""

    status: StageStatus
    value: ResultT | None
    evidence: StageEvidence | None
    warnings: tuple[str, ...] = ()
    failure: StageFailure | None = None

    def __post_init__(self) -> None:
        if self.status == "pass":
            if self.value is None or self.evidence is None or self.failure is not None:
                raise ValueError(
                    "passing StageResult requires value/evidence and no failure"
                )
        elif (
            self.value is not None or self.evidence is not None or self.failure is None
        ):
            raise ValueError(
                "failing StageResult requires failure and no value/evidence"
            )


@dataclass(frozen=True)
class ScoreSurfaceCorridorEvidence:
    """Provisional corridor materialization from the completed score surface."""

    stage_evidence: StageEvidence
    summary_path: Path
    fingerprints_path: Path
    modes_path: Path
    assignments_path: Path
    positions_available: int
    positions_used: int
    fingerprint_count: int
    mode_count: int
    assignment_count: int
    selected_exemplar_count: int
    selected_exemplars_linked: bool

    def __post_init__(self) -> None:
        if self.selected_exemplar_count != 0 or self.selected_exemplars_linked:
            raise ValueError("score-surface corridor evidence must remain provisional")


@dataclass(frozen=True)
class SelectedArtifactCorridorEvidence:
    """Final public corridor surface linked to selected C5 artifacts."""

    stage_evidence: StageEvidence
    summary_path: Path
    fingerprints_path: Path
    modes_path: Path
    assignments_path: Path
    positions_available: int
    positions_used: int
    fingerprint_count: int
    mode_count: int
    assignment_count: int
    selected_exemplar_count: int
    selected_exemplars_linked: bool
    delivery_report_path: Path
    authority_manifest_path: Path
    delivery_authority_hash: str

    def __post_init__(self) -> None:
        if self.selected_exemplar_count < 1 or not self.selected_exemplars_linked:
            raise ValueError(
                "selected-artifact corridor evidence requires selected-linked proof"
            )


@dataclass(frozen=True)
class NativePathBRunResult:
    """Terminal references for a native Path-B execution."""

    status: StageStatus
    production_report_path: Path | None
    validation_report_path: Path | None
    evidence: StageEvidence | None
    failure: StageFailure | None = None

    def __post_init__(self) -> None:
        if self.status == "pass" and self.failure is not None:
            raise ValueError("passing NativePathBRunResult cannot contain a failure")
        if self.status == "fail" and self.failure is None:
            raise ValueError("failing NativePathBRunResult requires a failure")
