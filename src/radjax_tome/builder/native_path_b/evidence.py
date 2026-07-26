"""Read-only M4A evidence readers for the two distinct corridor phases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.native_path_b.contracts import (
    EvidenceCount,
    EvidenceDiagnostic,
    FileHash,
    PriorStageProof,
    ScoreSurfaceCorridorEvidence,
    SelectedArtifactCorridorEvidence,
    StageEvidence,
    StageFailure,
    StageResult,
)

_FULL_TOKEN_POSITION_CORRIDOR = "full_token_position_corridor"
_EARLY_STAGE = "score_surface_corridor_materialization"
_LATE_STAGE = "selected_artifact_corridor_finalization"


def read_score_surface_corridor_evidence(
    output_dir: Path,
    *,
    run_manifest_path: Path | None = None,
) -> StageResult[ScoreSurfaceCorridorEvidence]:
    """Read provisional early-corridor proof, rejecting a final overwrite."""

    manifest_path = run_manifest_path or output_dir / "run_manifest.json"
    documents = _read_corridor_documents(output_dir, _EARLY_STAGE)
    if isinstance(documents, StageFailure):
        return _failed(documents)
    summary, fingerprints, modes, assignments, paths = documents
    manifest = _read_json_object(manifest_path)
    if isinstance(manifest, StageFailure):
        return _failed(
            _failure(
                _EARLY_STAGE,
                "score_pass_proof_unavailable",
                manifest.blockers,
                remediation="restore a complete score-pass run_manifest.json",
            )
        )
    if manifest.get("status") != "complete":
        return _failed(
            _failure(
                _EARLY_STAGE,
                "score_pass_incomplete",
                (
                    "run_manifest.status must be complete before corridor "
                    "materialization",
                ),
                diagnostics=(
                    EvidenceDiagnostic(
                        "run_manifest_status", str(manifest.get("status"))
                    ),
                ),
                remediation=(
                    "complete or resume the score pass before reading early evidence"
                ),
            )
        )
    common = _validate_common_corridor_fields(
        _EARLY_STAGE, summary, fingerprints, modes, assignments
    )
    if isinstance(common, StageFailure):
        return _failed(common)
    if common.selected_exemplar_count != 0 or common.selected_exemplars_linked:
        return _failed(
            _failure(
                _EARLY_STAGE,
                "provisional_corridor_evidence_overwritten",
                (
                    "score-surface corridor evidence requires zero selected exemplars "
                    "and no selected-link claim",
                ),
                remediation=(
                    "read this artifact as selected-artifact finalization "
                    "evidence instead"
                ),
            )
        )
    manifest_hash = _hash_file(manifest_path)
    proof = PriorStageProof(
        stage="score_pass",
        paths=(manifest_path,),
        hashes=(manifest_hash,),
        counts=_manifest_counts(manifest),
    )
    evidence = _stage_evidence(_EARLY_STAGE, paths, common, proof)
    return StageResult(
        status="pass",
        value=ScoreSurfaceCorridorEvidence(
            stage_evidence=evidence,
            summary_path=paths[0],
            fingerprints_path=paths[1],
            modes_path=paths[2],
            assignments_path=paths[3],
            **common.to_constructor_fields(),
        ),
        evidence=evidence,
    )


def read_selected_artifact_corridor_evidence(
    output_dir: Path,
) -> StageResult[SelectedArtifactCorridorEvidence]:
    """Read final selected-linked proof and refuse provisional corridor evidence."""

    documents = _read_corridor_documents(output_dir, _LATE_STAGE)
    if isinstance(documents, StageFailure):
        return _failed(documents)
    summary, fingerprints, modes, assignments, paths = documents
    common = _validate_common_corridor_fields(
        _LATE_STAGE, summary, fingerprints, modes, assignments
    )
    if isinstance(common, StageFailure):
        return _failed(common)
    if common.selected_exemplar_count < 1 or not common.selected_exemplars_linked:
        return _failed(
            _failure(
                _LATE_STAGE,
                "provisional_corridor_evidence_is_not_final",
                (
                    "selected-artifact corridor finalization requires at least one "
                    "selected exemplar linked to corridor modes",
                ),
                remediation="complete selected delivery and late corridor finalization",
            )
        )
    delivery_path = output_dir / "delivery_report.json"
    authority_path = output_dir / "c6" / "authority_manifest.json"
    delivery = _read_json_object(delivery_path)
    if isinstance(delivery, StageFailure):
        return _failed(
            _failure(
                _LATE_STAGE,
                "selected_delivery_proof_unavailable",
                delivery.blockers,
                remediation="restore the selected delivery report before finalization",
            )
        )
    authority = _read_json_object(authority_path)
    if isinstance(authority, StageFailure):
        return _failed(
            _failure(
                _LATE_STAGE,
                "selection_authority_proof_unavailable",
                authority.blockers,
                remediation="restore the C6 authority manifest before finalization",
            )
        )
    authority_hash = authority.get("score_pass_authority_hash")
    delivery_hash = delivery.get("delivery_authority_hash")
    if (
        delivery.get("status") != "pass"
        or not isinstance(authority_hash, str)
        or not authority_hash
        or delivery_hash != authority_hash
    ):
        return _failed(
            _failure(
                _LATE_STAGE,
                "selected_delivery_authority_mismatch",
                (
                    "delivery_report must pass and bind the current score-pass "
                    "authority hash",
                ),
                diagnostics=(
                    EvidenceDiagnostic("delivery_status", str(delivery.get("status"))),
                    EvidenceDiagnostic("delivery_authority_hash", str(delivery_hash)),
                    EvidenceDiagnostic(
                        "score_pass_authority_hash", str(authority_hash)
                    ),
                ),
                remediation="rerun selected delivery from the matching C5 authority",
            )
        )
    delivery_count = _integer(delivery.get("num_selected_exemplars"))
    if delivery_count != common.selected_exemplar_count:
        return _failed(
            _failure(
                _LATE_STAGE,
                "selected_delivery_count_mismatch",
                ("delivery_report selected count must equal corridor selected count",),
                diagnostics=(
                    EvidenceDiagnostic("delivery_selected_count", str(delivery_count)),
                    EvidenceDiagnostic(
                        "corridor_selected_count", str(common.selected_exemplar_count)
                    ),
                ),
                remediation=(
                    "rebuild final corridors from the selected delivery records"
                ),
            )
        )
    proof = PriorStageProof(
        stage="selected_delivery_rerun",
        paths=(delivery_path, authority_path),
        hashes=(_hash_file(delivery_path), _hash_file(authority_path)),
        counts=(EvidenceCount("selected_exemplar_count", delivery_count),),
    )
    evidence = _stage_evidence(_LATE_STAGE, paths, common, proof)
    return StageResult(
        status="pass",
        value=SelectedArtifactCorridorEvidence(
            stage_evidence=evidence,
            summary_path=paths[0],
            fingerprints_path=paths[1],
            modes_path=paths[2],
            assignments_path=paths[3],
            delivery_report_path=delivery_path,
            authority_manifest_path=authority_path,
            delivery_authority_hash=authority_hash,
            **common.to_constructor_fields(),
        ),
        evidence=evidence,
    )


def _read_corridor_documents(
    output_dir: Path,
    stage: str,
) -> (
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        tuple[Path, ...],
    ]
    | StageFailure
):
    paths = (
        output_dir / "corridors" / "corridor_summary.json",
        output_dir / "corridors" / "corridor_fingerprints.json",
        output_dir / "corridors" / "corridor_modes.json",
        output_dir / "corridors" / "mode_assignments.json",
    )
    documents: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in paths:
        document = _read_json_object(path)
        if isinstance(document, StageFailure):
            blockers.extend(document.blockers)
        else:
            documents.append(document)
    if blockers:
        return _failure(
            stage,
            "corridor_evidence_unavailable",
            tuple(blockers),
            remediation="restore the existing corridor JSON evidence",
        )
    return documents[0], documents[1], documents[2], documents[3], paths


def _read_json_object(path: Path) -> dict[str, Any] | StageFailure:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _failure(
            "evidence_reader",
            "json_unavailable_or_invalid",
            (f"cannot read JSON evidence {path}: {exc}",),
        )
    if not isinstance(payload, dict):
        return _failure(
            "evidence_reader",
            "json_object_required",
            (f"JSON evidence must be an object: {path}",),
        )
    return payload


@dataclass(frozen=True)
class _CorridorFields:
    positions_available: int
    positions_used: int
    fingerprint_count: int
    mode_count: int
    assignment_count: int
    selected_exemplar_count: int
    selected_exemplars_linked: bool

    def to_constructor_fields(self) -> dict[str, int | bool]:
        return {
            "positions_available": self.positions_available,
            "positions_used": self.positions_used,
            "fingerprint_count": self.fingerprint_count,
            "mode_count": self.mode_count,
            "assignment_count": self.assignment_count,
            "selected_exemplar_count": self.selected_exemplar_count,
            "selected_exemplars_linked": self.selected_exemplars_linked,
        }


def _validate_common_corridor_fields(
    stage: str,
    summary: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    modes: Mapping[str, Any],
    assignments: Mapping[str, Any],
) -> _CorridorFields | StageFailure:
    required_truths = (
        ("corridor_artifact_built", summary.get("corridor_artifact_built")),
        ("corridor_modes_built", summary.get("corridor_modes_built")),
    )
    blockers = [
        f"corridor_summary.{name} must be true"
        for name, value in required_truths
        if value is not True
    ]
    if summary.get("corridor_observation_basis") != _FULL_TOKEN_POSITION_CORRIDOR:
        blockers.append("corridor_summary must retain full token-position evidence")
    if summary.get("degraded_corridor_export") is not False:
        blockers.append("corridor_summary must not be degraded")
    fields = {
        "positions_available": _integer(summary.get("corridor_positions_available")),
        "positions_used": _integer(summary.get("corridor_positions_used")),
        "fingerprint_count": _integer(summary.get("fingerprint_count")),
        "mode_count": _integer(summary.get("mode_count")),
        "assignment_count": _integer(summary.get("corridor_assignment_count")),
        "selected_exemplar_count": _integer(summary.get("selected_exemplar_count")),
    }
    missing_counts = [name for name, value in fields.items() if value is None]
    blockers.extend(
        f"corridor_summary.{name} must be a non-negative integer"
        for name in missing_counts
    )
    selected_linked = summary.get("selected_exemplars_linked_to_corridor_modes")
    if not isinstance(selected_linked, bool):
        blockers.append(
            "corridor_summary.selected_exemplars_linked_to_corridor_modes must be "
            "boolean"
        )
    if blockers:
        return _failure(
            stage,
            "corridor_evidence_invalid",
            tuple(blockers),
            remediation="rebuild corridor artifacts from the verified score surface",
        )
    assert all(value is not None for value in fields.values())
    if (
        fingerprints.get("fingerprint_count") != fields["fingerprint_count"]
        or modes.get("mode_count") != fields["mode_count"]
        or assignments.get("num_assignments") != fields["assignment_count"]
        or fields["assignment_count"] != fields["positions_used"]
        or fields["positions_available"] < fields["positions_used"]
    ):
        return _failure(
            stage,
            "corridor_evidence_count_mismatch",
            ("corridor JSON count fields are not mutually consistent",),
            remediation=(
                "rebuild corridor artifacts without editing their JSON evidence"
            ),
        )
    return _CorridorFields(
        positions_available=int(fields["positions_available"]),
        positions_used=int(fields["positions_used"]),
        fingerprint_count=int(fields["fingerprint_count"]),
        mode_count=int(fields["mode_count"]),
        assignment_count=int(fields["assignment_count"]),
        selected_exemplar_count=int(fields["selected_exemplar_count"]),
        selected_exemplars_linked=selected_linked,
    )


def _stage_evidence(
    stage: str,
    paths: tuple[Path, ...],
    fields: _CorridorFields,
    prior_stage_proof: PriorStageProof,
) -> StageEvidence:
    return StageEvidence(
        stage=stage,
        paths=paths,
        hashes=tuple(_hash_file(path) for path in paths),
        counts=(
            EvidenceCount("positions_available", fields.positions_available),
            EvidenceCount("positions_used", fields.positions_used),
            EvidenceCount("fingerprint_count", fields.fingerprint_count),
            EvidenceCount("mode_count", fields.mode_count),
            EvidenceCount("assignment_count", fields.assignment_count),
            EvidenceCount("selected_exemplar_count", fields.selected_exemplar_count),
        ),
        prior_stage_proof=prior_stage_proof,
    )


def _manifest_counts(manifest: Mapping[str, Any]) -> tuple[EvidenceCount, ...]:
    fields = (
        ("num_examples_completed", manifest.get("num_examples_completed")),
        ("num_examples_planned", manifest.get("num_examples_planned")),
        ("num_shards_completed", manifest.get("num_shards_completed")),
        ("num_shards_planned", manifest.get("num_shards_planned")),
    )
    return tuple(
        EvidenceCount(name, value)
        for name, raw_value in fields
        if (value := _integer(raw_value)) is not None
    )


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _hash_file(path: Path) -> FileHash:
    return FileHash(
        path=path,
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _failure(
    stage: str,
    reason: str,
    blockers: tuple[str, ...],
    *,
    diagnostics: tuple[EvidenceDiagnostic, ...] = (),
    remediation: str | None = None,
) -> StageFailure:
    return StageFailure(
        stage=stage,
        reason=reason,
        blockers=blockers,
        diagnostics=diagnostics,
        resumable=True,
        remediation=remediation,
    )


def _failed(failure: StageFailure) -> StageResult[Any]:
    return StageResult(status="fail", value=None, evidence=None, failure=failure)
