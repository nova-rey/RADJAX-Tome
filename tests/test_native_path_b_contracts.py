from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from radjax_tome.builder.native_path_b.contracts import (
    EvidenceCount,
    EvidenceDiagnostic,
    FileHash,
    NativePathBRunResult,
    PriorStageProof,
    ScoreSurfaceCorridorEvidence,
    SelectedArtifactCorridorEvidence,
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.evidence import (
    read_score_surface_corridor_evidence,
    read_selected_artifact_corridor_evidence,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_corridor_evidence(
    output_dir: Path,
    *,
    selected_count: int,
    selected_linked: bool,
) -> None:
    corridors = output_dir / "corridors"
    _write_json(
        corridors / "corridor_summary.json",
        {
            "corridor_artifact_built": True,
            "corridor_modes_built": True,
            "corridor_observation_basis": "full_token_position_corridor",
            "degraded_corridor_export": False,
            "corridor_positions_available": 8,
            "corridor_positions_used": 8,
            "fingerprint_count": 3,
            "mode_count": 2,
            "corridor_assignment_count": 8,
            "selected_exemplar_count": selected_count,
            "selected_exemplars_linked_to_corridor_modes": selected_linked,
        },
    )
    _write_json(corridors / "corridor_fingerprints.json", {"fingerprint_count": 3})
    _write_json(corridors / "corridor_modes.json", {"mode_count": 2})
    _write_json(corridors / "mode_assignments.json", {"num_assignments": 8})


def _write_complete_score_manifest(output_dir: Path) -> None:
    _write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "num_examples_completed": 2,
            "num_examples_planned": 2,
            "num_shards_completed": 1,
            "num_shards_planned": 1,
        },
    )


def _write_selected_delivery_proof(
    output_dir: Path,
    *,
    selected_count: int = 2,
    authority_hash: str = "sha256:authority",
    delivery_hash: str | None = None,
) -> None:
    _write_json(
        output_dir / "delivery_report.json",
        {
            "status": "pass",
            "num_selected_exemplars": selected_count,
            "delivery_authority_hash": delivery_hash or authority_hash,
        },
    )
    _write_json(
        output_dir / "c6" / "authority_manifest.json",
        {"score_pass_authority_hash": authority_hash},
    )


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_stage_contracts_are_immutable_and_enforce_success_failure_invariants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    file_hash = FileHash(path=path, sha256="sha256:test")
    count = EvidenceCount("records", 1)
    proof = PriorStageProof("score_pass", (path,), (file_hash,), (count,))
    evidence = StageEvidence("stage", (path,), (file_hash,), (count,), proof)
    failure = StageFailure(
        stage="stage",
        reason="missing_proof",
        blockers=("missing",),
        diagnostics=(EvidenceDiagnostic("path", str(path)),),
        resumable=True,
        remediation="restore evidence",
    )

    passed = StageResult(status="pass", value="value", evidence=evidence)
    failed = StageResult[str](status="fail", value=None, evidence=None, failure=failure)

    assert passed.evidence is evidence
    assert failed.failure is failure
    with pytest.raises(FrozenInstanceError):
        evidence.stage = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        failure.reason = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceCount("invalid", -1)
    with pytest.raises(ValueError, match="passing StageResult"):
        StageResult(status="pass", value=None, evidence=evidence)
    with pytest.raises(ValueError, match="failing StageResult"):
        StageResult(status="fail", value="unexpected", evidence=None, failure=failure)
    with pytest.raises(ValueError, match="passing NativePathBRunResult"):
        NativePathBRunResult(
            status="pass",
            production_report_path=path,
            validation_report_path=path,
            evidence=evidence,
            failure=failure,
        )


def test_early_and_late_corridor_readers_construct_distinct_immutable_evidence(
    tmp_path: Path,
) -> None:
    early_output = tmp_path / "early"
    _write_corridor_evidence(
        early_output,
        selected_count=0,
        selected_linked=False,
    )
    _write_complete_score_manifest(early_output)
    early_before = _tree_bytes(early_output)
    early = read_score_surface_corridor_evidence(early_output)

    assert early.status == "pass"
    assert isinstance(early.value, ScoreSurfaceCorridorEvidence)
    assert early.value.selected_exemplar_count == 0
    assert early.value.selected_exemplars_linked is False
    assert early.evidence.stage == "score_surface_corridor_materialization"
    assert early.evidence.prior_stage_proof.stage == "score_pass"
    assert _tree_bytes(early_output) == early_before

    late_output = tmp_path / "late"
    _write_corridor_evidence(late_output, selected_count=2, selected_linked=True)
    _write_selected_delivery_proof(late_output)
    late_before = _tree_bytes(late_output)
    late = read_selected_artifact_corridor_evidence(late_output)

    assert late.status == "pass"
    assert isinstance(late.value, SelectedArtifactCorridorEvidence)
    assert late.value.selected_exemplar_count == 2
    assert late.value.selected_exemplars_linked is True
    assert late.evidence.stage == "selected_artifact_corridor_finalization"
    assert late.evidence.prior_stage_proof.stage == "selected_delivery_rerun"
    assert _tree_bytes(late_output) == late_before


def test_corridor_readers_reject_the_other_phase_without_persisted_phase_state(
    tmp_path: Path,
) -> None:
    early_output = tmp_path / "early"
    _write_corridor_evidence(
        early_output,
        selected_count=0,
        selected_linked=False,
    )
    _write_complete_score_manifest(early_output)
    late_from_early = read_selected_artifact_corridor_evidence(early_output)

    assert late_from_early.status == "fail"
    assert (
        late_from_early.failure.reason == "provisional_corridor_evidence_is_not_final"
    )

    late_output = tmp_path / "late"
    _write_corridor_evidence(late_output, selected_count=2, selected_linked=True)
    _write_complete_score_manifest(late_output)
    _write_selected_delivery_proof(late_output)
    early_from_late = read_score_surface_corridor_evidence(late_output)

    assert early_from_late.status == "fail"
    assert early_from_late.failure.reason == "provisional_corridor_evidence_overwritten"


def test_missing_corrupt_and_mismatched_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    _write_corridor_evidence(missing, selected_count=0, selected_linked=False)
    _write_complete_score_manifest(missing)
    (missing / "corridors" / "corridor_modes.json").unlink()
    missing_result = read_score_surface_corridor_evidence(missing)

    assert missing_result.status == "fail"
    assert missing_result.failure.reason == "corridor_evidence_unavailable"
    assert missing_result.failure.resumable is True

    corrupt = tmp_path / "corrupt"
    _write_corridor_evidence(corrupt, selected_count=2, selected_linked=True)
    _write_selected_delivery_proof(corrupt)
    (corrupt / "delivery_report.json").write_text("{not-json", encoding="utf-8")
    corrupt_result = read_selected_artifact_corridor_evidence(corrupt)

    assert corrupt_result.status == "fail"
    assert corrupt_result.failure.reason == "selected_delivery_proof_unavailable"

    mismatch = tmp_path / "mismatch"
    _write_corridor_evidence(mismatch, selected_count=2, selected_linked=True)
    _write_selected_delivery_proof(
        mismatch,
        authority_hash="sha256:authority-a",
        delivery_hash="sha256:authority-b",
    )
    mismatch_result = read_selected_artifact_corridor_evidence(mismatch)

    assert mismatch_result.status == "fail"
    assert mismatch_result.failure.reason == "selected_delivery_authority_mismatch"
