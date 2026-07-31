"""Provisional and selected-linked corridor plus payload assembly stages."""

from __future__ import annotations

from typing import Any

from radjax_tome.builder.corridor_artifacts import build_corridor_artifacts
from radjax_tome.builder.exemplar_delivery import (
    assemble_selected_delivery_artifacts,
    finalize_selected_delivery_corridor,
)
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.builder.teacher_textbook import load_text_examples
from radjax_tome.targets.store import TeacherTargetStore


def native_early_corridor_operation(state: Any) -> Any:
    from radjax_tome.builder.native_path_b.evidence import (
        read_score_surface_corridor_evidence,
    )

    config = state.config
    state.progress.stage("fingerprint_corridor_export")
    store = TeacherTargetStore.open(config.output_dir)
    examples = load_text_examples(
        config.dataset_path, max_examples=store.metadata.num_examples
    )
    build_corridor_artifacts(
        output_dir=config.output_dir,
        examples=examples,
        selected_records=[],
        selected_payloads=[],
        delivery_path=config.exemplar_delivery_path or "one_pass_pruned_candidate",
        non_selected_exemplar_payload_retained=config.retain_unselected_exemplar_payloads,
    )
    return read_score_surface_corridor_evidence(config.output_dir)


def native_late_corridor_operation(inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import (
        SelectedArtifactCorridorEvidence,
        StageResult,
    )

    rerun = inputs.selected_rerun
    finalized = finalize_selected_delivery_corridor(rerun["prepared"])
    rerun["prepared"] = finalized
    corridor = finalized.corridor_result
    if corridor is None:
        raise ValueError("late corridor finalization returned no corridor evidence")
    output = finalized.config.artifact_dir
    evidence = native_file_evidence(
        "selected_artifact_corridor_finalization",
        (
            corridor.summary_path,
            corridor.fingerprints_path,
            corridor.modes_path,
            corridor.assignments_path,
        ),
        prior=inputs.selected_rerun_evidence,
    )
    return StageResult(
        status="pass",
        value=SelectedArtifactCorridorEvidence(
            stage_evidence=evidence,
            summary_path=corridor.summary_path,
            fingerprints_path=corridor.fingerprints_path,
            modes_path=corridor.modes_path,
            assignments_path=corridor.assignments_path,
            positions_available=corridor.positions_available,
            positions_used=corridor.positions_used,
            fingerprint_count=corridor.fingerprint_count,
            mode_count=corridor.mode_count,
            assignment_count=corridor.assignment_count,
            selected_exemplar_count=len(finalized.selected_payloads),
            selected_exemplars_linked=corridor.selected_exemplars_linked,
            delivery_report_path=output / "delivery_report.json",
            authority_manifest_path=output / "c6" / "authority_manifest.json",
            delivery_authority_hash=str(finalized.config.delivery_authority_hash or ""),
        ),
        evidence=evidence,
    )


def native_artifact_assembly_operation(inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.assembly import ArtifactAssemblyHandoff
    from radjax_tome.builder.native_path_b.contracts import EvidenceCount, StageResult

    rerun = inputs.selected_rerun
    finalized = rerun["prepared"]
    report = assemble_selected_delivery_artifacts(finalized)
    output = finalized.config.artifact_dir
    evidence = native_file_evidence(
        "artifact_assembly",
        (
            output / "delivery_report.json",
            output / "leaderboards" / "selected_exemplars.json",
            output / "selected_exemplars" / "payload_index.json",
        ),
        counts=(
            EvidenceCount(
                "selected_exemplar_count", int(report["num_selected_exemplars"])
            ),
        ),
        prior=inputs.final_corridor_evidence,
    )
    return StageResult(
        status="pass",
        value=ArtifactAssemblyHandoff(
            value={
                "delivery_report": report,
                "context": rerun["context"],
                "prepared": finalized,
            },
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )
