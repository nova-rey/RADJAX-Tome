from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.native_path_b.api import (
    CANONICAL_DELIVERY_PATH,
    CANONICAL_SELECTION_INTEGRATION_POLICY,
    CANONICAL_TARGET_POLICY,
    CanonicalPathBConfig,
    resolve_canonical_path_b_config,
)
from radjax_tome.builder.native_path_b.assembly import (
    ArtifactAssemblyHandoff,
    ArtifactAssemblyInputs,
)
from radjax_tome.builder.native_path_b.contracts import (
    ScoreSurfaceCorridorEvidence,
    SelectedArtifactCorridorEvidence,
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.delivery import (
    LateCorridorInputs,
    SelectedDeliveryInputs,
    SelectedRerunHandoff,
    run_selected_artifact_corridor_finalization_stage,
    run_selected_delivery_rerun_stage,
)
from radjax_tome.builder.native_path_b.orchestrator import (
    SliceFourOperations,
    SliceOneOperations,
    SliceThreeOperations,
    SliceTwoOperations,
    run_preflight_then_score_pass,
    run_slice_four,
    run_slice_three,
    run_slice_two,
)
from radjax_tome.builder.native_path_b.preflight import run_preflight_stage
from radjax_tome.builder.native_path_b.score_pass import run_score_pass_stage
from radjax_tome.builder.native_path_b.selection import (
    IntegratedSelectionHandoff,
    SelectionAuthorityInputs,
    run_integrated_selection_stage,
)


@dataclass(frozen=True)
class _ConfigSource:
    target_policy: str = CANONICAL_TARGET_POLICY
    selection_integration_policy: str = CANONICAL_SELECTION_INTEGRATION_POLICY
    exemplar_selection_enabled: bool = True
    exemplar_delivery_path: str | None = CANONICAL_DELIVERY_PATH
    total_selected_exemplar_budget: int | None = 2


def _canonical_config() -> CanonicalPathBConfig:
    config = resolve_canonical_path_b_config(_ConfigSource())
    assert config is not None
    return config


def _passed(stage: str, value: Any) -> StageResult[Any]:
    return StageResult(
        status="pass",
        value=value,
        evidence=StageEvidence(stage=stage, paths=(), hashes=(), counts=()),
    )


def _failed(stage: str, reason: str) -> StageResult[Any]:
    return StageResult(
        status="fail",
        value=None,
        evidence=None,
        failure=StageFailure(stage=stage, reason=reason, blockers=(reason,)),
    )


def _provisional_early_evidence() -> StageResult[ScoreSurfaceCorridorEvidence]:
    stage_evidence = StageEvidence(
        stage="score_surface_corridor_materialization",
        paths=(),
        hashes=(),
        counts=(),
    )
    return StageResult(
        status="pass",
        value=ScoreSurfaceCorridorEvidence(
            stage_evidence=stage_evidence,
            summary_path=Path("corridors/corridor_summary.json"),
            fingerprints_path=Path("corridors/corridor_fingerprints.json"),
            modes_path=Path("corridors/corridor_modes.json"),
            assignments_path=Path("corridors/mode_assignments.json"),
            positions_available=2,
            positions_used=2,
            fingerprint_count=2,
            mode_count=2,
            assignment_count=2,
            selected_exemplar_count=0,
            selected_exemplars_linked=False,
        ),
        evidence=stage_evidence,
    )


def _integrated_selection_result() -> StageResult[IntegratedSelectionHandoff[str]]:
    stage_evidence = StageEvidence(
        stage="integrated_selection",
        paths=(),
        hashes=(),
        counts=(),
    )
    return StageResult(
        status="pass",
        value=IntegratedSelectionHandoff(
            value="selected",
            stage_evidence=stage_evidence,
            c2_evidence=StageEvidence(
                stage="c2_corridor_candidate_leaderboards",
                paths=(),
                hashes=(),
                counts=(),
            ),
            c3_evidence=StageEvidence(
                stage="c3_corridor_coverage_plan",
                paths=(),
                hashes=(),
                counts=(),
            ),
            c4_evidence=StageEvidence(
                stage="c4_corridor_global_claims",
                paths=(),
                hashes=(),
                counts=(),
            ),
            c5_evidence=StageEvidence(
                stage="c5_multi_role_selection",
                paths=(),
                hashes=(),
                counts=(),
            ),
        ),
        evidence=stage_evidence,
    )


def _selected_rerun_result() -> StageResult[SelectedRerunHandoff[str]]:
    stage_evidence = StageEvidence(
        stage="selected_delivery_rerun",
        paths=(),
        hashes=(),
        counts=(),
    )
    return StageResult(
        status="pass",
        value=SelectedRerunHandoff(value="rerun", stage_evidence=stage_evidence),
        evidence=stage_evidence,
    )


def _selected_artifact_corridor_result() -> StageResult[
    SelectedArtifactCorridorEvidence
]:
    stage_evidence = StageEvidence(
        stage="selected_artifact_corridor_finalization",
        paths=(),
        hashes=(),
        counts=(),
    )
    return StageResult(
        status="pass",
        value=SelectedArtifactCorridorEvidence(
            stage_evidence=stage_evidence,
            summary_path=Path("corridors/corridor_summary.json"),
            fingerprints_path=Path("corridors/corridor_fingerprints.json"),
            modes_path=Path("corridors/corridor_modes.json"),
            assignments_path=Path("corridors/mode_assignments.json"),
            positions_available=2,
            positions_used=2,
            fingerprint_count=2,
            mode_count=2,
            assignment_count=2,
            selected_exemplar_count=1,
            selected_exemplars_linked=True,
            delivery_report_path=Path("delivery_report.json"),
            authority_manifest_path=Path("c6/authority_manifest.json"),
            delivery_authority_hash="sha256:authority",
        ),
        evidence=stage_evidence,
    )


def _assembly_result() -> StageResult[ArtifactAssemblyHandoff[str]]:
    stage_evidence = StageEvidence(
        stage="artifact_assembly",
        paths=(),
        hashes=(),
        counts=(),
    )
    return StageResult(
        status="pass",
        value=ArtifactAssemblyHandoff(value="assembled", stage_evidence=stage_evidence),
        evidence=stage_evidence,
    )


def _passing_slice_two():
    canonical = _canonical_config()
    slice_one = run_preflight_then_score_pass(
        canonical,
        operations=SliceOneOperations(
            preflight=lambda config: _passed("preflight", "ready"),
            score_pass=lambda config, preflight_value: _passed("score_pass", "scored"),
        ),
    )
    return canonical, run_slice_two(
        canonical,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=lambda config, score_value: _provisional_early_evidence(),
            fingerprint_authority=lambda config, early_evidence: _passed(
                "fingerprint_corridor_selection_authority_export", "fingerprint"
            ),
            global_authority=lambda config, early_evidence, fingerprint_value: _passed(
                "global_authority_export", "global"
            ),
        ),
    )


def _passing_slice_three():
    canonical, slice_two = _passing_slice_two()
    return canonical, run_slice_three(
        canonical,
        slice_two,
        operations=SliceThreeOperations(
            integrated_selection=lambda config, authorities: (
                _integrated_selection_result()
            )
        ),
    )


def test_slice_one_runs_only_preflight_then_score_with_no_artifact_side_effects(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []
    untouched_output = tmp_path / "artifact"

    def preflight(config: CanonicalPathBConfig) -> StageResult[str]:
        events.append(("preflight", config))
        return _passed("preflight", "preflight-value")

    def score(
        config: CanonicalPathBConfig,
        preflight_value: str,
    ) -> StageResult[dict[str, str]]:
        events.append(("score_pass", (config, preflight_value)))
        return _passed("score_pass", {"score": "value"})

    execution = run_preflight_then_score_pass(
        _canonical_config(),
        operations=SliceOneOperations(preflight=preflight, score_pass=score),
    )

    assert [event[0] for event in events] == ["preflight", "score_pass"]
    assert events[1][1][1] == "preflight-value"
    assert execution.status == "pass"
    assert execution.preflight.evidence.stage == "preflight"
    assert execution.score_pass is not None
    assert execution.score_pass.evidence.stage == "score_pass"
    assert not untouched_output.exists()
    assert not (untouched_output / "production_progress.json").exists()
    assert not (untouched_output / "production_build_report.json").exists()
    assert not (untouched_output / "corridors").exists()


def test_slice_one_failure_skips_score() -> None:
    events: list[str] = []
    original_failure = _failed("preflight", "known_preflight_blocker")

    def preflight(config: CanonicalPathBConfig) -> StageResult[Any]:
        events.append("preflight")
        return original_failure

    def score(config: CanonicalPathBConfig, value: Any) -> StageResult[Any]:
        events.append("score_pass")
        return _passed("score_pass", "unexpected")

    execution = run_preflight_then_score_pass(
        _canonical_config(),
        operations=SliceOneOperations(preflight=preflight, score_pass=score),
    )

    assert events == ["preflight"]
    assert execution.status == "fail"
    assert execution.preflight is original_failure
    assert execution.score_pass is None
    assert execution.preflight.failure.reason == "known_preflight_blocker"


def test_stage_adapters_normalize_failures() -> None:
    canonical = _canonical_config()

    def exploding_preflight(config: CanonicalPathBConfig) -> StageResult[Any]:
        raise RuntimeError("preflight boom")

    preflight_failure = run_preflight_stage(canonical, operation=exploding_preflight)
    assert preflight_failure.status == "fail"
    assert preflight_failure.failure.stage == "preflight"
    assert preflight_failure.failure.reason == "preflight_operation_failed"
    assert preflight_failure.failure.resumable is True

    wrong_preflight = run_preflight_stage(
        canonical,
        operation=lambda config: _passed("score_pass", "wrong-stage"),
    )
    assert wrong_preflight.status == "fail"
    assert wrong_preflight.failure.reason == "preflight_evidence_stage_mismatch"

    ready_preflight = _passed("preflight", "ready")
    score_failure = run_score_pass_stage(
        canonical,
        ready_preflight,
        operation=lambda config, value: _passed("preflight", "wrong-stage"),
    )
    assert score_failure.status == "fail"
    assert score_failure.failure.stage == "score_pass"
    assert score_failure.failure.reason == "score_pass_evidence_stage_mismatch"


def test_global_only_skips_slice_one() -> None:
    global_only = _ConfigSource(
        target_policy="dynamic_cascaded_soft_labels_v1",
        selection_integration_policy="global_only_v1",
        exemplar_selection_enabled=False,
        exemplar_delivery_path=None,
        total_selected_exemplar_budget=None,
    )

    assert resolve_canonical_path_b_config(global_only) is None


def test_slice_two_orders_provisional_corridor_and_authorities_without_writes(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []
    untouched_output = tmp_path / "artifact"

    def preflight(config: CanonicalPathBConfig) -> StageResult[str]:
        events.append(("preflight", config))
        return _passed("preflight", "ready")

    def score(
        config: CanonicalPathBConfig,
        preflight_value: str,
    ) -> StageResult[str]:
        events.append(("score_pass", preflight_value))
        return _passed("score_pass", "scored")

    def early_corridor(
        config: CanonicalPathBConfig,
        score_value: str,
    ) -> StageResult[ScoreSurfaceCorridorEvidence]:
        events.append(("early_corridor", score_value))
        return _provisional_early_evidence()

    def fingerprint_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
    ) -> StageResult[str]:
        events.append(("fingerprint_authority", early_evidence))
        return _passed("fingerprint_corridor_selection_authority_export", "fingerprint")

    def global_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
        fingerprint_value: str,
    ) -> StageResult[str]:
        events.append(("global_authority", (early_evidence, fingerprint_value)))
        return _passed("global_authority_export", "global")

    canonical = _canonical_config()
    slice_one = run_preflight_then_score_pass(
        canonical,
        operations=SliceOneOperations(preflight=preflight, score_pass=score),
    )
    execution = run_slice_two(
        canonical,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=early_corridor,
            fingerprint_authority=fingerprint_authority,
            global_authority=global_authority,
        ),
    )

    assert [event[0] for event in events] == [
        "preflight",
        "score_pass",
        "early_corridor",
        "fingerprint_authority",
        "global_authority",
    ]
    assert events[2][1] == "scored"
    assert execution.status == "pass"
    assert execution.early_corridor.value.selected_exemplar_count == 0
    assert execution.early_corridor.value.selected_exemplars_linked is False
    assert execution.fingerprint_authority is not None
    assert execution.global_authority is not None
    assert not untouched_output.exists()
    assert not (untouched_output / "selected_exemplars").exists()
    assert not (untouched_output / "reruns").exists()
    assert not (untouched_output / "late_corridor").exists()
    assert not (untouched_output / "evidence").exists()


def test_slice_two_failed_early_corridor_skips_both_authorities() -> None:
    events: list[str] = []

    def preflight(config: CanonicalPathBConfig) -> StageResult[str]:
        events.append("preflight")
        return _passed("preflight", "ready")

    def score(
        config: CanonicalPathBConfig,
        preflight_value: str,
    ) -> StageResult[str]:
        events.append("score_pass")
        return _passed("score_pass", "scored")

    def early_corridor(
        config: CanonicalPathBConfig,
        score_value: str,
    ) -> StageResult[ScoreSurfaceCorridorEvidence]:
        events.append("early_corridor")
        return _failed(
            "score_surface_corridor_materialization",
            "known_early_corridor_blocker",
        )

    def fingerprint_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
    ) -> StageResult[str]:
        events.append("fingerprint_authority")
        return _passed("fingerprint_corridor_selection_authority_export", "unexpected")

    def global_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
        fingerprint_value: str,
    ) -> StageResult[str]:
        events.append("global_authority")
        return _passed("global_authority_export", "unexpected")

    canonical = _canonical_config()
    slice_one = run_preflight_then_score_pass(
        canonical,
        operations=SliceOneOperations(preflight=preflight, score_pass=score),
    )
    execution = run_slice_two(
        canonical,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=early_corridor,
            fingerprint_authority=fingerprint_authority,
            global_authority=global_authority,
        ),
    )

    assert events == ["preflight", "score_pass", "early_corridor"]
    assert execution.status == "fail"
    assert execution.early_corridor.failure.reason == "known_early_corridor_blocker"
    assert execution.fingerprint_authority is None
    assert execution.global_authority is None


def test_slice_three_orders_authority_backed_c2_through_c5_without_rerun_writes(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    untouched_output = tmp_path / "artifact"
    canonical = _canonical_config()

    def preflight(config: CanonicalPathBConfig) -> StageResult[str]:
        events.append("preflight")
        return _passed("preflight", "ready")

    def score(
        config: CanonicalPathBConfig,
        preflight_value: str,
    ) -> StageResult[str]:
        events.append("score_pass")
        return _passed("score_pass", "scored")

    def early_corridor(
        config: CanonicalPathBConfig,
        score_value: str,
    ) -> StageResult[ScoreSurfaceCorridorEvidence]:
        events.append("early_corridor")
        return _provisional_early_evidence()

    def fingerprint_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
    ) -> StageResult[str]:
        events.append("fingerprint_authority")
        return _passed("fingerprint_corridor_selection_authority_export", "fingerprint")

    def global_authority(
        config: CanonicalPathBConfig,
        early_evidence: ScoreSurfaceCorridorEvidence,
        fingerprint_value: str,
    ) -> StageResult[str]:
        events.append("global_authority")
        return _passed("global_authority_export", "global")

    def integrated_selection(
        config: CanonicalPathBConfig,
        authorities: SelectionAuthorityInputs[str, str],
    ) -> StageResult[IntegratedSelectionHandoff[str]]:
        assert authorities.fingerprint_value == "fingerprint"
        assert (
            authorities.fingerprint_evidence.stage
            == "fingerprint_corridor_selection_authority_export"
        )
        assert authorities.global_value == "global"
        assert authorities.global_evidence.stage == "global_authority_export"
        events.extend(("c2", "c3", "c4", "c5"))
        return _integrated_selection_result()

    slice_one = run_preflight_then_score_pass(
        canonical,
        operations=SliceOneOperations(preflight=preflight, score_pass=score),
    )
    slice_two = run_slice_two(
        canonical,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=early_corridor,
            fingerprint_authority=fingerprint_authority,
            global_authority=global_authority,
        ),
    )
    execution = run_slice_three(
        canonical,
        slice_two,
        operations=SliceThreeOperations(integrated_selection=integrated_selection),
    )

    assert events == [
        "preflight",
        "score_pass",
        "early_corridor",
        "fingerprint_authority",
        "global_authority",
        "c2",
        "c3",
        "c4",
        "c5",
    ]
    assert execution.status == "pass"
    assert execution.integrated_selection is not None
    assert execution.integrated_selection.value.c2_evidence.stage.startswith("c2_")
    assert execution.integrated_selection.value.c3_evidence.stage.startswith("c3_")
    assert execution.integrated_selection.value.c4_evidence.stage.startswith("c4_")
    assert execution.integrated_selection.value.c5_evidence.stage.startswith("c5_")
    assert not untouched_output.exists()
    assert not (untouched_output / "reruns").exists()
    assert not (untouched_output / "late_corridor").exists()


def test_slice_three_requires_authority_evidence_before_selection() -> None:
    events: list[str] = []

    def selection(
        config: CanonicalPathBConfig,
        authorities: SelectionAuthorityInputs[str, str],
    ) -> StageResult[IntegratedSelectionHandoff[str]]:
        events.append("selection")
        return _integrated_selection_result()

    result = run_integrated_selection_stage(
        _canonical_config(),
        fingerprint_authority=None,
        global_authority=_passed("global_authority_export", "global"),
        operation=selection,
    )

    assert result.status == "fail"
    assert result.failure.reason == "fingerprint_authority_required"
    assert events == []


def test_slice_three_selection_failure_stops_before_rerun_and_normalizes_callback_error(
    tmp_path: Path,
) -> None:
    canonical, slice_two = _passing_slice_two()
    untouched_output = tmp_path / "artifact"

    def failed_selection(
        config: CanonicalPathBConfig,
        authorities: SelectionAuthorityInputs[str, str],
    ) -> StageResult[IntegratedSelectionHandoff[str]]:
        return _failed("integrated_selection", "known_selection_blocker")

    execution = run_slice_three(
        canonical,
        slice_two,
        operations=SliceThreeOperations(integrated_selection=failed_selection),
    )

    assert execution.status == "fail"
    assert execution.integrated_selection is not None
    assert execution.integrated_selection.failure.reason == "known_selection_blocker"
    assert not untouched_output.exists()
    assert not (untouched_output / "reruns").exists()
    assert not (untouched_output / "late_corridor").exists()

    def exploding_selection(
        config: CanonicalPathBConfig,
        authorities: SelectionAuthorityInputs[str, str],
    ) -> StageResult[IntegratedSelectionHandoff[str]]:
        raise RuntimeError("selection boom")

    normalized = run_integrated_selection_stage(
        canonical,
        fingerprint_authority=slice_two.fingerprint_authority,
        global_authority=slice_two.global_authority,
        operation=exploding_selection,
    )

    assert normalized.status == "fail"
    assert normalized.failure.reason == "integrated_selection_operation_failed"
    assert normalized.failure.resumable is True


def test_slice_four_orders_c5_rerun_selected_corridor_and_assembly_without_writes(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    untouched_output = tmp_path / "artifact"
    canonical, slice_two = _passing_slice_two()

    def integrated_selection(
        config: CanonicalPathBConfig,
        authorities: SelectionAuthorityInputs[str, str],
    ) -> StageResult[IntegratedSelectionHandoff[str]]:
        events.append("c5")
        return _integrated_selection_result()

    slice_three = run_slice_three(
        canonical,
        slice_two,
        operations=SliceThreeOperations(integrated_selection=integrated_selection),
    )

    def selected_rerun(
        config: CanonicalPathBConfig,
        inputs: SelectedDeliveryInputs[str],
    ) -> StageResult[SelectedRerunHandoff[str]]:
        assert inputs.selection == "selected"
        assert inputs.c5_evidence.stage == "c5_multi_role_selection"
        events.append("selected_rerun")
        return _selected_rerun_result()

    def late_corridor(
        config: CanonicalPathBConfig,
        inputs: LateCorridorInputs[str, str],
    ) -> StageResult[SelectedArtifactCorridorEvidence]:
        assert inputs.selected_rerun == "rerun"
        assert inputs.selected_rerun_evidence.stage == "selected_delivery_rerun"
        events.append("late_corridor")
        return _selected_artifact_corridor_result()

    def assembly(
        config: CanonicalPathBConfig,
        inputs: ArtifactAssemblyInputs[str, str],
    ) -> StageResult[ArtifactAssemblyHandoff[str]]:
        assert inputs.final_corridor.selected_exemplar_count == 1
        assert inputs.final_corridor.selected_exemplars_linked is True
        events.append("assembly")
        return _assembly_result()

    execution = run_slice_four(
        canonical,
        slice_three,
        operations=SliceFourOperations(
            selected_rerun=selected_rerun,
            late_corridor=late_corridor,
            assembly=assembly,
        ),
    )

    assert events == ["c5", "selected_rerun", "late_corridor", "assembly"]
    assert execution.status == "pass"
    assert execution.late_corridor is not None
    assert execution.late_corridor.value.selected_exemplars_linked is True
    assert execution.assembly is not None
    assert not untouched_output.exists()
    assert not (untouched_output / "validation").exists()
    assert not (untouched_output / "reconciliation").exists()
    assert not (untouched_output / "reports").exists()


def test_slice_four_late_corridor_failure_stops_assembly() -> None:
    events: list[str] = []
    canonical, slice_three = _passing_slice_three()

    def selected_rerun(
        config: CanonicalPathBConfig,
        inputs: SelectedDeliveryInputs[str],
    ) -> StageResult[SelectedRerunHandoff[str]]:
        events.append("selected_rerun")
        return _selected_rerun_result()

    def late_corridor(
        config: CanonicalPathBConfig,
        inputs: LateCorridorInputs[str, str],
    ) -> StageResult[SelectedArtifactCorridorEvidence]:
        events.append("late_corridor")
        return _failed(
            "selected_artifact_corridor_finalization",
            "known_late_corridor_blocker",
        )

    def assembly(
        config: CanonicalPathBConfig,
        inputs: ArtifactAssemblyInputs[str, str],
    ) -> StageResult[ArtifactAssemblyHandoff[str]]:
        events.append("assembly")
        return _assembly_result()

    execution = run_slice_four(
        canonical,
        slice_three,
        operations=SliceFourOperations(
            selected_rerun=selected_rerun,
            late_corridor=late_corridor,
            assembly=assembly,
        ),
    )

    assert events == ["selected_rerun", "late_corridor"]
    assert execution.status == "fail"
    assert execution.late_corridor is not None
    assert execution.late_corridor.failure.reason == "known_late_corridor_blocker"
    assert execution.assembly is None


def test_late_corridor_rejects_provisional_evidence() -> None:
    canonical, slice_three = _passing_slice_three()
    rerun = run_selected_delivery_rerun_stage(
        canonical,
        slice_three.integrated_selection,
        operation=lambda config, inputs: _selected_rerun_result(),
    )

    def provisional_late_corridor(
        config: CanonicalPathBConfig,
        inputs: LateCorridorInputs[str, str],
    ) -> StageResult[SelectedArtifactCorridorEvidence]:
        return _provisional_early_evidence()

    late_corridor = run_selected_artifact_corridor_finalization_stage(
        canonical,
        slice_three.integrated_selection,
        rerun,
        operation=provisional_late_corridor,
    )

    assert late_corridor.status == "fail"
    assert (
        late_corridor.failure.reason
        == "selected_artifact_corridor_evidence_stage_mismatch"
    )
