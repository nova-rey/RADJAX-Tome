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
from radjax_tome.builder.native_path_b.contracts import (
    ScoreSurfaceCorridorEvidence,
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.orchestrator import (
    SliceOneOperations,
    SliceTwoOperations,
    run_preflight_then_score_pass,
    run_slice_two,
)
from radjax_tome.builder.native_path_b.preflight import run_preflight_stage
from radjax_tome.builder.native_path_b.score_pass import run_score_pass_stage


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
