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
    StageEvidence,
    StageFailure,
    StageResult,
)
from radjax_tome.builder.native_path_b.orchestrator import (
    SliceOneOperations,
    run_preflight_then_score_pass,
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
