"""Ordered M4B stage adapter; slice one owns preflight then score pass only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import StageResult
from radjax_tome.builder.native_path_b.preflight import (
    PreflightOperation,
    PreflightValueT,
    run_preflight_stage,
)
from radjax_tome.builder.native_path_b.score_pass import (
    ScorePassOperation,
    ScorePassValueT,
    run_score_pass_stage,
)


@dataclass(frozen=True)
class SliceOneOperations(Generic[PreflightValueT, ScorePassValueT]):
    """Injected existing operations for the first two canonical stages."""

    preflight: PreflightOperation[PreflightValueT]
    score_pass: ScorePassOperation[PreflightValueT, ScorePassValueT]


@dataclass(frozen=True)
class SliceOneExecution(Generic[PreflightValueT, ScorePassValueT]):
    """In-memory handoff for later explicit native Path-B stages."""

    preflight: StageResult[PreflightValueT]
    score_pass: StageResult[ScorePassValueT] | None

    @property
    def status(self) -> str:
        if self.preflight.status != "pass":
            return "fail"
        if self.score_pass is None or self.score_pass.status != "pass":
            return "fail"
        return "pass"


def run_preflight_then_score_pass(
    config: CanonicalPathBConfig,
    *,
    operations: SliceOneOperations[PreflightValueT, ScorePassValueT],
) -> SliceOneExecution[PreflightValueT, ScorePassValueT]:
    """Execute only ordered slice one; later stages are intentionally absent."""

    preflight = run_preflight_stage(config, operation=operations.preflight)
    if preflight.status != "pass":
        return SliceOneExecution(preflight=preflight, score_pass=None)
    score_pass = run_score_pass_stage(
        config,
        preflight,
        operation=operations.score_pass,
    )
    return SliceOneExecution(preflight=preflight, score_pass=score_pass)
