"""Ordered M4B stage adapters with no persistent workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.authorities import (
    FingerprintAuthorityOperation,
    FingerprintAuthorityValueT,
    GlobalAuthorityOperation,
    GlobalAuthorityValueT,
    run_fingerprint_selection_authority_stage,
    run_global_authority_stage,
)
from radjax_tome.builder.native_path_b.contracts import StageResult
from radjax_tome.builder.native_path_b.corridors import (
    EarlyCorridorOperation,
    run_score_surface_corridor_stage,
)
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
from radjax_tome.builder.native_path_b.selection import (
    FingerprintAuthorityValueT as SelectionFingerprintAuthorityValueT,
)
from radjax_tome.builder.native_path_b.selection import (
    GlobalAuthorityValueT as SelectionGlobalAuthorityValueT,
)
from radjax_tome.builder.native_path_b.selection import (
    IntegratedSelectionHandoff,
    IntegratedSelectionOperation,
    SelectionValueT,
    run_integrated_selection_stage,
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


@dataclass(frozen=True)
class SliceTwoOperations(
    Generic[ScorePassValueT, FingerprintAuthorityValueT, GlobalAuthorityValueT]
):
    """Injected existing operations for early corridor and authority export."""

    early_corridor: EarlyCorridorOperation[ScorePassValueT]
    fingerprint_authority: FingerprintAuthorityOperation[FingerprintAuthorityValueT]
    global_authority: GlobalAuthorityOperation[
        FingerprintAuthorityValueT,
        GlobalAuthorityValueT,
    ]


@dataclass(frozen=True)
class SliceTwoExecution(
    Generic[
        PreflightValueT,
        ScorePassValueT,
        FingerprintAuthorityValueT,
        GlobalAuthorityValueT,
    ]
):
    """In-memory handoff after ordered early corridor and authority export."""

    slice_one: SliceOneExecution[PreflightValueT, ScorePassValueT]
    early_corridor: StageResult[object]
    fingerprint_authority: StageResult[FingerprintAuthorityValueT] | None
    global_authority: StageResult[GlobalAuthorityValueT] | None

    @property
    def status(self) -> str:
        results = (
            self.slice_one.preflight,
            self.slice_one.score_pass,
            self.early_corridor,
            self.fingerprint_authority,
            self.global_authority,
        )
        if all(item is not None and item.status == "pass" for item in results):
            return "pass"
        return "fail"


def run_slice_two(
    config: CanonicalPathBConfig,
    slice_one: SliceOneExecution[PreflightValueT, ScorePassValueT],
    *,
    operations: SliceTwoOperations[
        ScorePassValueT,
        FingerprintAuthorityValueT,
        GlobalAuthorityValueT,
    ],
) -> SliceTwoExecution[
    PreflightValueT,
    ScorePassValueT,
    FingerprintAuthorityValueT,
    GlobalAuthorityValueT,
]:
    """Run score-surface corridor, fingerprint authority, then global authority."""

    early_corridor = run_score_surface_corridor_stage(
        config,
        slice_one.score_pass,
        operation=operations.early_corridor,
    )
    if early_corridor.status != "pass":
        return SliceTwoExecution(
            slice_one=slice_one,
            early_corridor=early_corridor,
            fingerprint_authority=None,
            global_authority=None,
        )
    fingerprint_authority = run_fingerprint_selection_authority_stage(
        config,
        early_corridor,
        operation=operations.fingerprint_authority,
    )
    if fingerprint_authority.status != "pass":
        return SliceTwoExecution(
            slice_one=slice_one,
            early_corridor=early_corridor,
            fingerprint_authority=fingerprint_authority,
            global_authority=None,
        )
    global_authority = run_global_authority_stage(
        config,
        early_corridor,
        fingerprint_authority,
        operation=operations.global_authority,
    )
    return SliceTwoExecution(
        slice_one=slice_one,
        early_corridor=early_corridor,
        fingerprint_authority=fingerprint_authority,
        global_authority=global_authority,
    )


@dataclass(frozen=True)
class SliceThreeOperations(
    Generic[
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
        SelectionValueT,
    ]
):
    """Injected existing C2-C5 operation for the third native slice."""

    integrated_selection: IntegratedSelectionOperation[
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
        SelectionValueT,
    ]


@dataclass(frozen=True)
class SliceThreeExecution(
    Generic[
        PreflightValueT,
        ScorePassValueT,
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
        SelectionValueT,
    ]
):
    """In-memory C2-C5 handoff; rerun and late corridor remain out of scope."""

    slice_two: SliceTwoExecution[
        PreflightValueT,
        ScorePassValueT,
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
    ]
    integrated_selection: (
        StageResult[IntegratedSelectionHandoff[SelectionValueT]] | None
    )

    @property
    def status(self) -> str:
        if self.slice_two.status != "pass":
            return "fail"
        if (
            self.integrated_selection is None
            or self.integrated_selection.status != "pass"
        ):
            return "fail"
        return "pass"


def run_slice_three(
    config: CanonicalPathBConfig,
    slice_two: SliceTwoExecution[
        PreflightValueT,
        ScorePassValueT,
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
    ],
    *,
    operations: SliceThreeOperations[
        SelectionFingerprintAuthorityValueT,
        SelectionGlobalAuthorityValueT,
        SelectionValueT,
    ],
) -> SliceThreeExecution[
    PreflightValueT,
    ScorePassValueT,
    SelectionFingerprintAuthorityValueT,
    SelectionGlobalAuthorityValueT,
    SelectionValueT,
]:
    """Run C2-C5 after both authorities; no delivery stages are invoked."""

    if slice_two.status != "pass":
        return SliceThreeExecution(slice_two=slice_two, integrated_selection=None)
    integrated_selection = run_integrated_selection_stage(
        config,
        fingerprint_authority=slice_two.fingerprint_authority,
        global_authority=slice_two.global_authority,
        operation=operations.integrated_selection,
    )
    return SliceThreeExecution(
        slice_two=slice_two,
        integrated_selection=integrated_selection,
    )
