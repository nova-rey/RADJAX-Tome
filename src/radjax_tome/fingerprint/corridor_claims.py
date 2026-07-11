"""C4 corridor-first coordinate claims and global-board backfill."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetError,
    CorridorCoveragePlan,
    validate_corridor_coverage_plan_object,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardArtifact,
    CorridorLeaderboardError,
    validate_corridor_candidate_leaderboard_artifact,
)
from radjax_tome.io.json import read_json_object, write_json

CORRIDOR_CLAIM_SCHEMA = "radjax.c4_corridor_global_claims.v1"
CORRIDOR_CLAIM_VALIDATION_SCHEMA = "radjax.c4_corridor_global_claims_validation.v1"
GLOBAL_BOARD_SUPPLY_SCHEMA = "radjax.c4_global_board_supply.v1"
CLAIM_POLICY_ID = "corridor_first_global_backfill_v1"
CLAIM_MANIFEST_FILENAME = "claim_manifest.json"
CORRIDOR_CLAIMS_FILENAME = "corridor_claims.jsonl"
GLOBAL_CLAIMS_FILENAME = "global_claims.jsonl"
COLLISION_OBLIGATIONS_FILENAME = "collision_obligations.jsonl"
SELECTED_COORDINATES_FILENAME = "selected_coordinates.jsonl"
CLAIM_VALIDATION_FILENAME = "validation_report.json"
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "logits",
        "dense_logits",
        "top_probs",
        "top_log_probs",
        "payload_ref",
        "source_text",
        "input_ids",
    }
)


class CorridorClaimError(ValueError):
    """Actionable C4 input, claim, or artifact error."""


@dataclass(frozen=True)
class CorridorGlobalClaimPolicy:
    total_selected_exemplar_budget: int
    require_full_budget: bool = True
    claim_policy_id: str = CLAIM_POLICY_ID
    allow_nonproduction_sources: bool = False
    nonproduction_override_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.total_selected_exemplar_budget, bool) or not isinstance(
            self.total_selected_exemplar_budget, int
        ):
            raise ValueError(
                "total_selected_exemplar_budget must be a nonnegative integer"
            )
        if self.total_selected_exemplar_budget < 0:
            raise ValueError("total_selected_exemplar_budget must be nonnegative")
        if not isinstance(self.require_full_budget, bool):
            raise TypeError("require_full_budget must be a boolean")
        if self.claim_policy_id != CLAIM_POLICY_ID:
            raise ValueError(f"unsupported claim policy: {self.claim_policy_id}")
        if not isinstance(self.allow_nonproduction_sources, bool):
            raise TypeError("allow_nonproduction_sources must be a boolean")
        if self.allow_nonproduction_sources and not self.nonproduction_override_reason:
            raise ValueError(
                "nonproduction_override_reason is required for non-production sources"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_CLAIM_SCHEMA,
            "claim_policy_id": self.claim_policy_id,
            "total_selected_exemplar_budget": self.total_selected_exemplar_budget,
            "require_full_budget": self.require_full_budget,
            "allow_nonproduction_sources": self.allow_nonproduction_sources,
            "nonproduction_override_reason": self.nonproduction_override_reason,
        }


@dataclass(frozen=True)
class SelectionObligation:
    role: Literal["fingerprint_corridor_representative", "global_board"]
    source_id: str
    rank: int
    score: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"fingerprint_corridor_representative", "global_board"}:
            raise ValueError("unsupported selection obligation role")
        if not self.source_id:
            raise ValueError("obligation source_id must be nonempty")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("obligation rank must be a positive integer")
        if self.score is not None and (
            isinstance(self.score, bool) or not math.isfinite(float(self.score))
        ):
            raise ValueError("obligation score must be finite or None")
        _reject_payload_fields(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_id": self.source_id,
            "rank": self.rank,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GlobalBoardCandidate:
    example_id: str
    position: int
    rank: int
    score: float
    eligible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.example_id, self.position)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("global candidate example_id must be nonempty")
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position < 0
        ):
            raise ValueError("global candidate position must be a nonnegative integer")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("global candidate rank must be a positive integer")
        if isinstance(self.score, bool) or not math.isfinite(float(self.score)):
            raise ValueError("global candidate score must be finite")
        if not isinstance(self.eligible, bool):
            raise TypeError("global candidate eligible must be a boolean")
        _reject_payload_fields(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "position": self.position,
            "rank": self.rank,
            "score": self.score,
            "eligible": self.eligible,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GlobalBoard:
    board_id: str
    priority: int
    requested_slots: int
    candidates: tuple[GlobalBoardCandidate, ...]

    def __post_init__(self) -> None:
        if not self.board_id:
            raise ValueError("global board_id must be nonempty")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("global board priority must be an integer")
        if (
            isinstance(self.requested_slots, bool)
            or not isinstance(self.requested_slots, int)
            or self.requested_slots < 0
        ):
            raise ValueError("global requested_slots must be a nonnegative integer")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError(
                "global board candidate ranks must be contiguous and ordered"
            )
        identities = [candidate.coordinate for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("global board contains duplicate coordinates")
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_id": self.board_id,
            "priority": self.priority,
            "requested_slots": self.requested_slots,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ExistingGlobalBoardInput:
    boards: tuple[GlobalBoard, ...]
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [board.board_id for board in self.boards]
        if len(ids) != len(set(ids)):
            raise ValueError("global board IDs must be unique")
        for board_id in ids:
            if not board_id:
                raise ValueError("global board ID must be nonempty")
        _reject_payload_fields(self.source_provenance)
        object.__setattr__(self, "boards", tuple(self.boards))
        object.__setattr__(self, "source_provenance", dict(self.source_provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_BOARD_SUPPLY_SCHEMA,
            "source_provenance": dict(self.source_provenance),
            "boards": [board.to_dict() for board in self.boards],
        }


@dataclass(frozen=True)
class CorridorClaim:
    example_id: str
    position: int
    corridor_mode_id: int
    c2_rank: int
    score: float
    score_components: Mapping[str, float]
    obligation: SelectionObligation

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.example_id, self.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "position": self.position,
            "corridor_mode_id": self.corridor_mode_id,
            "c2_rank": self.c2_rank,
            "score": self.score,
            "score_components": dict(self.score_components),
            "obligation": self.obligation.to_dict(),
        }


@dataclass(frozen=True)
class GlobalClaim:
    example_id: str
    position: int
    board_id: str
    board_priority: int
    global_rank: int
    score: float
    backfilled: bool
    obligation: SelectionObligation

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.example_id, self.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "position": self.position,
            "board_id": self.board_id,
            "board_priority": self.board_priority,
            "global_rank": self.global_rank,
            "score": self.score,
            "backfilled": self.backfilled,
            "obligation": self.obligation.to_dict(),
        }


@dataclass(frozen=True)
class CollisionObligation:
    example_id: str
    position: int
    board_id: str
    board_priority: int
    global_rank: int
    score: float
    collision_kind: Literal["corridor", "global"]
    occupied_primary_claim: str
    obligation: SelectionObligation

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.example_id, self.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "position": self.position,
            "board_id": self.board_id,
            "board_priority": self.board_priority,
            "global_rank": self.global_rank,
            "score": self.score,
            "collision_kind": self.collision_kind,
            "occupied_primary_claim": self.occupied_primary_claim,
            "obligation": self.obligation.to_dict(),
        }


@dataclass(frozen=True)
class BackfillLineage:
    board_id: str
    skipped_rank: int
    skipped_coordinate: tuple[str, int]
    reason: Literal[
        "corridor_collision", "global_collision", "ineligible", "supply_exhausted"
    ]
    replacement_rank: int | None
    replacement_coordinate: tuple[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_id": self.board_id,
            "skipped_rank": self.skipped_rank,
            "skipped_example_id": self.skipped_coordinate[0],
            "skipped_position": self.skipped_coordinate[1],
            "reason": self.reason,
            "replacement_rank": self.replacement_rank,
            "replacement_example_id": (
                None
                if self.replacement_coordinate is None
                else self.replacement_coordinate[0]
            ),
            "replacement_position": (
                None
                if self.replacement_coordinate is None
                else self.replacement_coordinate[1]
            ),
        }


@dataclass(frozen=True)
class ClaimedCoordinate:
    example_id: str
    position: int
    primary_claim: Literal["fingerprint_corridor_representative", "global_board"]
    claim_order: int
    obligations: tuple[SelectionObligation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_order": self.claim_order,
            "example_id": self.example_id,
            "position": self.position,
            "primary_claim": self.primary_claim,
            "obligations": [obligation.to_dict() for obligation in self.obligations],
        }


@dataclass(frozen=True)
class CorridorGlobalClaimResult:
    policy: CorridorGlobalClaimPolicy
    source_provenance: Mapping[str, Any]
    corridor_claims: tuple[CorridorClaim, ...]
    global_claims: tuple[GlobalClaim, ...]
    collision_obligations: tuple[CollisionObligation, ...]
    backfill_lineage: tuple[BackfillLineage, ...]
    selected_coordinates: tuple[ClaimedCoordinate, ...]
    summary: Mapping[str, Any] | None = None

    @property
    def production_grade(self) -> bool:
        return bool(self.source_provenance.get("production_grade", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_CLAIM_SCHEMA,
            "claim_policy": self.policy.to_dict(),
            "source_provenance": dict(self.source_provenance),
            "summary": dict(self.summary or _claim_summary(self)),
            "corridor_claims": [claim.to_dict() for claim in self.corridor_claims],
            "global_claims": [claim.to_dict() for claim in self.global_claims],
            "collision_obligations": [
                collision.to_dict() for collision in self.collision_obligations
            ],
            "backfill_lineage": [
                lineage.to_dict() for lineage in self.backfill_lineage
            ],
            "selected_coordinates": [
                coordinate.to_dict() for coordinate in self.selected_coordinates
            ],
        }


@dataclass(frozen=True)
class CorridorClaimValidationResult:
    status: Literal["pass", "fail", "warn"]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_CLAIM_VALIDATION_SCHEMA,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


def claim_corridor_then_backfill_global(
    leaderboards: CorridorLeaderboardArtifact,
    coverage_plan: CorridorCoveragePlan,
    global_boards: ExistingGlobalBoardInput | Mapping[str, Any],
    policy: CorridorGlobalClaimPolicy,
    *,
    _validate_result: bool = True,
) -> CorridorGlobalClaimResult:
    """Claim C3 corridor obligations first, then backfill global boards."""

    if not isinstance(policy, CorridorGlobalClaimPolicy):
        raise TypeError("policy must be CorridorGlobalClaimPolicy")
    if not isinstance(leaderboards, CorridorLeaderboardArtifact):
        raise TypeError("leaderboards must be CorridorLeaderboardArtifact")
    if not isinstance(coverage_plan, CorridorCoveragePlan):
        raise TypeError("coverage_plan must be CorridorCoveragePlan")
    global_input = _normalize_global_input(global_boards)
    _validate_c4_sources(leaderboards, coverage_plan, global_input, policy)
    if (
        policy.total_selected_exemplar_budget
        != coverage_plan.policy.total_selected_exemplar_budget
    ):
        raise CorridorClaimError("C4 total budget disagrees with C3 policy")
    c2_modes = {mode.corridor_mode_id: mode for mode in leaderboards.modes}
    _reject_cross_corridor_coordinates(leaderboards)
    claims_by_coordinate: dict[tuple[str, int], list[SelectionObligation]] = {}
    primary_by_coordinate: dict[tuple[str, int], str] = {}
    corridor_claims: list[CorridorClaim] = []
    for allocation in coverage_plan.modes:
        mode = c2_modes.get(allocation.corridor_mode_id)
        if mode is None:
            raise CorridorClaimError(
                f"C3 mode {allocation.corridor_mode_id} is missing from C2"
            )
        if allocation.allocated_slots > len(mode.candidates):
            raise CorridorClaimError(
                f"C3 mode {allocation.corridor_mode_id} exceeds C2 retained pool"
            )
        for rank, candidate in enumerate(
            mode.candidates[: allocation.allocated_slots], 1
        ):
            coordinate = (candidate.candidate_id, candidate.position)
            obligation = SelectionObligation(
                role="fingerprint_corridor_representative",
                source_id=str(allocation.corridor_mode_id),
                rank=rank,
                score=candidate.corridor_training_utility,
                metadata={
                    "corridor_mode_id": allocation.corridor_mode_id,
                    "c2_rank": rank,
                    "membership_score": candidate.membership_score,
                    "centrality_score": candidate.centrality_score,
                    "useful_difficulty_score": candidate.useful_difficulty_score,
                    "quality_score": candidate.quality_score,
                },
            )
            claims_by_coordinate[coordinate] = [obligation]
            primary_by_coordinate[coordinate] = "fingerprint_corridor_representative"
            corridor_claims.append(
                CorridorClaim(
                    example_id=coordinate[0],
                    position=coordinate[1],
                    corridor_mode_id=allocation.corridor_mode_id,
                    c2_rank=rank,
                    score=float(candidate.corridor_training_utility or 0.0),
                    score_components={
                        "membership_score": candidate.membership_score,
                        "centrality_score": candidate.centrality_score,
                        "useful_difficulty_score": candidate.useful_difficulty_score,
                        "quality_score": candidate.quality_score,
                    },
                    obligation=obligation,
                )
            )
    if len(corridor_claims) != coverage_plan.actual_corridor_budget:
        raise CorridorClaimError("corridor claims do not fulfill the C3 allocation")

    global_budget = coverage_plan.global_budget
    global_claims: list[GlobalClaim] = []
    collisions: list[CollisionObligation] = []
    lineage: list[BackfillLineage] = []
    board_summaries: list[dict[str, Any]] = []
    for board in sorted(
        global_input.boards, key=lambda item: (item.priority, item.board_id)
    ):
        remaining_global = global_budget - len(global_claims)
        target = min(board.requested_slots, max(0, remaining_global))
        fulfilled = 0
        pending: deque[tuple[GlobalBoardCandidate, str]] = deque()
        collision_count = 0
        skipped_ineligible = 0
        for candidate in board.candidates:
            if fulfilled >= target:
                break
            coordinate = candidate.coordinate
            if not candidate.eligible:
                skipped_ineligible += 1
                pending.append((candidate, "ineligible"))
                continue
            if coordinate in claims_by_coordinate:
                primary = primary_by_coordinate[coordinate]
                kind: Literal["corridor", "global"] = (
                    "corridor"
                    if primary == "fingerprint_corridor_representative"
                    else "global"
                )
                obligation = SelectionObligation(
                    role="global_board",
                    source_id=board.board_id,
                    rank=candidate.rank,
                    score=candidate.score,
                    metadata={
                        "board_priority": board.priority,
                        "collision_kind": kind,
                    },
                )
                claims_by_coordinate[coordinate].append(obligation)
                collisions.append(
                    CollisionObligation(
                        example_id=coordinate[0],
                        position=coordinate[1],
                        board_id=board.board_id,
                        board_priority=board.priority,
                        global_rank=candidate.rank,
                        score=candidate.score,
                        collision_kind=kind,
                        occupied_primary_claim=primary,
                        obligation=obligation,
                    )
                )
                pending.append((candidate, f"{kind}_collision"))
                collision_count += 1
                continue
            obligation = SelectionObligation(
                role="global_board",
                source_id=board.board_id,
                rank=candidate.rank,
                score=candidate.score,
                metadata={"board_priority": board.priority},
            )
            claims_by_coordinate[coordinate] = [obligation]
            primary_by_coordinate[coordinate] = "global_board"
            global_claims.append(
                GlobalClaim(
                    example_id=coordinate[0],
                    position=coordinate[1],
                    board_id=board.board_id,
                    board_priority=board.priority,
                    global_rank=candidate.rank,
                    score=candidate.score,
                    backfilled=bool(pending),
                    obligation=obligation,
                )
            )
            fulfilled += 1
            if pending:
                skipped, reason = pending.popleft()
                lineage.append(
                    BackfillLineage(
                        board_id=board.board_id,
                        skipped_rank=skipped.rank,
                        skipped_coordinate=skipped.coordinate,
                        reason=reason,  # type: ignore[arg-type]
                        replacement_rank=candidate.rank,
                        replacement_coordinate=candidate.coordinate,
                    )
                )
        for skipped, reason in pending:
            lineage.append(
                BackfillLineage(
                    board_id=board.board_id,
                    skipped_rank=skipped.rank,
                    skipped_coordinate=skipped.coordinate,
                    reason=reason,  # type: ignore[arg-type]
                    replacement_rank=None,
                    replacement_coordinate=None,
                )
            )
        board_summaries.append(
            {
                "board_id": board.board_id,
                "board_priority": board.priority,
                "requested_slots": board.requested_slots,
                "target_slots": target,
                "fulfilled_slots": fulfilled,
                "collision_count": collision_count,
                "skipped_ineligible_count": skipped_ineligible,
                "backfill_count": sum(
                    1
                    for claim in global_claims
                    if claim.board_id == board.board_id and claim.backfilled
                ),
                "supply_exhausted": fulfilled < target,
            }
        )

    selected: list[ClaimedCoordinate] = []
    ordered_coordinates = [claim.coordinate for claim in corridor_claims]
    ordered_coordinates.extend(claim.coordinate for claim in global_claims)
    for claim_order, coordinate in enumerate(ordered_coordinates):
        selected.append(
            ClaimedCoordinate(
                example_id=coordinate[0],
                position=coordinate[1],
                primary_claim=primary_by_coordinate[coordinate],  # type: ignore[arg-type]
                claim_order=claim_order,
                obligations=tuple(claims_by_coordinate[coordinate]),
            )
        )
    source = _claim_source_provenance(leaderboards, coverage_plan, global_input)
    result = CorridorGlobalClaimResult(
        policy=policy,
        source_provenance=source,
        corridor_claims=tuple(corridor_claims),
        global_claims=tuple(global_claims),
        collision_obligations=tuple(collisions),
        backfill_lineage=tuple(lineage),
        selected_coordinates=tuple(selected),
        summary={
            "board_summaries": board_summaries,
            "global_board_count": len(global_input.boards),
        },
    )
    result = replace(result, summary=_claim_summary(result))
    if _validate_result:
        validation = validate_corridor_global_claim_result(
            result,
            leaderboards=leaderboards,
            coverage_plan=coverage_plan,
            global_boards=global_input,
        )
        if validation.status == "fail":
            raise CorridorClaimError(
                "invalid C4 claim result: " + "; ".join(validation.blockers)
            )
    if (
        policy.require_full_budget
        and len(selected) != policy.total_selected_exemplar_budget
    ):
        raise CorridorClaimError(
            "global supply exhausted before the required full budget"
        )
    return result


def validate_corridor_global_claim_result(
    result: CorridorGlobalClaimResult,
    *,
    leaderboards: CorridorLeaderboardArtifact | None = None,
    coverage_plan: CorridorCoveragePlan | None = None,
    global_boards: ExistingGlobalBoardInput | None = None,
) -> CorridorClaimValidationResult:
    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(result, CorridorGlobalClaimResult):
        return CorridorClaimValidationResult(
            status="fail", blockers=("result must be CorridorGlobalClaimResult",)
        )
    source = result.source_provenance
    source_production = source.get("production_grade")
    if not isinstance(source_production, bool):
        blockers.append("claim source production_grade must be boolean")
        source_production = False
    if not source_production:
        if result.policy.allow_nonproduction_sources:
            warnings.append(
                "claim result uses explicitly allowed non-production sources"
            )
        else:
            blockers.append("claim result has non-production sources without override")
    if result.policy.claim_policy_id != CLAIM_POLICY_ID:
        blockers.append("unsupported claim policy")
    corridor_coordinates = [claim.coordinate for claim in result.corridor_claims]
    global_coordinates = [claim.coordinate for claim in result.global_claims]
    selected_coordinates = [
        (item.example_id, item.position) for item in result.selected_coordinates
    ]
    if len(selected_coordinates) != len(set(selected_coordinates)):
        blockers.append("selected coordinates contain duplicates")
    if set(corridor_coordinates) & set(global_coordinates):
        blockers.append("corridor and global unique claims overlap")
    if len(selected_coordinates) != len(corridor_coordinates) + len(global_coordinates):
        blockers.append(
            "selected coordinates do not equal corridor union global claims"
        )
    if len(result.selected_coordinates) != result.policy.total_selected_exemplar_budget:
        if result.policy.require_full_budget:
            blockers.append("required full budget is not filled")
        elif (
            len(result.selected_coordinates)
            > result.policy.total_selected_exemplar_budget
        ):
            blockers.append("underfilled result contains too many coordinates")
    if result.summary is not None and dict(result.summary) != _claim_summary(result):
        blockers.append("claim summary arithmetic is inconsistent")
    _validate_claim_order(result, blockers)
    _validate_obligations(result, blockers)
    _validate_backfill_lineage(result, blockers)
    if leaderboards is not None and coverage_plan is not None:
        try:
            _validate_c4_sources(
                leaderboards,
                coverage_plan,
                global_boards or ExistingGlobalBoardInput(boards=()),
                result.policy,
            )
            expected = claim_corridor_then_backfill_global(
                leaderboards,
                coverage_plan,
                global_boards or ExistingGlobalBoardInput(boards=()),
                result.policy,
                _validate_result=False,
            )
            if result.to_dict() != expected.to_dict():
                blockers.append(
                    "claim result does not match deterministic source allocation"
                )
        except (
            CorridorClaimError,
            CorridorBudgetError,
            CorridorLeaderboardError,
        ) as exc:
            blockers.append(str(exc))
    status: Literal["pass", "fail", "warn"] = "fail" if blockers else "pass"
    if status == "pass" and warnings:
        status = "warn"
    return CorridorClaimValidationResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=_claim_summary(result),
    )


def write_corridor_global_claim_result(
    result: CorridorGlobalClaimResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise ValueError(f"claim output exists: {output}")
    validation = validate_corridor_global_claim_result(result)
    if validation.status == "fail":
        raise CorridorClaimError(
            "cannot write invalid claim result: " + "; ".join(validation.blockers)
        )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        files = {
            CORRIDOR_CLAIMS_FILENAME: [
                claim.to_dict() for claim in result.corridor_claims
            ],
            GLOBAL_CLAIMS_FILENAME: [claim.to_dict() for claim in result.global_claims],
            COLLISION_OBLIGATIONS_FILENAME: [
                collision.to_dict() for collision in result.collision_obligations
            ],
            SELECTED_COORDINATES_FILENAME: [
                coordinate.to_dict() for coordinate in result.selected_coordinates
            ],
        }
        file_hashes: dict[str, str] = {}
        for filename, rows in files.items():
            path = temporary / filename
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            file_hashes[filename] = _sha256(path)
        lineage_path = temporary / "backfill_lineage.jsonl"
        lineage_path.write_text(
            "".join(
                json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for item in result.backfill_lineage
            ),
            encoding="utf-8",
        )
        file_hashes["backfill_lineage.jsonl"] = _sha256(lineage_path)
        manifest = {
            "schema_version": CORRIDOR_CLAIM_SCHEMA,
            "claim_policy": result.policy.to_dict(),
            "source_provenance": dict(result.source_provenance),
            "summary": dict(result.summary or _claim_summary(result)),
            "files": {
                filename: {"sha256": digest}
                for filename, digest in sorted(file_hashes.items())
            },
        }
        write_json(temporary / CLAIM_MANIFEST_FILENAME, manifest)
        report = validation.to_dict()
        report["file_hashes"] = file_hashes
        write_json(temporary / CLAIM_VALIDATION_FILENAME, report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def validate_corridor_global_claim_artifact(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> CorridorClaimValidationResult:
    root = Path(path)
    try:
        manifest = read_json_object(root / CLAIM_MANIFEST_FILENAME)
        report = read_json_object(root / CLAIM_VALIDATION_FILENAME)
        if manifest.get("schema_version") != CORRIDOR_CLAIM_SCHEMA:
            raise CorridorClaimError("unsupported claim artifact schema")
        result = _result_from_artifact(root, manifest)
        validation = validate_corridor_global_claim_result(
            result,  # Source-object comparison is performed by callers when available.
        )
        blockers = list(validation.blockers)
        if not production_grade and not result.production_grade:
            blockers = [item for item in blockers if "without override" not in item]
        if production_grade and not result.production_grade:
            blockers.append("claim artifact is non-production")
        for filename, info in (manifest.get("files") or {}).items():
            file_path = root / filename
            if not file_path.is_file() or info.get("sha256") != _sha256(file_path):
                blockers.append(f"claim file hash mismatch: {filename}")
        if report.get("schema_version") != CORRIDOR_CLAIM_VALIDATION_SCHEMA:
            blockers.append("unsupported claim validation schema")
        status: Literal["pass", "fail", "warn"] = (
            "fail" if blockers else validation.status
        )
        return CorridorClaimValidationResult(
            status=status,
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=validation.warnings,
            summary=validation.summary,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return CorridorClaimValidationResult(
            status="fail", blockers=(f"claim artifact unreadable: {exc}",)
        )


def load_corridor_global_claim_result(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> CorridorGlobalClaimResult:
    """Load a hash-validated C4 claim result for downstream offline stages."""

    root = Path(path)
    validation = validate_corridor_global_claim_artifact(
        root,
        production_grade=production_grade,
    )
    if validation.status == "fail":
        raise CorridorClaimError(
            "cannot load invalid corridor claim artifact: "
            + "; ".join(validation.blockers)
        )
    manifest = read_json_object(root / CLAIM_MANIFEST_FILENAME)
    return _result_from_artifact(root, manifest)


def inspect_corridor_global_claim_artifact(path: str | Path) -> dict[str, Any]:
    validation = validate_corridor_global_claim_artifact(path, production_grade=False)
    manifest = read_json_object(Path(path) / CLAIM_MANIFEST_FILENAME)
    summary = dict(manifest.get("summary") or {})
    return {
        "status": validation.status,
        "blockers": list(validation.blockers),
        "warnings": list(validation.warnings),
        **summary,
        "source_production_grade": (manifest.get("source_provenance", {}) or {}).get(
            "production_grade"
        ),
    }


def load_global_board_input(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> ExistingGlobalBoardInput:
    """Load the C4 global-board supply contract or an existing selector manifest."""

    source = Path(path)
    payload = read_json_object(source)
    source_hash = _sha256(source)
    if payload.get("schema_version") == "exemplar_selection_manifest_v1":
        boards = []
        for board in payload.get("boards", []):
            board_id = str(board["board_id"])
            winners = board.get("winners", [])
            candidates = tuple(
                GlobalBoardCandidate(
                    example_id=str(item["example_id"]),
                    position=int(item["selected_position"]),
                    rank=index,
                    score=float(item["score"]),
                )
                for index, item in enumerate(winners, start=1)
            )
            boards.append(
                GlobalBoard(
                    board_id=board_id,
                    priority=_existing_board_priority(board_id),
                    requested_slots=int(board.get("capacity", len(candidates))),
                    candidates=candidates,
                )
            )
        result = ExistingGlobalBoardInput(
            boards=tuple(boards),
            source_provenance={
                "source_artifact_id": str(source.resolve()),
                "source_artifact_hash": source_hash,
                "production_global_selector": payload.get(
                    "production_global_selector", False
                ),
                "production_grade": False,
            },
        )
        if production_grade:
            raise CorridorClaimError(
                "existing selector manifest is non-production; provide the C4 override"
            )
        return result
    if payload.get("schema_version") != GLOBAL_BOARD_SUPPLY_SCHEMA:
        raise CorridorClaimError("unsupported global board supply schema")
    boards = tuple(_global_board_from_dict(item) for item in payload.get("boards", ()))
    provenance = dict(payload.get("source_provenance") or {})
    provenance.setdefault("source_artifact_id", str(source.resolve()))
    provenance.setdefault("source_artifact_hash", source_hash)
    provenance.setdefault("production_grade", False)
    result = ExistingGlobalBoardInput(boards=boards, source_provenance=provenance)
    if production_grade and not bool(provenance.get("production_grade", False)):
        raise CorridorClaimError("global board supply is not production-grade")
    return result


def _normalize_global_input(
    global_boards: ExistingGlobalBoardInput | Mapping[str, Any],
) -> ExistingGlobalBoardInput:
    if isinstance(global_boards, ExistingGlobalBoardInput):
        return global_boards
    if not isinstance(global_boards, Mapping):
        raise TypeError("global_boards must be ExistingGlobalBoardInput or mapping")
    if "boards" in global_boards:
        return ExistingGlobalBoardInput(
            boards=tuple(
                _global_board_from_dict(item) for item in global_boards["boards"]
            ),
            source_provenance=dict(
                global_boards.get("source_provenance")
                or {
                    "source_artifact_id": "in_memory_global",
                    "production_grade": True,
                }
            ),
        )
    boards = []
    for priority, (board_id, value) in enumerate(sorted(global_boards.items())):
        if isinstance(value, Mapping):
            candidates = value.get("candidates", ())
            requested = int(value.get("requested_slots", len(candidates)))
            board_priority = int(value.get("priority", priority))
        else:
            candidates = value
            requested = len(candidates)
            board_priority = priority
        boards.append(
            GlobalBoard(
                board_id=str(board_id),
                priority=board_priority,
                requested_slots=requested,
                candidates=tuple(
                    _global_candidate_from_dict(item, index)
                    for index, item in enumerate(candidates, 1)
                ),
            )
        )
    return ExistingGlobalBoardInput(
        boards=tuple(boards),
        source_provenance={
            "source_artifact_id": "in_memory_global",
            "production_grade": True,
        },
    )


def _global_board_from_dict(payload: Mapping[str, Any]) -> GlobalBoard:
    candidates = payload.get("candidates", ())
    return GlobalBoard(
        board_id=str(payload["board_id"]),
        priority=int(payload.get("priority", 0)),
        requested_slots=int(payload.get("requested_slots", len(candidates))),
        candidates=tuple(
            _global_candidate_from_dict(item, index)
            for index, item in enumerate(candidates, 1)
        ),
    )


def _global_candidate_from_dict(
    payload: Mapping[str, Any], index: int
) -> GlobalBoardCandidate:
    return GlobalBoardCandidate(
        example_id=str(payload["example_id"]),
        position=int(payload["position"]),
        rank=int(payload.get("rank", index)),
        score=float(payload["score"]),
        eligible=bool(payload.get("eligible", True)),
        metadata=dict(payload.get("metadata") or {}),
    )


def _validate_c4_sources(
    leaderboards: CorridorLeaderboardArtifact,
    coverage_plan: CorridorCoveragePlan,
    global_input: ExistingGlobalBoardInput,
    policy: CorridorGlobalClaimPolicy,
) -> None:
    c2_validation = validate_corridor_candidate_leaderboard_artifact(
        leaderboards, production_grade=leaderboards.production_grade
    )
    if c2_validation.status == "fail":
        raise CorridorClaimError(
            "C2 validation failed: " + "; ".join(c2_validation.blockers)
        )
    c3_validation = validate_corridor_coverage_plan_object(
        coverage_plan,
        production_grade=coverage_plan.production_grade,
        leaderboards=leaderboards,
    )
    if c3_validation.status == "fail":
        raise CorridorClaimError(
            "C3 validation failed: " + "; ".join(c3_validation.blockers)
        )
    if (
        policy.total_selected_exemplar_budget
        != coverage_plan.policy.total_selected_exemplar_budget
    ):
        raise CorridorClaimError("C4 total budget disagrees with C3")
    c2_provenance = leaderboards.feature_provenance
    source = coverage_plan.source_leaderboard_provenance
    if source.get("c2_policy_id") != leaderboards.policy.policy_id:
        raise CorridorClaimError("C3 source policy does not identify supplied C2")
    if c2_provenance is not None:
        if source.get("source_artifact_id") != c2_provenance.source_artifact_id:
            raise CorridorClaimError("C3 source artifact identity does not match C2")
        if source.get("source_artifact_hash") != c2_provenance.source_artifact_hash:
            raise CorridorClaimError("C3 source artifact hash does not match C2")
        if source.get("feature_fidelity") != c2_provenance.fidelity:
            raise CorridorClaimError("C3 feature fidelity does not match C2")
    if not leaderboards.production_grade or not coverage_plan.production_grade:
        if not policy.allow_nonproduction_sources:
            raise CorridorClaimError(
                "non-production C2/C3 source requires explicit override"
            )
    global_production = global_input.source_provenance.get("production_grade")
    if not isinstance(global_production, bool):
        raise CorridorClaimError("global source production_grade must be boolean")
    if not global_production and not policy.allow_nonproduction_sources:
        raise CorridorClaimError(
            "non-production global source requires explicit override"
        )


def _reject_cross_corridor_coordinates(
    leaderboards: CorridorLeaderboardArtifact,
) -> None:
    owners: dict[tuple[str, int], int] = {}
    for mode in leaderboards.modes:
        for candidate in mode.candidates:
            coordinate = (candidate.candidate_id, candidate.position)
            previous = owners.get(coordinate)
            if previous is not None and previous != mode.corridor_mode_id:
                raise CorridorClaimError(
                    "coordinate appears in multiple corridor pools: "
                    f"{coordinate[0]}:{coordinate[1]}"
                )
            owners[coordinate] = mode.corridor_mode_id


def _claim_source_provenance(
    leaderboards: CorridorLeaderboardArtifact,
    coverage_plan: CorridorCoveragePlan,
    global_input: ExistingGlobalBoardInput,
) -> dict[str, Any]:
    return {
        "production_grade": bool(
            leaderboards.production_grade
            and coverage_plan.production_grade
            and global_input.source_provenance.get("production_grade")
        ),
        "c2": {
            "policy_id": leaderboards.policy.policy_id,
            "feature_fidelity": (
                leaderboards.feature_provenance.fidelity
                if leaderboards.feature_provenance
                else "none"
            ),
            "source_artifact_id": (
                leaderboards.feature_provenance.source_artifact_id
                if leaderboards.feature_provenance
                else None
            ),
            "source_artifact_hash": (
                leaderboards.feature_provenance.source_artifact_hash
                if leaderboards.feature_provenance
                else None
            ),
        },
        "c3": dict(coverage_plan.source_leaderboard_provenance),
        "global": dict(global_input.source_provenance),
    }


def _validate_claim_order(
    result: CorridorGlobalClaimResult, blockers: list[str]
) -> None:
    corridor_order = [
        (claim.corridor_mode_id, claim.c2_rank) for claim in result.corridor_claims
    ]
    if corridor_order != sorted(corridor_order):
        blockers.append("corridor claims are not in mode/rank order")
    previous_by_board: dict[str, int] = {}
    for claim in result.global_claims:
        previous = previous_by_board.get(claim.board_id, 0)
        if claim.global_rank <= previous:
            blockers.append(
                f"global board {claim.board_id} claims are not rank ordered"
            )
        previous_by_board[claim.board_id] = claim.global_rank


def _validate_obligations(
    result: CorridorGlobalClaimResult, blockers: list[str]
) -> None:
    selected = {
        (item.example_id, item.position): item for item in result.selected_coordinates
    }
    for collision in result.collision_obligations:
        item = selected.get(collision.coordinate)
        if item is None:
            blockers.append("collision obligation references an unselected coordinate")
        elif collision.obligation not in item.obligations:
            blockers.append(
                "collision obligation is missing from coordinate obligations"
            )
    for item in result.selected_coordinates:
        if not item.obligations:
            blockers.append("selected coordinate has no obligations")
        _reject_payload_fields(
            {"obligations": [ob.to_dict() for ob in item.obligations]}
        )


def _validate_backfill_lineage(
    result: CorridorGlobalClaimResult, blockers: list[str]
) -> None:
    """Ensure every skipped event has at most one replacement seat."""

    global_claims = {
        (claim.board_id, claim.global_rank): claim.coordinate
        for claim in result.global_claims
    }
    seen_skips: set[tuple[str, int]] = set()
    seen_replacements: set[tuple[str, int]] = set()
    for lineage in result.backfill_lineage:
        skip_key = (lineage.board_id, lineage.skipped_rank)
        if skip_key in seen_skips:
            blockers.append(
                "backfill lineage repeats a skipped board rank: "
                f"{lineage.board_id}:{lineage.skipped_rank}"
            )
        seen_skips.add(skip_key)
        if lineage.replacement_rank is None:
            if lineage.replacement_coordinate is not None:
                blockers.append("unresolved lineage has a replacement coordinate")
            continue
        replacement_key = (lineage.board_id, lineage.replacement_rank)
        if replacement_key in seen_replacements:
            blockers.append(
                "backfill lineage maps multiple skipped events to one replacement: "
                f"{lineage.board_id}:{lineage.replacement_rank}"
            )
        seen_replacements.add(replacement_key)
        selected_coordinate = global_claims.get(replacement_key)
        if selected_coordinate is None:
            blockers.append(
                "backfill lineage replacement is not a selected global claim: "
                f"{lineage.board_id}:{lineage.replacement_rank}"
            )
        elif lineage.replacement_coordinate != selected_coordinate:
            blockers.append(
                "backfill lineage replacement coordinate does not match its claim: "
                f"{lineage.board_id}:{lineage.replacement_rank}"
            )


def _claim_summary(result: CorridorGlobalClaimResult) -> dict[str, Any]:
    corridor_count = len(result.corridor_claims)
    global_count = len(result.global_claims)
    selected_count = len(result.selected_coordinates)
    return {
        "requested_total_budget": result.policy.total_selected_exemplar_budget,
        "corridor_budget": corridor_count,
        "global_budget": result.policy.total_selected_exemplar_budget - corridor_count,
        "corridor_claim_count": corridor_count,
        "global_unique_claim_count": global_count,
        "unique_selected_count": selected_count,
        "unfilled_slots": max(
            0, result.policy.total_selected_exemplar_budget - selected_count
        ),
        "collision_count": len(result.collision_obligations),
        "backfill_lineage_count": len(result.backfill_lineage),
        "coordinates_with_multiple_obligations": sum(
            len(item.obligations) > 1 for item in result.selected_coordinates
        ),
        "underfilled": selected_count < result.policy.total_selected_exemplar_budget,
        "board_summaries": result.summary.get("board_summaries", [])
        if result.summary
        else [],
    }


def _result_from_artifact(
    root: Path, manifest: Mapping[str, Any]
) -> CorridorGlobalClaimResult:
    policy_payload = manifest["claim_policy"]
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=policy_payload["total_selected_exemplar_budget"],
        require_full_budget=policy_payload["require_full_budget"],
        claim_policy_id=policy_payload["claim_policy_id"],
        allow_nonproduction_sources=policy_payload.get(
            "allow_nonproduction_sources", False
        ),
        nonproduction_override_reason=policy_payload.get(
            "nonproduction_override_reason"
        ),
    )
    corridor = tuple(
        _corridor_claim_from_dict(item)
        for item in _read_jsonl(root / CORRIDOR_CLAIMS_FILENAME)
    )
    global_claims = tuple(
        _global_claim_from_dict(item)
        for item in _read_jsonl(root / GLOBAL_CLAIMS_FILENAME)
    )
    collisions = tuple(
        _collision_from_dict(item)
        for item in _read_jsonl(root / COLLISION_OBLIGATIONS_FILENAME)
    )
    selected = tuple(
        _selected_coordinate_from_dict(item)
        for item in _read_jsonl(root / SELECTED_COORDINATES_FILENAME)
    )
    lineage = tuple(
        _lineage_from_dict(item)
        for item in _read_jsonl(root / "backfill_lineage.jsonl")
    )
    return CorridorGlobalClaimResult(
        policy=policy,
        source_provenance=dict(manifest.get("source_provenance") or {}),
        corridor_claims=corridor,
        global_claims=global_claims,
        collision_obligations=collisions,
        backfill_lineage=lineage,
        selected_coordinates=selected,
        summary=dict(manifest.get("summary") or {}),
    )


def _obligation_from_dict(payload: Mapping[str, Any]) -> SelectionObligation:
    return SelectionObligation(
        role=payload["role"],
        source_id=str(payload["source_id"]),
        rank=payload["rank"],
        score=payload.get("score"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _corridor_claim_from_dict(payload: Mapping[str, Any]) -> CorridorClaim:
    return CorridorClaim(
        example_id=str(payload["example_id"]),
        position=payload["position"],
        corridor_mode_id=payload["corridor_mode_id"],
        c2_rank=payload["c2_rank"],
        score=payload["score"],
        score_components=dict(payload["score_components"]),
        obligation=_obligation_from_dict(payload["obligation"]),
    )


def _global_claim_from_dict(payload: Mapping[str, Any]) -> GlobalClaim:
    return GlobalClaim(
        example_id=str(payload["example_id"]),
        position=payload["position"],
        board_id=str(payload["board_id"]),
        board_priority=payload["board_priority"],
        global_rank=payload["global_rank"],
        score=payload["score"],
        backfilled=payload["backfilled"],
        obligation=_obligation_from_dict(payload["obligation"]),
    )


def _collision_from_dict(payload: Mapping[str, Any]) -> CollisionObligation:
    return CollisionObligation(
        example_id=str(payload["example_id"]),
        position=payload["position"],
        board_id=str(payload["board_id"]),
        board_priority=payload["board_priority"],
        global_rank=payload["global_rank"],
        score=payload["score"],
        collision_kind=payload["collision_kind"],
        occupied_primary_claim=payload["occupied_primary_claim"],
        obligation=_obligation_from_dict(payload["obligation"]),
    )


def _selected_coordinate_from_dict(payload: Mapping[str, Any]) -> ClaimedCoordinate:
    return ClaimedCoordinate(
        example_id=str(payload["example_id"]),
        position=payload["position"],
        primary_claim=payload["primary_claim"],
        claim_order=payload["claim_order"],
        obligations=tuple(
            _obligation_from_dict(item) for item in payload["obligations"]
        ),
    )


def _lineage_from_dict(payload: Mapping[str, Any]) -> BackfillLineage:
    skipped = (str(payload["skipped_example_id"]), payload["skipped_position"])
    replacement = (
        None
        if payload.get("replacement_example_id") is None
        else (str(payload["replacement_example_id"]), payload["replacement_position"])
    )
    return BackfillLineage(
        board_id=str(payload["board_id"]),
        skipped_rank=payload["skipped_rank"],
        skipped_coordinate=skipped,
        reason=payload["reason"],
        replacement_rank=payload.get("replacement_rank"),
        replacement_coordinate=replacement,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CorridorClaimError(f"claim file missing: {path.name}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise CorridorClaimError(
                    f"claim JSONL row is not an object: {path.name}"
                )
            rows.append(payload)
    return rows


def _existing_board_priority(board_id: str) -> int:
    prefixes = (
        "global_max_entropy",
        "low_confidence",
        "high_tail_mass",
        "high_effective_top_k",
        "global_mean_entropy",
        "position_bucket_entropy",
        "length_bucket_entropy",
        "shard_coverage",
    )
    for index, prefix in enumerate(prefixes):
        if board_id == prefix or board_id.startswith(prefix + ":"):
            return index
    return len(prefixes)


def _reject_payload_fields(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if str(key).lower() in FORBIDDEN_PAYLOAD_KEYS:
            raise ValueError(f"claim input contains forbidden payload field: {key}")
        if isinstance(value, Mapping):
            _reject_payload_fields(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    _reject_payload_fields(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
