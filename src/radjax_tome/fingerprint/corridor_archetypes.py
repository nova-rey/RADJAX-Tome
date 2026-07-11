"""C1 corridor-core eligibility and archetype scoring.

This module deliberately stops at a pure candidate-level primitive. It does not
construct candidate pools, choose budgets, or alter production artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

CORRIDOR_ARCHETYPE_POLICY_SCHEMA = "radjax.c1_corridor_archetype_policy.v1"
CORRIDOR_ARCHETYPE_SCORE_SCHEMA = "radjax.c1_corridor_archetype_score.v1"
CORRIDOR_ARCHETYPE_POLICY_ID = "corridor_archetype_v1"

LINKED_ASSIGNMENT_STATUSES = frozenset({"linked", "assigned"})
UNASSIGNED_STATUSES = frozenset({"missing", "unassigned", "failed", "invalid"})


@dataclass(frozen=True)
class CorridorArchetypePolicy:
    """Validated thresholds and weights for one corridor archetype score."""

    policy_id: str = CORRIDOR_ARCHETYPE_POLICY_ID
    minimum_membership_strength: float = 0.5
    maximum_core_distance: float = 0.5
    minimum_mode_support: int = 1
    membership_weight: float = 0.45
    centrality_weight: float = 0.40
    difficulty_weight: float = 0.15
    quality_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be nonempty")
        _unit_interval(self.minimum_membership_strength, "minimum_membership_strength")
        _unit_interval(self.maximum_core_distance, "maximum_core_distance")
        if isinstance(self.minimum_mode_support, bool) or self.minimum_mode_support < 0:
            raise ValueError("minimum_mode_support must be a nonnegative integer")
        if not isinstance(self.minimum_mode_support, int):
            raise ValueError("minimum_mode_support must be a nonnegative integer")
        weights = (
            self.membership_weight,
            self.centrality_weight,
            self.difficulty_weight,
            self.quality_weight,
        )
        if any(not math.isfinite(float(weight)) or weight < 0.0 for weight in weights):
            raise ValueError("policy weights must be finite and nonnegative")
        if sum(weights) <= 0.0:
            raise ValueError("policy weights must have a positive total")
        if self.difficulty_weight > self.membership_weight:
            raise ValueError("difficulty_weight must not exceed membership_weight")
        if self.difficulty_weight > self.centrality_weight:
            raise ValueError("difficulty_weight must not exceed centrality_weight")

    @property
    def normalized_weights(self) -> Mapping[str, float]:
        total = sum(
            (
                self.membership_weight,
                self.centrality_weight,
                self.difficulty_weight,
                self.quality_weight,
            )
        )
        return MappingProxyType(
            {
                "membership": self.membership_weight / total,
                "centrality": self.centrality_weight / total,
                "difficulty": self.difficulty_weight / total,
                "quality": self.quality_weight / total,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_ARCHETYPE_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "minimum_membership_strength": self.minimum_membership_strength,
            "maximum_core_distance": self.maximum_core_distance,
            "minimum_mode_support": self.minimum_mode_support,
            "membership_weight": self.membership_weight,
            "centrality_weight": self.centrality_weight,
            "difficulty_weight": self.difficulty_weight,
            "quality_weight": self.quality_weight,
        }


@dataclass(frozen=True)
class CorridorCandidateFeatures:
    """Compact, already-computed features for one assigned position."""

    candidate_id: str
    position: int
    corridor_mode_id: int | None
    assignment_status: str
    membership_strength: float
    core_distance: float
    mode_support: int
    difficulty_score: float
    quality_score: float | None = None
    corridor_fingerprint_id: str | None = None
    position_valid: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not isinstance(self.position, int) or isinstance(self.position, bool):
            raise ValueError("position must be an integer")
        if self.corridor_mode_id is not None and (
            isinstance(self.corridor_mode_id, bool)
            or not isinstance(self.corridor_mode_id, int)
        ):
            raise ValueError("corridor_mode_id must be an integer or None")
        if isinstance(self.mode_support, bool) or not isinstance(
            self.mode_support, int
        ):
            raise ValueError("mode_support must be a nonnegative integer")
        if self.mode_support < 0:
            raise ValueError("mode_support must be a nonnegative integer")
        if not isinstance(self.assignment_status, str) or not self.assignment_status:
            raise ValueError("assignment_status must be nonempty")
        if not isinstance(self.position_valid, bool):
            raise TypeError("position_valid must be a boolean")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CorridorCandidateFeatures:
        """Map current corridor/selected fields onto the C1 feature contract.

        Current selected records do not yet expose explicit membership or core
        distance. For those records, a linked assignment is the documented C1
        proxy for full membership, zero core distance, and one supported mode.
        """

        status = str(payload.get("corridor_assignment_status", "missing"))
        linked = status in LINKED_ASSIGNMENT_STATUSES
        return cls(
            candidate_id=str(
                payload.get("candidate_id")
                or payload.get("selected_example_id")
                or payload.get("example_id")
                or ""
            ),
            position=int(payload.get("position", payload.get("selected_position", -1))),
            corridor_mode_id=_strict_optional_int(
                payload.get("corridor_mode_id", payload.get("mode_id")),
                "corridor_mode_id",
            ),
            assignment_status=status,
            membership_strength=float(
                payload.get("membership_strength", 1.0 if linked else 0.0)
            ),
            core_distance=float(payload.get("core_distance", 0.0 if linked else 1.0)),
            mode_support=_strict_int(
                payload.get("mode_support", 1 if linked else 0),
                "mode_support",
            ),
            difficulty_score=float(
                payload.get(
                    "difficulty_score",
                    payload.get("useful_difficulty_score", 0.0),
                )
            ),
            quality_score=(
                None
                if payload.get("quality_score") is None
                else float(payload["quality_score"])
            ),
            corridor_fingerprint_id=(
                None
                if payload.get("corridor_fingerprint_id") is None
                else str(payload["corridor_fingerprint_id"])
            ),
            position_valid=_strict_bool(
                payload.get("position_valid", True),
                "position_valid",
            ),
        )


@dataclass(frozen=True)
class CorridorArchetypeScore:
    """Deterministic eligibility result and bounded ranking components."""

    candidate_id: str
    position: int
    corridor_mode_id: int | None
    corridor_fingerprint_id: str | None
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    membership_score: float
    centrality_score: float
    useful_difficulty_score: float
    quality_score: float
    corridor_training_utility: float | None
    policy_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_ARCHETYPE_SCORE_SCHEMA,
            "candidate_id": self.candidate_id,
            "position": self.position,
            "corridor_mode_id": self.corridor_mode_id,
            "corridor_fingerprint_id": self.corridor_fingerprint_id,
            "eligible": self.eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "membership_score": self.membership_score,
            "centrality_score": self.centrality_score,
            "useful_difficulty_score": self.useful_difficulty_score,
            "quality_score": self.quality_score,
            "corridor_training_utility": self.corridor_training_utility,
            "policy_id": self.policy_id,
        }


def score_corridor_archetype_candidate(
    features: CorridorCandidateFeatures,
    policy: CorridorArchetypePolicy,
) -> CorridorArchetypeScore:
    """Evaluate eligibility first, then score an eligible corridor archetype."""

    if not isinstance(features, CorridorCandidateFeatures):
        raise TypeError("features must be CorridorCandidateFeatures")
    if not isinstance(policy, CorridorArchetypePolicy):
        raise TypeError("policy must be CorridorArchetypePolicy")

    membership_score = _safe_unit(features.membership_strength)
    centrality_score = _centrality(features.core_distance, policy)
    difficulty_score = _safe_unit(features.difficulty_score)
    quality_score = _safe_unit(features.quality_score)
    reasons: list[str] = []

    if not features.position_valid or features.position < 0:
        reasons.append("invalid_position")
    if features.corridor_mode_id is None or features.corridor_mode_id < 0:
        reasons.append("unassigned_corridor")
    elif features.assignment_status in UNASSIGNED_STATUSES:
        reasons.append("unassigned_corridor")
    elif features.assignment_status not in LINKED_ASSIGNMENT_STATUSES:
        reasons.append("invalid_assignment_status")

    numeric_values = (
        features.membership_strength,
        features.core_distance,
        features.mode_support,
        features.difficulty_score,
    )
    if features.quality_score is not None:
        numeric_values += (features.quality_score,)
    if any(not _finite(value) for value in numeric_values):
        reasons.append("nonfinite_feature")
    if any(
        _finite(value) and not 0.0 <= float(value) <= 1.0
        for value in (
            features.membership_strength,
            features.core_distance,
            features.difficulty_score,
        )
    ) or (
        features.quality_score is not None
        and _finite(features.quality_score)
        and not 0.0 <= float(features.quality_score) <= 1.0
    ):
        reasons.append("feature_out_of_range")
    if (
        _finite(features.membership_strength)
        and features.membership_strength < policy.minimum_membership_strength
    ):
        reasons.append("membership_below_minimum")
    if (
        _finite(features.core_distance)
        and features.core_distance > policy.maximum_core_distance
    ):
        reasons.append("outside_corridor_core")
    if features.mode_support < policy.minimum_mode_support:
        reasons.append("mode_support_below_minimum")

    ordered_reasons = tuple(dict.fromkeys(reasons))
    eligible = not ordered_reasons
    utility = None
    if eligible:
        weights = policy.normalized_weights
        utility = _bounded(
            weights["membership"] * membership_score
            + weights["centrality"] * centrality_score
            + weights["difficulty"] * difficulty_score
            + weights["quality"] * quality_score
        )
    return CorridorArchetypeScore(
        candidate_id=features.candidate_id,
        position=features.position,
        corridor_mode_id=features.corridor_mode_id,
        corridor_fingerprint_id=features.corridor_fingerprint_id,
        eligible=eligible,
        eligibility_reasons=ordered_reasons,
        membership_score=membership_score,
        centrality_score=centrality_score,
        useful_difficulty_score=difficulty_score,
        quality_score=quality_score,
        corridor_training_utility=utility,
        policy_id=policy.policy_id,
    )


def _centrality(core_distance: float, policy: CorridorArchetypePolicy) -> float:
    if not _finite(core_distance) or core_distance < 0.0:
        return 0.0
    if policy.maximum_core_distance == 0.0:
        return 1.0 if core_distance == 0.0 else 0.0
    return _bounded(1.0 - core_distance / policy.maximum_core_distance)


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_unit(value: float | None) -> float:
    if value is None or not _finite(value):
        return 0.0
    return _bounded(float(value))


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _strict_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or None")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be a boolean or explicit true/false string")
