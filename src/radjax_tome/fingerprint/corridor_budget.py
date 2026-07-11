"""C3 bounded offline fingerprint-corridor budget allocation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardArtifact,
    CorridorLeaderboardError,
    CorridorModeLeaderboard,
    validate_corridor_candidate_leaderboard_artifact,
)
from radjax_tome.io.json import read_json_object, write_json

CORRIDOR_BUDGET_SCHEMA = "radjax.c3_corridor_coverage_plan.v1"
CORRIDOR_BUDGET_VALIDATION_SCHEMA = "radjax.c3_corridor_coverage_validation.v1"
CORRIDOR_BUDGET_POLICY_ID = "breadth_first_round_robin_v1"
COVERAGE_PLAN_FILENAME = "coverage_plan.json"
COVERAGE_VALIDATION_FILENAME = "validation_report.json"
ZERO_ALLOCATION_REASONS = frozenset(
    {
        "no_eligible_candidates",
        "empty_candidate_pool",
        "corridor_budget_exhausted",
        "mode_capacity_zero",
    }
)


class CorridorBudgetError(ValueError):
    """Actionable C3 policy, allocation, or artifact error."""


@dataclass(frozen=True)
class CorridorBudgetPolicy:
    total_selected_exemplar_budget: int
    corridor_budget_fraction: Decimal | float | str = Decimal("0.50")
    corridor_budget_max: int | None = None
    corridor_mode_cap: int = 10
    allocation_policy: str = CORRIDOR_BUDGET_POLICY_ID
    allow_nonproduction_leaderboards: bool = False
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
        fraction = _decimal_fraction(self.corridor_budget_fraction)
        object.__setattr__(self, "corridor_budget_fraction", fraction)
        if self.corridor_budget_max is not None and (
            isinstance(self.corridor_budget_max, bool)
            or not isinstance(self.corridor_budget_max, int)
        ):
            raise ValueError(
                "corridor_budget_max must be a nonnegative integer or None"
            )
        if self.corridor_budget_max is not None and self.corridor_budget_max < 0:
            raise ValueError("corridor_budget_max must be nonnegative")
        if isinstance(self.corridor_mode_cap, bool) or not isinstance(
            self.corridor_mode_cap, int
        ):
            raise ValueError("corridor_mode_cap must be a positive integer")
        if self.corridor_mode_cap < 1:
            raise ValueError("corridor_mode_cap must be a positive integer")
        if not isinstance(self.allocation_policy, str) or not self.allocation_policy:
            raise ValueError("allocation_policy must be nonempty")
        if self.allocation_policy != CORRIDOR_BUDGET_POLICY_ID:
            raise ValueError(f"unsupported allocation_policy: {self.allocation_policy}")
        if not isinstance(self.allow_nonproduction_leaderboards, bool):
            raise TypeError("allow_nonproduction_leaderboards must be a boolean")
        if (
            self.allow_nonproduction_leaderboards
            and not self.nonproduction_override_reason
        ):
            raise ValueError(
                "nonproduction_override_reason is required when nonproduction "
                "leaderboards are allowed"
            )

    @property
    def fractional_ceiling(self) -> int:
        return int(
            (
                Decimal(self.total_selected_exemplar_budget) * self.fraction
            ).to_integral_value(rounding=ROUND_FLOOR)
        )

    @property
    def fraction(self) -> Decimal:
        return self.corridor_budget_fraction  # type: ignore[return-value]

    @property
    def corridor_budget_ceiling(self) -> int:
        if self.corridor_budget_max is None:
            return self.fractional_ceiling
        return min(self.fractional_ceiling, self.corridor_budget_max)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_BUDGET_SCHEMA,
            "allocation_policy": self.allocation_policy,
            "total_selected_exemplar_budget": self.total_selected_exemplar_budget,
            "corridor_budget_fraction": _decimal_text(self.fraction),
            "corridor_budget_max": self.corridor_budget_max,
            "corridor_mode_cap": self.corridor_mode_cap,
            "allow_nonproduction_leaderboards": self.allow_nonproduction_leaderboards,
            "nonproduction_override_reason": self.nonproduction_override_reason,
        }


@dataclass(frozen=True)
class CorridorModeAllocation:
    corridor_mode_id: int
    allocatable_capacity: int
    allocated_slots: int
    retained_pool_count: int
    eligible_candidate_count: int
    mode_support: int
    candidates_seen: int
    candidates_rejected: int
    zero_allocation_reason: str | None = None
    coverage_priority: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_mode_id": self.corridor_mode_id,
            "allocatable_capacity": self.allocatable_capacity,
            "allocated_slots": self.allocated_slots,
            "retained_pool_count": self.retained_pool_count,
            "eligible_candidate_count": self.eligible_candidate_count,
            "mode_support": self.mode_support,
            "candidates_seen": self.candidates_seen,
            "candidates_rejected": self.candidates_rejected,
            "zero_allocation_reason": self.zero_allocation_reason,
            "coverage_priority": dict(self.coverage_priority),
        }


@dataclass(frozen=True)
class CorridorCoveragePlan:
    policy: CorridorBudgetPolicy
    source_leaderboard_provenance: Mapping[str, Any]
    modes: tuple[CorridorModeAllocation, ...]
    fractional_ceiling: int
    corridor_budget_ceiling: int
    raw_mode_capacity: int
    actual_corridor_budget: int
    global_budget: int
    summary: Mapping[str, Any] | None = None

    @property
    def production_grade(self) -> bool:
        return bool(self.source_leaderboard_provenance.get("production_grade"))

    @property
    def unused_corridor_ceiling(self) -> int:
        return self.corridor_budget_ceiling - self.actual_corridor_budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_BUDGET_SCHEMA,
            "allocation_policy_id": self.policy.allocation_policy,
            "policy": self.policy.to_dict(),
            "source_leaderboard_provenance": dict(self.source_leaderboard_provenance),
            "fractional_ceiling": self.fractional_ceiling,
            "corridor_budget_ceiling": self.corridor_budget_ceiling,
            "raw_mode_capacity": self.raw_mode_capacity,
            "actual_corridor_budget": self.actual_corridor_budget,
            "global_budget": self.global_budget,
            "summary": dict(self.summary or _plan_summary(self)),
            "modes": [mode.to_dict() for mode in self.modes],
        }


@dataclass(frozen=True)
class CorridorBudgetValidationResult:
    status: Literal["pass", "fail", "warn"]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_BUDGET_VALIDATION_SCHEMA,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


def allocate_corridor_coverage(
    leaderboards: CorridorLeaderboardArtifact,
    policy: CorridorBudgetPolicy,
    *,
    source_leaderboard_provenance: Mapping[str, Any] | None = None,
) -> CorridorCoveragePlan:
    """Allocate bounded corridor slots without claiming candidate coordinates."""

    if not isinstance(leaderboards, CorridorLeaderboardArtifact):
        raise TypeError("leaderboards must be a CorridorLeaderboardArtifact")
    if not isinstance(policy, CorridorBudgetPolicy):
        raise TypeError("policy must be a CorridorBudgetPolicy")
    source = _default_source_provenance(leaderboards)
    if source_leaderboard_provenance is not None:
        source.update(dict(source_leaderboard_provenance))
    if source.get("production_grade") != leaderboards.production_grade:
        raise CorridorBudgetError(
            "source leaderboard production_grade does not match C2 artifact"
        )
    c2_validation = validate_corridor_candidate_leaderboard_artifact(
        leaderboards,
        production_grade=leaderboards.production_grade,
    )
    if c2_validation.status == "fail":
        raise CorridorBudgetError(
            "C2 leaderboard validation failed: " + "; ".join(c2_validation.blockers)
        )
    if (
        not leaderboards.production_grade
        and not policy.allow_nonproduction_leaderboards
    ):
        raise CorridorBudgetError(
            "non-production C2 leaderboards are disabled; provide an explicit "
            "nonproduction override"
        )

    mode_info = [
        _mode_info(mode, policy.corridor_mode_cap) for mode in leaderboards.modes
    ]
    capacities = {item["mode_id"]: item["capacity"] for item in mode_info}
    priorities = {item["mode_id"]: item["priority"] for item in mode_info}
    raw_capacity = sum(capacities.values())
    corridor_ceiling = policy.corridor_budget_ceiling
    actual_budget = min(corridor_ceiling, raw_capacity)
    allocations = _allocate_counts(
        mode_info,
        actual_budget,
    )
    modes = tuple(
        CorridorModeAllocation(
            corridor_mode_id=item["mode_id"],
            allocatable_capacity=item["capacity"],
            allocated_slots=allocations[item["mode_id"]],
            retained_pool_count=item["retained"],
            eligible_candidate_count=item["eligible"],
            mode_support=item["mode_support"],
            candidates_seen=item["seen"],
            candidates_rejected=item["rejected"],
            zero_allocation_reason=_zero_reason(item, allocations[item["mode_id"]]),
            coverage_priority=priorities[item["mode_id"]],
        )
        for item in mode_info
    )
    return CorridorCoveragePlan(
        policy=policy,
        source_leaderboard_provenance=source,
        modes=modes,
        fractional_ceiling=policy.fractional_ceiling,
        corridor_budget_ceiling=corridor_ceiling,
        raw_mode_capacity=raw_capacity,
        actual_corridor_budget=actual_budget,
        global_budget=policy.total_selected_exemplar_budget - actual_budget,
    )


def validate_corridor_coverage_plan_object(
    plan: CorridorCoveragePlan,
    *,
    production_grade: bool = True,
    leaderboards: CorridorLeaderboardArtifact | None = None,
) -> CorridorBudgetValidationResult:
    """Validate a C3 plan, optionally against its source C2 object."""

    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, CorridorCoveragePlan):
        return CorridorBudgetValidationResult(
            status="fail", blockers=("plan must be a CorridorCoveragePlan",)
        )
    source = plan.source_leaderboard_provenance
    for name in (
        "leaderboard_artifact_id",
        "c2_policy_id",
        "feature_fidelity",
        "production_grade",
    ):
        if name not in source:
            blockers.append(f"source provenance missing {name}")
    source_production = source.get("production_grade")
    if not isinstance(source_production, bool):
        blockers.append("source production_grade must be boolean")
        source_production = False
    if not source_production:
        if production_grade:
            blockers.append("source C2 leaderboard is non-production")
        elif plan.policy.allow_nonproduction_leaderboards:
            warnings.append("plan uses explicitly allowed non-production C2 source")
        else:
            blockers.append("non-production source lacks explicit policy override")
    if source.get("feature_fidelity") == "compatibility_proxy" and source_production:
        blockers.append("compatibility_proxy source cannot be production-grade")
    if (
        source.get("leaderboard_manifest_sha256") is None
        and source.get("leaderboard_artifact_id") != "in_memory_c2"
    ):
        blockers.append("source leaderboard manifest hash is missing")
    if source.get("mode_leaderboards_sha256") is None and plan.modes:
        if source.get("leaderboard_artifact_id") != "in_memory_c2":
            blockers.append("source mode leaderboard hash is missing")
    for hash_name in (
        "source_artifact_hash",
        "leaderboard_manifest_sha256",
        "mode_leaderboards_sha256",
    ):
        hash_value = source.get(hash_name)
        if hash_value is not None and not _is_sha256(hash_value):
            blockers.append(f"source {hash_name} is not a SHA-256 hash")

    if plan.fractional_ceiling != plan.policy.fractional_ceiling:
        blockers.append("fractional ceiling does not match policy")
    if plan.corridor_budget_ceiling != plan.policy.corridor_budget_ceiling:
        blockers.append("corridor budget ceiling does not match policy")
    if plan.actual_corridor_budget != min(
        plan.corridor_budget_ceiling, plan.raw_mode_capacity
    ):
        blockers.append("actual corridor budget does not match ceiling/capacity")
    if plan.global_budget + plan.actual_corridor_budget != (
        plan.policy.total_selected_exemplar_budget
    ):
        blockers.append("global plus corridor budget does not equal total budget")
    if not 0 <= plan.actual_corridor_budget <= plan.corridor_budget_ceiling:
        blockers.append("actual corridor budget is outside its bounds")
    if plan.raw_mode_capacity < 0:
        blockers.append("raw mode capacity must be nonnegative")

    previous_mode = None
    capacities: dict[int, int] = {}
    priorities: dict[int, Mapping[str, float]] = {}
    for mode in plan.modes:
        if previous_mode is not None and mode.corridor_mode_id <= previous_mode:
            blockers.append("serialized modes are not in ascending mode order")
        previous_mode = mode.corridor_mode_id
        if mode.corridor_mode_id in capacities:
            blockers.append("duplicate corridor mode allocation")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                mode.allocatable_capacity,
                mode.allocated_slots,
                mode.retained_pool_count,
                mode.eligible_candidate_count,
                mode.mode_support,
                mode.candidates_seen,
                mode.candidates_rejected,
            )
        ):
            blockers.append(f"mode {mode.corridor_mode_id} has invalid integer fields")
        expected_capacity = min(
            plan.policy.corridor_mode_cap,
            mode.retained_pool_count,
            mode.eligible_candidate_count,
        )
        if mode.allocatable_capacity != expected_capacity:
            blockers.append(f"mode {mode.corridor_mode_id} capacity is inconsistent")
        if mode.allocated_slots > mode.allocatable_capacity:
            blockers.append(f"mode {mode.corridor_mode_id} exceeds capacity")
        if mode.candidates_seen != (
            mode.eligible_candidate_count + mode.candidates_rejected
        ):
            blockers.append(f"mode {mode.corridor_mode_id} count arithmetic is invalid")
        if mode.zero_allocation_reason not in (None, *ZERO_ALLOCATION_REASONS):
            blockers.append(f"mode {mode.corridor_mode_id} has invalid zero reason")
        if mode.allocated_slots == 0:
            expected_reason = _zero_reason(
                {
                    "eligible": mode.eligible_candidate_count,
                    "retained": mode.retained_pool_count,
                    "capacity": mode.allocatable_capacity,
                    "seen": mode.candidates_seen,
                },
                0,
            )
            if mode.zero_allocation_reason != expected_reason:
                blockers.append(f"mode {mode.corridor_mode_id} zero reason is invalid")
        elif mode.zero_allocation_reason is not None:
            blockers.append(
                f"mode {mode.corridor_mode_id} has a reason while allocated"
            )
        if mode.allocatable_capacity > 0:
            for name in (
                "top_candidate_utility",
                "top_candidate_membership",
                "top_candidate_centrality",
            ):
                value = mode.coverage_priority.get(name)
                if not _unit_finite(value):
                    blockers.append(
                        f"mode {mode.corridor_mode_id} priority {name} is invalid"
                    )
        capacities[mode.corridor_mode_id] = mode.allocatable_capacity
        priorities[mode.corridor_mode_id] = mode.coverage_priority

    if sum(capacities.values()) != plan.raw_mode_capacity:
        blockers.append("raw mode capacity total is inconsistent")
    if sum(mode.allocated_slots for mode in plan.modes) != plan.actual_corridor_budget:
        blockers.append("mode allocations do not equal actual corridor budget")
    expected_allocations = _allocate_counts(
        [
            {
                "mode_id": mode_id,
                "capacity": capacity,
                "eligible": next(
                    mode.eligible_candidate_count
                    for mode in plan.modes
                    if mode.corridor_mode_id == mode_id
                ),
                "retained": next(
                    mode.retained_pool_count
                    for mode in plan.modes
                    if mode.corridor_mode_id == mode_id
                ),
                "mode_support": next(
                    mode.mode_support
                    for mode in plan.modes
                    if mode.corridor_mode_id == mode_id
                ),
                "priority": priorities[mode_id],
            }
            for mode_id, capacity in capacities.items()
        ],
        plan.actual_corridor_budget,
    )
    for mode in plan.modes:
        if mode.allocated_slots != expected_allocations[mode.corridor_mode_id]:
            blockers.append("allocation violates breadth-first fairness or priority")

    if leaderboards is not None:
        try:
            source_validation = validate_corridor_candidate_leaderboard_artifact(
                leaderboards, production_grade=leaderboards.production_grade
            )
            if source_validation.status == "fail":
                blockers.extend(source_validation.blockers)
            source_info = {
                item["mode_id"]: item
                for item in (
                    _mode_info(mode, plan.policy.corridor_mode_cap)
                    for mode in leaderboards.modes
                )
            }
            if tuple(sorted(source_info)) != tuple(
                mode.corridor_mode_id for mode in plan.modes
            ):
                blockers.append("plan modes do not match source leaderboard modes")
            for mode in plan.modes:
                item = source_info.get(mode.corridor_mode_id)
                if item is None:
                    continue
                if mode.coverage_priority != item["priority"]:
                    blockers.append(
                        f"mode {mode.corridor_mode_id} priority snapshot mismatch"
                    )
        except (CorridorLeaderboardError, ValueError, TypeError) as exc:
            blockers.append(f"source leaderboard comparison failed: {exc}")

    expected_summary = _plan_summary(plan)
    if plan.summary is not None and dict(plan.summary) != expected_summary:
        blockers.append("plan summary arithmetic is inconsistent")
    status: Literal["pass", "fail", "warn"] = "fail" if blockers else "pass"
    if status == "pass" and warnings:
        status = "warn"
    return CorridorBudgetValidationResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=expected_summary,
    )


def write_corridor_coverage_plan(
    plan: CorridorCoveragePlan,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a compact C3 coverage-plan artifact."""

    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise ValueError(f"coverage plan output exists: {output}")
    validation = validate_corridor_coverage_plan_object(
        plan, production_grade=plan.production_grade
    )
    if validation.status == "fail":
        raise CorridorBudgetError(
            "cannot write invalid corridor coverage plan: "
            + "; ".join(validation.blockers)
        )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        plan_path = temporary / COVERAGE_PLAN_FILENAME
        write_json(plan_path, plan.to_dict())
        report = validation.to_dict()
        report["coverage_plan_sha256"] = _sha256(plan_path)
        write_json(temporary / COVERAGE_VALIDATION_FILENAME, report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def validate_corridor_coverage_plan(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> CorridorBudgetValidationResult:
    """Validate a serialized C3 plan and its content hash."""

    root = Path(path)
    try:
        plan_payload = read_json_object(root / COVERAGE_PLAN_FILENAME)
        report = read_json_object(root / COVERAGE_VALIDATION_FILENAME)
        plan = _plan_from_dict(plan_payload)
        result = validate_corridor_coverage_plan_object(
            plan, production_grade=production_grade
        )
        expected_hash = report.get("coverage_plan_sha256")
        actual_hash = _sha256(root / COVERAGE_PLAN_FILENAME)
        blockers = list(result.blockers)
        if expected_hash != actual_hash:
            blockers.append("coverage_plan.json hash mismatch")
        if report.get("schema_version") != CORRIDOR_BUDGET_VALIDATION_SCHEMA:
            blockers.append("unsupported coverage validation schema")
        status: Literal["pass", "fail", "warn"] = "fail" if blockers else result.status
        return CorridorBudgetValidationResult(
            status=status,
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=result.warnings,
            summary=result.summary,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return CorridorBudgetValidationResult(
            status="fail", blockers=(f"coverage plan unreadable: {exc}",)
        )


def inspect_corridor_coverage_plan(path: str | Path) -> dict[str, Any]:
    """Return a compact human/machine-readable C3 plan summary."""

    plan = _plan_from_dict(read_json_object(Path(path) / COVERAGE_PLAN_FILENAME))
    validation = validate_corridor_coverage_plan(path, production_grade=False)
    covered = [mode.allocated_slots for mode in plan.modes if mode.allocated_slots]
    return {
        "status": validation.status,
        "blockers": list(validation.blockers),
        "warnings": list(validation.warnings),
        "total_selected_exemplar_budget": plan.policy.total_selected_exemplar_budget,
        "corridor_budget_fraction": _decimal_text(plan.policy.fraction),
        "corridor_budget_max": plan.policy.corridor_budget_max,
        "fractional_ceiling": plan.fractional_ceiling,
        "corridor_budget_ceiling": plan.corridor_budget_ceiling,
        "actual_corridor_budget": plan.actual_corridor_budget,
        "global_budget": plan.global_budget,
        "raw_mode_capacity": plan.raw_mode_capacity,
        "modes_observed": len(plan.modes),
        "modes_capacity_positive": sum(
            mode.allocatable_capacity > 0 for mode in plan.modes
        ),
        "modes_covered": sum(mode.allocated_slots > 0 for mode in plan.modes),
        "modes_empty": sum(
            mode.zero_allocation_reason
            in {"no_eligible_candidates", "empty_candidate_pool"}
            for mode in plan.modes
        ),
        "modes_budget_starved": sum(
            mode.zero_allocation_reason == "corridor_budget_exhausted"
            for mode in plan.modes
        ),
        "allocated_slots": {
            "min": min(covered, default=0),
            "median": _median(covered),
            "max": max(covered, default=0),
        },
        "unused_corridor_ceiling": plan.unused_corridor_ceiling,
        "source_production_grade": plan.production_grade,
        "source_feature_fidelity": plan.source_leaderboard_provenance.get(
            "feature_fidelity"
        ),
    }


def _mode_info(mode: CorridorModeLeaderboard, mode_cap: int) -> dict[str, Any]:
    retained = len(mode.candidates)
    eligible = mode.candidates_eligible
    if eligible < retained:
        raise CorridorBudgetError(
            f"mode {mode.corridor_mode_id} eligible count is below retained pool count"
        )
    capacity = min(mode_cap, retained, eligible)
    top = mode.candidates[0] if mode.candidates else None
    priority = (
        {}
        if top is None
        else {
            "top_candidate_utility": float(top.corridor_training_utility or 0.0),
            "top_candidate_membership": float(top.membership_score),
            "top_candidate_centrality": float(top.centrality_score),
        }
    )
    return {
        "mode_id": mode.corridor_mode_id,
        "capacity": capacity,
        "retained": retained,
        "eligible": eligible,
        "mode_support": mode.mode_support,
        "seen": mode.candidates_seen,
        "rejected": mode.candidates_rejected,
        "priority": priority,
    }


def _allocate_counts(
    mode_info: Sequence[Mapping[str, Any]], budget: int
) -> dict[int, int]:
    ordered_ids = sorted(int(item["mode_id"]) for item in mode_info)
    by_id = {int(item["mode_id"]): item for item in mode_info}
    eligible_ids = [
        mode_id for mode_id in ordered_ids if by_id[mode_id]["capacity"] > 0
    ]
    allocations = {mode_id: 0 for mode_id in ordered_ids}
    if budget < len(eligible_ids):
        recipients = sorted(
            eligible_ids,
            key=lambda mode_id: _priority_sort_key(by_id[mode_id]),
        )
        for mode_id in recipients[:budget]:
            allocations[mode_id] = 1
        return allocations
    remaining = budget
    while remaining:
        progressed = False
        for mode_id in ordered_ids:
            if allocations[mode_id] >= by_id[mode_id]["capacity"]:
                continue
            allocations[mode_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise CorridorBudgetError("allocator exhausted before satisfying budget")
    return allocations


def _priority_sort_key(item: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
    priority = item["priority"]
    return (
        -float(priority.get("top_candidate_utility", 0.0)),
        -float(priority.get("top_candidate_membership", 0.0)),
        -float(priority.get("top_candidate_centrality", 0.0)),
        -int(item["mode_support"]),
        int(item["mode_id"]),
    )


def _zero_reason(item: Mapping[str, Any], allocated: int) -> str | None:
    if allocated:
        return None
    if int(item.get("eligible", 0)) == 0 and int(item.get("seen", 1)) == 0:
        return "empty_candidate_pool"
    if int(item.get("eligible", 0)) == 0:
        return "no_eligible_candidates"
    if int(item.get("retained", 0)) == 0:
        return "empty_candidate_pool"
    if int(item.get("capacity", 0)) == 0:
        return "mode_capacity_zero"
    return "corridor_budget_exhausted"


def _plan_summary(plan: CorridorCoveragePlan) -> dict[str, Any]:
    return {
        "modes_observed": len(plan.modes),
        "modes_capacity_positive": sum(
            mode.allocatable_capacity > 0 for mode in plan.modes
        ),
        "modes_covered": sum(mode.allocated_slots > 0 for mode in plan.modes),
        "modes_empty": sum(
            mode.zero_allocation_reason
            in {"no_eligible_candidates", "empty_candidate_pool"}
            for mode in plan.modes
        ),
        "modes_budget_starved": sum(
            mode.zero_allocation_reason == "corridor_budget_exhausted"
            for mode in plan.modes
        ),
        "actual_corridor_budget": plan.actual_corridor_budget,
        "global_budget": plan.global_budget,
        "raw_mode_capacity": plan.raw_mode_capacity,
        "unused_corridor_ceiling": plan.unused_corridor_ceiling,
        "allocated_slots": sum(mode.allocated_slots for mode in plan.modes),
    }


def _default_source_provenance(
    leaderboards: CorridorLeaderboardArtifact,
) -> dict[str, Any]:
    provenance = leaderboards.feature_provenance
    return {
        "leaderboard_artifact_id": "in_memory_c2",
        "c2_policy_id": leaderboards.policy.policy_id,
        "feature_fidelity": provenance.fidelity if provenance else "none",
        "source_artifact_id": provenance.source_artifact_id if provenance else None,
        "source_artifact_hash": provenance.source_artifact_hash if provenance else None,
        "production_grade": leaderboards.production_grade,
        "compatibility_proxy_used": bool(
            provenance and provenance.compatibility_proxy_used
        ),
        "leaderboard_manifest_sha256": None,
        "mode_leaderboards_sha256": None,
    }


def _plan_from_dict(payload: Mapping[str, Any]) -> CorridorCoveragePlan:
    if payload.get("schema_version") != CORRIDOR_BUDGET_SCHEMA:
        raise CorridorBudgetError("unsupported coverage plan schema")
    policy_payload = payload.get("policy")
    if not isinstance(policy_payload, Mapping):
        raise CorridorBudgetError("coverage plan policy is missing")
    policy = CorridorBudgetPolicy(
        total_selected_exemplar_budget=policy_payload["total_selected_exemplar_budget"],
        corridor_budget_fraction=policy_payload["corridor_budget_fraction"],
        corridor_budget_max=policy_payload.get("corridor_budget_max"),
        corridor_mode_cap=policy_payload["corridor_mode_cap"],
        allocation_policy=policy_payload["allocation_policy"],
        allow_nonproduction_leaderboards=policy_payload.get(
            "allow_nonproduction_leaderboards", False
        ),
        nonproduction_override_reason=policy_payload.get(
            "nonproduction_override_reason"
        ),
    )
    modes = tuple(_mode_from_dict(item) for item in payload.get("modes", ()))
    return CorridorCoveragePlan(
        policy=policy,
        source_leaderboard_provenance=dict(
            payload.get("source_leaderboard_provenance") or {}
        ),
        modes=modes,
        fractional_ceiling=payload["fractional_ceiling"],
        corridor_budget_ceiling=payload["corridor_budget_ceiling"],
        raw_mode_capacity=payload["raw_mode_capacity"],
        actual_corridor_budget=payload["actual_corridor_budget"],
        global_budget=payload["global_budget"],
        summary=payload.get("summary"),
    )


def _mode_from_dict(payload: Mapping[str, Any]) -> CorridorModeAllocation:
    return CorridorModeAllocation(
        corridor_mode_id=payload["corridor_mode_id"],
        allocatable_capacity=payload["allocatable_capacity"],
        allocated_slots=payload["allocated_slots"],
        retained_pool_count=payload["retained_pool_count"],
        eligible_candidate_count=payload["eligible_candidate_count"],
        mode_support=payload["mode_support"],
        candidates_seen=payload["candidates_seen"],
        candidates_rejected=payload["candidates_rejected"],
        zero_allocation_reason=payload.get("zero_allocation_reason"),
        coverage_priority=dict(payload.get("coverage_priority") or {}),
    )


def _decimal_fraction(value: Decimal | float | str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, float, int, str)):
        raise ValueError("corridor_budget_fraction must be a finite decimal")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("corridor_budget_fraction must be a finite decimal") from exc
    if not decimal.is_finite() or not Decimal("0") <= decimal <= Decimal("1"):
        raise ValueError("corridor_budget_fraction must be finite and in [0, 1]")
    return decimal


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _unit_finite(value: Any) -> bool:
    try:
        if isinstance(value, bool):
            return False
        return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0
