from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from radjax_tome.fingerprint.corridor_archetypes import CorridorCandidateFeatures
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetError,
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    inspect_corridor_coverage_plan,
    validate_corridor_coverage_plan,
    validate_corridor_coverage_plan_object,
    write_corridor_coverage_plan,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorCandidateRecord,
    CorridorFeatureProvenance,
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = CorridorFeatureProvenance(
    source_artifact_schema="corridor_mode_assignments_v3",
    source_artifact_id="c3-fixture",
    source_artifact_hash="b" * 64,
    fidelity="explicit",
)


def _record(
    candidate_id: str,
    mode: int,
    *,
    position: int = 0,
    support: int = 10,
    membership: float = 0.9,
    distance: float = 0.1,
    difficulty: float = 0.5,
    provenance: CorridorFeatureProvenance = PROVENANCE,
) -> CorridorCandidateRecord:
    return CorridorCandidateRecord(
        features=CorridorCandidateFeatures(
            candidate_id=candidate_id,
            position=position,
            corridor_mode_id=mode,
            assignment_status="linked",
            membership_strength=membership,
            core_distance=distance,
            mode_support=support,
            difficulty_score=difficulty,
        ),
        feature_provenance=provenance,
    )


def _leaderboards(
    mode_count: int,
    pool_count: int = 10,
    *,
    difficulty_by_mode: dict[int, float] | None = None,
):
    records = [
        _record(
            f"mode-{mode}-candidate-{index}",
            mode,
            difficulty=(difficulty_by_mode or {}).get(mode, 0.5),
        )
        for mode in range(mode_count)
        for index in range(pool_count)
    ]
    return build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=pool_count),
    )


def _allocations(plan) -> list[int]:
    return [mode.allocated_slots for mode in plan.modes]


def test_roadmap_example_uses_real_capacity_and_global_remainder() -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(50),
        CorridorBudgetPolicy(total_selected_exemplar_budget=5000),
    )

    assert plan.fractional_ceiling == 2500
    assert plan.actual_corridor_budget == 500
    assert plan.global_budget == 4500
    assert sum(_allocations(plan)) == 500


def test_dense_example_reaches_fractional_ceiling() -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(400),
        CorridorBudgetPolicy(total_selected_exemplar_budget=5000),
    )

    assert plan.actual_corridor_budget == 2500
    assert plan.global_budget == 2500
    assert max(_allocations(plan)) == 7


def test_fraction_and_hard_max_are_floored_without_binary_float_drift() -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(10, pool_count=10),
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=10,
            corridor_budget_fraction="0.3",
            corridor_budget_max=2,
        ),
    )

    assert plan.fractional_ceiling == 3
    assert plan.corridor_budget_ceiling == 2
    assert plan.actual_corridor_budget == 2


def test_breadth_first_round_robin_precedes_mode_depth() -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(3, pool_count=3),
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=6, corridor_budget_fraction=1
        ),
    )

    assert _allocations(plan) == [2, 2, 2]


def test_unequal_capacities_are_exhausted_without_wasting_budget() -> None:
    leaderboards = build_corridor_candidate_leaderboards(
        [
            _record("m0-a", 0),
            _record("m0-b", 0),
            _record("m1-a", 1),
        ],
        CorridorLeaderboardPolicy(candidate_pool_cap=10),
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=10, corridor_budget_fraction=1
        ),
    )

    assert _allocations(plan) == [2, 1]
    assert plan.actual_corridor_budget == 3
    assert plan.unused_corridor_ceiling == 7


def test_oversubscription_uses_quality_priority_then_mode_id() -> None:
    leaderboards = _leaderboards(
        4,
        pool_count=1,
        difficulty_by_mode={0: 0.2, 1: 0.9, 2: 0.9, 3: 0.1},
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=2, corridor_budget_fraction=1
        ),
    )

    assert _allocations(plan) == [0, 1, 1, 0]


def test_shuffled_logical_input_produces_identical_plan() -> None:
    records = [_record(f"candidate-{index}", index % 5) for index in range(50)]
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    first = allocate_corridor_coverage(
        build_corridor_candidate_leaderboards(records, CorridorLeaderboardPolicy(10)),
        CorridorBudgetPolicy(total_selected_exemplar_budget=17),
    )
    second = allocate_corridor_coverage(
        build_corridor_candidate_leaderboards(shuffled, CorridorLeaderboardPolicy(10)),
        CorridorBudgetPolicy(total_selected_exemplar_budget=17),
    )

    assert first.to_dict() == second.to_dict()


def test_zero_budget_and_zero_fraction_leave_global_budget_intact() -> None:
    leaderboards = _leaderboards(3)
    zero_budget = allocate_corridor_coverage(
        leaderboards, CorridorBudgetPolicy(total_selected_exemplar_budget=0)
    )
    zero_fraction = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=20, corridor_budget_fraction=0
        ),
    )

    assert _allocations(zero_budget) == [0, 0, 0]
    assert zero_budget.global_budget == 0
    assert zero_fraction.actual_corridor_budget == 0
    assert zero_fraction.global_budget == 20
    assert all(
        mode.zero_allocation_reason == "corridor_budget_exhausted"
        for mode in zero_fraction.modes
    )


def test_no_eligible_modes_remain_visible_with_zero_capacity() -> None:
    records = [_record("rejected", 4, membership=0.1)]
    leaderboards = build_corridor_candidate_leaderboards(
        records, CorridorLeaderboardPolicy(candidate_pool_cap=4)
    )
    plan = allocate_corridor_coverage(
        leaderboards, CorridorBudgetPolicy(total_selected_exemplar_budget=20)
    )

    assert len(plan.modes) == 1
    assert plan.actual_corridor_budget == 0
    assert plan.modes[0].zero_allocation_reason == "no_eligible_candidates"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"total_selected_exemplar_budget": -1},
        {"total_selected_exemplar_budget": True},
        {"total_selected_exemplar_budget": 1.5},
        {"total_selected_exemplar_budget": 1, "corridor_budget_fraction": float("nan")},
        {"total_selected_exemplar_budget": 1, "corridor_budget_fraction": 1.1},
        {"total_selected_exemplar_budget": 1, "corridor_mode_cap": 0},
        {"total_selected_exemplar_budget": 1, "corridor_budget_max": -1},
    ),
)
def test_invalid_policy_values_fail_closed(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        CorridorBudgetPolicy(**kwargs)


def test_nonproduction_source_requires_explicit_override() -> None:
    proxy = CorridorFeatureProvenance(
        source_artifact_schema="synthetic",
        source_artifact_id="proxy",
        fidelity="compatibility_proxy",
        compatibility_proxy_used=True,
    )
    leaderboards = build_corridor_candidate_leaderboards(
        [_record("proxy", 0, provenance=proxy)],
        CorridorLeaderboardPolicy(
            candidate_pool_cap=1,
            allow_compatibility_proxies=True,
            proxy_override_reason="test only",
        ),
    )

    with pytest.raises(CorridorBudgetError, match="non-production"):
        allocate_corridor_coverage(
            leaderboards, CorridorBudgetPolicy(total_selected_exemplar_budget=1)
        )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=1,
            allow_nonproduction_leaderboards=True,
            nonproduction_override_reason="test only",
        ),
    )
    assert plan.production_grade is False
    assert (
        validate_corridor_coverage_plan_object(plan, production_grade=False).status
        == "warn"
    )


def test_round_trip_hashes_and_inspection(tmp_path: Path) -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(4), CorridorBudgetPolicy(total_selected_exemplar_budget=7)
    )
    output = write_corridor_coverage_plan(plan, tmp_path / "coverage")

    validation = validate_corridor_coverage_plan(output)
    inspection = inspect_corridor_coverage_plan(output)
    assert validation.ok
    assert inspection["actual_corridor_budget"] == 3
    assert (output / "coverage_plan.json").is_file()
    assert (output / "validation_report.json").is_file()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload["modes"][0].__setitem__("allocated_slots", 0),
        lambda payload: payload.__setitem__("corridor_budget_ceiling", 0),
        lambda payload: payload["modes"][0].__setitem__(
            "zero_allocation_reason", "corridor_budget_exhausted"
        ),
        lambda payload: payload["summary"].__setitem__("global_budget", 999),
        lambda payload: payload["source_leaderboard_provenance"].__setitem__(
            "source_artifact_hash", "bad-hash"
        ),
    ),
)
def test_tampered_plan_fails_validation(tmp_path: Path, mutator) -> None:
    plan = allocate_corridor_coverage(
        _leaderboards(3), CorridorBudgetPolicy(total_selected_exemplar_budget=5)
    )
    output = write_corridor_coverage_plan(plan, tmp_path / "coverage")
    plan_path = output / "coverage_plan.json"
    payload = json.loads(plan_path.read_text())
    mutator(payload)
    plan_path.write_text(json.dumps(payload) + "\n")

    validation = validate_corridor_coverage_plan(output)
    assert validation.status == "fail"
    assert validation.blockers


def test_cli_builds_and_validates_plan(tmp_path: Path) -> None:
    leaderboard_dir = write_corridor_candidate_leaderboards(
        _leaderboards(3), tmp_path / "leaderboards"
    )
    output = tmp_path / "coverage"
    result = run_cli(
        ROOT,
        "allocate-fingerprint-corridor-coverage",
        "--leaderboards",
        str(leaderboard_dir),
        "--total-selected-exemplar-budget",
        "10",
        "--corridor-budget-fraction",
        "0.5",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "pass"' in result.stdout
    assert validate_corridor_coverage_plan(output).ok


def test_cli_missing_leaderboards_fails_actionably(tmp_path: Path) -> None:
    result = run_cli(
        ROOT,
        "allocate-fingerprint-corridor-coverage",
        "--leaderboards",
        str(tmp_path / "missing"),
        "--total-selected-exemplar-budget",
        "10",
        "--output",
        str(tmp_path / "coverage"),
    )

    assert result.returncode == 1
    assert "cannot load invalid corridor leaderboard artifact" in result.stderr
