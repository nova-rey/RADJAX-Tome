from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.fingerprint.corridor_archetypes import CorridorCandidateFeatures
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    write_corridor_coverage_plan,
)
from radjax_tome.fingerprint.corridor_claims import (
    CLAIM_MANIFEST_FILENAME,
    CorridorClaimError,
    CorridorGlobalClaimPolicy,
    ExistingGlobalBoardInput,
    GlobalBoard,
    GlobalBoardCandidate,
    SelectionObligation,
    claim_corridor_then_backfill_global,
    inspect_corridor_global_claim_artifact,
    validate_corridor_global_claim_artifact,
    write_corridor_global_claim_result,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorCandidateRecord,
    CorridorFeatureProvenance,
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
from tests.helpers.subprocess import run_cli

PROVENANCE = CorridorFeatureProvenance(
    source_artifact_schema="corridor_mode_assignments_v3",
    source_artifact_id="c4-corridor-fixture",
    source_artifact_hash="c" * 64,
    fidelity="explicit",
)


def _record(example_id: str, mode: int, position: int = 0) -> CorridorCandidateRecord:
    return CorridorCandidateRecord(
        features=CorridorCandidateFeatures(
            candidate_id=example_id,
            position=position,
            corridor_mode_id=mode,
            assignment_status="linked",
            membership_strength=0.9,
            core_distance=0.1,
            mode_support=4,
            difficulty_score=0.8,
        ),
        feature_provenance=PROVENANCE,
    )


def _inputs(*, corridor_count: int = 2, pool_count: int = 3):
    records = [
        _record(f"corridor-{mode}-{index}", mode)
        for mode in range(corridor_count)
        for index in range(pool_count)
    ]
    leaderboards = build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=pool_count),
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=5,
            corridor_budget_fraction="0.4",
            corridor_mode_cap=1,
        ),
    )
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard(
                board_id="global_max_entropy",
                priority=0,
                requested_slots=2,
                candidates=tuple(
                    GlobalBoardCandidate(
                        example_id=f"global-a-{index}",
                        position=0,
                        rank=index + 1,
                        score=10.0 - index,
                    )
                    for index in range(4)
                ),
            ),
            GlobalBoard(
                board_id="low_confidence",
                priority=1,
                requested_slots=2,
                candidates=tuple(
                    GlobalBoardCandidate(
                        example_id=f"global-b-{index}",
                        position=0,
                        rank=index + 1,
                        score=8.0 - index,
                    )
                    for index in range(4)
                ),
            ),
        ),
        source_provenance={
            "source_artifact_id": "global-c4-fixture",
            "source_artifact_hash": "d" * 64,
            "production_grade": True,
        },
    )
    policy = CorridorGlobalClaimPolicy(total_selected_exemplar_budget=5)
    return leaderboards, plan, global_input, policy


def test_no_collision_claims_corridors_then_global_remainder() -> None:
    leaderboards, plan, global_input, policy = _inputs()
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert len(result.corridor_claims) == 2
    assert len(result.global_claims) == 3
    assert len(result.selected_coordinates) == 5
    assert result.collision_obligations == ()
    assert [claim.corridor_mode_id for claim in result.corridor_claims] == [0, 1]
    assert result.selected_coordinates[0].primary_claim == (
        "fingerprint_corridor_representative"
    )


def test_corridor_uses_c2_rank_order_and_claims_modes_ascending() -> None:
    leaderboards, plan, global_input, policy = _inputs(pool_count=4)
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert [claim.c2_rank for claim in result.corridor_claims] == [1, 1]
    assert [claim.corridor_mode_id for claim in result.corridor_claims] == [0, 1]
    assert result.corridor_claims[0].example_id == "corridor-0-0"
    assert result.corridor_claims[1].example_id == "corridor-1-0"


def test_corridor_collision_records_global_obligation_and_backfills() -> None:
    leaderboards, plan, _, policy = _inputs()
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard(
                board_id="global_max_entropy",
                priority=0,
                requested_slots=2,
                candidates=(
                    GlobalBoardCandidate("corridor-0-0", 0, 1, 10.0),
                    GlobalBoardCandidate("global-replacement", 0, 2, 9.0),
                ),
            ),
        ),
        source_provenance={"source_artifact_id": "global", "production_grade": True},
    )
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=5,
        require_full_budget=False,
    )
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert len(result.collision_obligations) == 1
    collision = result.collision_obligations[0]
    assert collision.collision_kind == "corridor"
    assert collision.coordinate == ("corridor-0-0", 0)
    assert result.global_claims[0].example_id == "global-replacement"
    assert result.global_claims[0].backfilled is True
    assert result.backfill_lineage[0].replacement_rank == 2
    corridor_item = result.selected_coordinates[0]
    assert [obligation.role for obligation in corridor_item.obligations] == [
        "fingerprint_corridor_representative",
        "global_board",
    ]


def test_multiple_global_collisions_backfill_in_rank_order() -> None:
    leaderboards, plan, _, policy = _inputs()
    board = GlobalBoard(
        board_id="global_max_entropy",
        priority=0,
        requested_slots=2,
        candidates=(
            GlobalBoardCandidate("corridor-0-0", 0, 1, 10.0),
            GlobalBoardCandidate("corridor-1-0", 0, 2, 9.0),
            GlobalBoardCandidate("replacement", 0, 3, 8.0),
        ),
    )
    global_input = ExistingGlobalBoardInput(
        boards=(board,),
        source_provenance={"source_artifact_id": "global", "production_grade": True},
    )
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=5,
        require_full_budget=False,
    )
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert len(result.collision_obligations) == 2
    assert len(result.global_claims) == 1
    assert result.global_claims[0].example_id == "replacement"
    assert [item.skipped_rank for item in result.backfill_lineage] == [1, 2]


def test_cross_global_collision_preserves_both_board_obligations() -> None:
    leaderboards, plan, _, policy = _inputs()
    shared = GlobalBoardCandidate("shared", 0, 1, 10.0)
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard("global_max_entropy", 0, 1, (shared,)),
            GlobalBoard(
                "low_confidence",
                1,
                1,
                (shared, GlobalBoardCandidate("second", 0, 2, 8.0)),
            ),
        ),
        source_provenance={"source_artifact_id": "global", "production_grade": True},
    )
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=5,
        require_full_budget=False,
    )
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert result.collision_obligations[0].collision_kind == "global"
    shared_item = next(
        item for item in result.selected_coordinates if item.example_id == "shared"
    )
    assert [item.source_id for item in shared_item.obligations] == [
        "global_max_entropy",
        "low_confidence",
    ]
    assert result.global_claims[-1].example_id == "second"


def test_zero_global_budget_produces_corridor_only_result() -> None:
    records = [_record("corridor", 0)]
    leaderboards = build_corridor_candidate_leaderboards(
        records, CorridorLeaderboardPolicy(candidate_pool_cap=1)
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=1,
            corridor_budget_fraction=1,
            corridor_mode_cap=1,
        ),
    )
    global_input = ExistingGlobalBoardInput(
        boards=(),
        source_provenance={"source_artifact_id": "global", "production_grade": True},
    )
    result = claim_corridor_then_backfill_global(
        leaderboards,
        plan,
        global_input,
        CorridorGlobalClaimPolicy(total_selected_exemplar_budget=1),
    )

    assert len(result.corridor_claims) == 1
    assert result.global_claims == ()


def test_underfilled_supply_can_be_explicitly_permitted() -> None:
    leaderboards, plan, _, _ = _inputs()
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard(
                "global_max_entropy",
                0,
                1,
                (GlobalBoardCandidate("only", 0, 1, 1.0),),
            ),
        ),
        source_provenance={"source_artifact_id": "global", "production_grade": True},
    )
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=5,
        require_full_budget=False,
    )
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )

    assert len(result.selected_coordinates) == 3
    assert result.summary["unfilled_slots"] == 2


def test_source_mismatch_and_nonproduction_fail_closed() -> None:
    leaderboards, plan, global_input, policy = _inputs()
    other_leaderboards = build_corridor_candidate_leaderboards(
        [_record("other", 8)], CorridorLeaderboardPolicy(candidate_pool_cap=1)
    )
    with pytest.raises(CorridorClaimError, match="source"):
        claim_corridor_then_backfill_global(
            other_leaderboards, plan, global_input, policy
        )


def test_reversed_global_mapping_order_is_deterministic() -> None:
    leaderboards, plan, _, policy = _inputs()
    boards = {
        "low_confidence": {
            "priority": 1,
            "requested_slots": 1,
            "candidates": [{"example_id": "b", "position": 0, "score": 1.0}],
        },
        "global_max_entropy": {
            "priority": 0,
            "requested_slots": 1,
            "candidates": [{"example_id": "a", "position": 0, "score": 2.0}],
        },
    }
    policy = CorridorGlobalClaimPolicy(
        total_selected_exemplar_budget=5,
        require_full_budget=False,
    )
    first = claim_corridor_then_backfill_global(leaderboards, plan, boards, policy)
    second = claim_corridor_then_backfill_global(
        leaderboards, plan, dict(reversed(list(boards.items()))), policy
    )
    assert first.to_dict() == second.to_dict()


def test_claim_artifact_round_trip_and_payload_exclusion(tmp_path: Path) -> None:
    leaderboards, plan, global_input, policy = _inputs()
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )
    output = write_corridor_global_claim_result(result, tmp_path / "claims")

    validation = validate_corridor_global_claim_artifact(output)
    inspection = inspect_corridor_global_claim_artifact(output)
    assert validation.ok
    assert inspection["unique_selected_count"] == 5
    assert "top_probs" not in "".join(
        path.read_text() for path in output.glob("*.jsonl")
    )
    assert (output / CLAIM_MANIFEST_FILENAME).is_file()


def test_tampered_claim_coordinate_fails_artifact_validation(tmp_path: Path) -> None:
    leaderboards, plan, global_input, policy = _inputs()
    result = claim_corridor_then_backfill_global(
        leaderboards, plan, global_input, policy
    )
    output = write_corridor_global_claim_result(result, tmp_path / "claims")
    selected_path = output / "selected_coordinates.jsonl"
    rows = [json.loads(line) for line in selected_path.read_text().splitlines()]
    rows[0]["position"] = 99
    selected_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    validation = validate_corridor_global_claim_artifact(output)
    assert validation.status == "fail"


def test_obligation_shape_is_c5_ready_without_multi_role_payload_schema() -> None:
    obligation = SelectionObligation(
        role="fingerprint_corridor_representative",
        source_id="7",
        rank=1,
        score=0.8,
        metadata={"c2_rank": 1},
    )
    assert obligation.to_dict()["role"] == "fingerprint_corridor_representative"


def test_public_cli_claims_validated_c2_c3_and_global_supply(tmp_path: Path) -> None:
    leaderboards, plan, _, _ = _inputs()
    leaderboards_path = write_corridor_candidate_leaderboards(
        leaderboards,
        tmp_path / "leaderboards",
    )
    plan_path = write_corridor_coverage_plan(plan, tmp_path / "coverage_plan")
    global_path = tmp_path / "global.json"
    global_path.write_text(
        json.dumps(
            {
                "schema_version": "radjax.c4_global_board_supply.v1",
                "source_provenance": {
                    "source_artifact_id": "global-cli-fixture",
                    "source_artifact_hash": "e" * 64,
                    "production_grade": True,
                },
                "boards": [
                    {
                        "board_id": "global_max_entropy",
                        "priority": 0,
                        "requested_slots": 3,
                        "candidates": [
                            {
                                "example_id": f"global-{index}",
                                "position": 0,
                                "rank": index + 1,
                                "score": 10.0 - index,
                            }
                            for index in range(3)
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "claims"

    completed = run_cli(
        Path(__file__).resolve().parents[1],
        "claim-corridor-and-backfill-global",
        "--leaderboards",
        str(leaderboards_path),
        "--coverage-plan",
        str(plan_path),
        "--global-leaderboards",
        str(global_path),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    assert '"status": "pass"' in completed.stdout
    assert (output / CLAIM_MANIFEST_FILENAME).is_file()
