from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from radjax_tome.fingerprint.corridor_archetypes import CorridorCandidateFeatures
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
)
from radjax_tome.fingerprint.corridor_claims import (
    CorridorGlobalClaimPolicy,
    ExistingGlobalBoardInput,
    GlobalBoard,
    GlobalBoardCandidate,
    claim_corridor_then_backfill_global,
    write_corridor_global_claim_result,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorCandidateRecord,
    CorridorFeatureProvenance,
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
)
from radjax_tome.fingerprint.multi_role_selection import (
    CORRIDOR_ROLE,
    GLOBAL_ROLE,
    MultiRoleSelectionError,
    build_multi_role_selected_exemplars,
    inspect_multi_role_selection_artifact,
    load_multi_role_selection_artifact,
    project_legacy_selected_exemplars,
    validate_multi_role_selection_artifact,
    write_multi_role_selection_artifact,
)
from tests.helpers.subprocess import run_cli

PROVENANCE = CorridorFeatureProvenance(
    source_artifact_schema="corridor_mode_assignments_v3",
    source_artifact_id="c5-fixture",
    source_artifact_hash="a" * 64,
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


def _claims(*, corridor_fraction: str = "0.4"):
    records = [
        _record(f"corridor-{mode}-{index}", mode)
        for mode in range(2)
        for index in range(3)
    ]
    leaderboards = build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=3),
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=5,
            corridor_budget_fraction=corridor_fraction,
            corridor_mode_cap=1,
        ),
    )
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard(
                "global_max_entropy",
                0,
                3,
                tuple(
                    GlobalBoardCandidate(
                        f"global-a-{index}", 0, index + 1, 10.0 - index
                    )
                    for index in range(3)
                ),
            ),
            GlobalBoard(
                "low_confidence",
                1,
                2,
                tuple(
                    GlobalBoardCandidate(f"global-b-{index}", 0, index + 1, 8.0 - index)
                    for index in range(2)
                ),
            ),
        ),
        source_provenance={
            "source_artifact_id": "c5-global-fixture",
            "source_artifact_hash": "b" * 64,
            "production_grade": True,
        },
    )
    return claim_corridor_then_backfill_global(
        leaderboards,
        plan,
        global_input,
        CorridorGlobalClaimPolicy(total_selected_exemplar_budget=5),
    )


def _collision_claims():
    claims = _claims()
    global_input = ExistingGlobalBoardInput(
        boards=(
            GlobalBoard(
                "global_max_entropy",
                0,
                3,
                (
                    GlobalBoardCandidate("corridor-0-0", 0, 1, 10.0),
                    GlobalBoardCandidate("shared", 0, 2, 9.0),
                    GlobalBoardCandidate("replacement", 0, 3, 8.0),
                ),
            ),
            GlobalBoard(
                "low_confidence",
                1,
                1,
                (
                    GlobalBoardCandidate("shared", 0, 1, 7.0),
                    GlobalBoardCandidate("replacement-b", 0, 2, 6.0),
                ),
            ),
        ),
        source_provenance={
            "source_artifact_id": "c5-global-collision",
            "production_grade": True,
        },
    )
    return claim_corridor_then_backfill_global(
        _claims_leaderboards(claims),
        _claims_plan(claims),
        global_input,
        CorridorGlobalClaimPolicy(total_selected_exemplar_budget=5),
    )


def _claims_leaderboards(claims):
    del claims
    records = [
        _record(f"corridor-{mode}-{index}", mode)
        for mode in range(2)
        for index in range(3)
    ]
    return build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=3),
    )


def _claims_plan(claims):
    del claims
    leaderboards = _claims_leaderboards(None)
    return allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=5,
            corridor_budget_fraction="0.4",
            corridor_mode_cap=1,
        ),
    )


def test_corridor_records_are_one_per_coordinate_with_derived_roles() -> None:
    artifact = build_multi_role_selected_exemplars(_claims())

    assert len(artifact.records) == 5
    assert len({(item.example_id, item.position) for item in artifact.records}) == 5
    assert artifact.records[0].primary_claim == CORRIDOR_ROLE
    assert artifact.records[0].selection_roles == (CORRIDOR_ROLE,)
    assert artifact.records[0].payload_identity["materialization_status"] == (
        "not_materialized_in_c5"
    )


def test_collision_and_cross_global_roles_preserve_one_payload_identity() -> None:
    artifact = build_multi_role_selected_exemplars(_collision_claims())
    shared = next(item for item in artifact.records if item.example_id == "shared")

    assert shared.primary_claim == GLOBAL_ROLE
    assert shared.selection_roles == (GLOBAL_ROLE,)
    assert shared.global_board_ids == ("global_max_entropy", "low_confidence")
    assert len(shared.selection_obligations) == 2
    assert len(
        {item.payload_identity["payload_key"] for item in artifact.records}
    ) == len(artifact.records)


def test_source_passports_are_verified_and_preserved() -> None:
    claims = _claims()
    passports = {
        (item.example_id, item.position): {
            "example_id": item.example_id,
            "position": item.position,
            "source_shard_id": "shard-0",
            "source_row": item.claim_order,
            "source_position": item.position,
            "corridor_mode_id": (
                item.obligations[0].metadata.get("corridor_mode_id")
                if item.obligations[0].role == CORRIDOR_ROLE
                else None
            ),
            "corridor_assignment_status": (
                "linked"
                if item.obligations[0].role == CORRIDOR_ROLE
                else "not_applicable"
            ),
        }
        for item in claims.selected_coordinates
    }
    artifact = build_multi_role_selected_exemplars(
        claims,
        source_passports=passports,
    )

    assert artifact.records[0].source_passport["source_shard_id"] == "shard-0"
    with pytest.raises(MultiRoleSelectionError, match="missing source passport"):
        build_multi_role_selected_exemplars(claims, source_passports={})


def test_legacy_projection_has_exact_coordinate_order_and_count() -> None:
    artifact = build_multi_role_selected_exemplars(_claims())
    projection = project_legacy_selected_exemplars(artifact)

    assert len(projection) == len(artifact.records)
    assert [
        (item["selected_example_id"], item["selected_position"]) for item in projection
    ] == [(item.example_id, item.position) for item in artifact.records]
    assert [item["rank"] for item in projection] == list(range(1, 6))


def test_validation_detects_primary_role_and_payload_tampering() -> None:
    artifact = build_multi_role_selected_exemplars(_claims())
    record = artifact.records[0]
    tampered_record = replace(
        record,
        primary_claim=GLOBAL_ROLE,
        payload_identity={
            "payload_key": "coordinate_v1:wrong",
            "materialization_status": "not_materialized_in_c5",
        },
    )
    invalid = replace(artifact, records=(tampered_record, *artifact.records[1:]))

    validation = validate_multi_role_selection_artifact(invalid, production_grade=False)
    assert validation.status == "fail"
    assert any("payload key mismatch" in item for item in validation.blockers)
    assert any("obligation error" in item for item in validation.blockers)


def test_multiple_corridor_obligations_fail_closed() -> None:
    claims = _claims()
    item = claims.selected_coordinates[0]
    duplicate = replace(
        item,
        obligations=(item.obligations[0], item.obligations[0]),
    )
    invalid_claims = replace(
        claims,
        selected_coordinates=(duplicate, *claims.selected_coordinates[1:]),
    )

    with pytest.raises(MultiRoleSelectionError, match="multiple corridor"):
        build_multi_role_selected_exemplars(invalid_claims)


def test_artifact_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    claims = _claims()
    artifact = build_multi_role_selected_exemplars(claims)
    output = write_multi_role_selection_artifact(artifact, tmp_path / "c5")

    assert validate_multi_role_selection_artifact(output).ok
    loaded = load_multi_role_selection_artifact(output)
    assert loaded.to_dict() == artifact.to_dict()
    assert inspect_multi_role_selection_artifact(output)["status"] == "pass"

    rich_path = output / "selected_exemplars.jsonl"
    rows = [json.loads(line) for line in rich_path.read_text().splitlines()]
    rows[0]["represented_fingerprint_corridor_ids"] = [999]
    rich_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    validation = validate_multi_role_selection_artifact(output)
    assert validation.status == "fail"
    assert any("hash mismatch" in item for item in validation.blockers)


def test_nonproduction_override_is_serialized_without_becoming_production() -> None:
    claims = _claims()
    policy = replace(
        claims.policy,
        allow_nonproduction_sources=True,
        nonproduction_override_reason="C5 fixture",
    )
    nonproduction_claims = replace(
        claims,
        policy=policy,
        source_provenance={**claims.source_provenance, "production_grade": False},
    )
    artifact = build_multi_role_selected_exemplars(nonproduction_claims)

    assert artifact.production_grade is False
    assert (
        validate_multi_role_selection_artifact(
            artifact,
            production_grade=False,
        ).status
        == "warn"
    )


def test_zero_budget_produces_valid_empty_artifact() -> None:
    records = [_record("unused", 0)]
    leaderboards = build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=1),
    )
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=0,
            corridor_budget_fraction="0.5",
            corridor_mode_cap=1,
        ),
    )
    claims = claim_corridor_then_backfill_global(
        leaderboards,
        plan,
        ExistingGlobalBoardInput(
            boards=(),
            source_provenance={"source_artifact_id": "empty", "production_grade": True},
        ),
        CorridorGlobalClaimPolicy(total_selected_exemplar_budget=0),
    )
    artifact = build_multi_role_selected_exemplars(claims)

    assert artifact.records == ()
    assert validate_multi_role_selection_artifact(artifact).ok


def test_public_cli_builds_c5_from_serialized_c4_claims(tmp_path: Path) -> None:
    claims_path = write_corridor_global_claim_result(
        _claims(),
        tmp_path / "c4_claims",
    )
    output = tmp_path / "c5_selection"

    result = run_cli(
        Path(__file__).resolve().parents[1],
        "build-multi-role-selected-exemplars",
        "--claims",
        str(claims_path),
        "--output",
        str(output),
        "--overwrite",
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "pass"' in result.stdout
    assert (output / "selected_exemplars.jsonl").is_file()
    assert (output / "legacy_selected_exemplars.json").is_file()
