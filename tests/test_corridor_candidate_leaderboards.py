from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from radjax_tome.fingerprint.corridor_archetypes import (
    CorridorCandidateFeatures,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorCandidateRecord,
    CorridorFeatureProvenance,
    CorridorLeaderboardError,
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    inspect_corridor_candidate_leaderboards,
    load_candidate_records_jsonl,
    validate_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]

PROVENANCE = CorridorFeatureProvenance(
    source_artifact_schema="corridor_mode_assignments_v3",
    source_artifact_id="fixture-corridor",
    source_artifact_hash="a" * 64,
    fidelity="explicit",
)


def _record(
    candidate_id: str,
    *,
    position: int = 0,
    mode: int | None = 1,
    support: int = 10,
    membership: float = 0.9,
    distance: float = 0.1,
    difficulty: float = 0.5,
    status: str = "linked",
    provenance: CorridorFeatureProvenance = PROVENANCE,
) -> CorridorCandidateRecord:
    return CorridorCandidateRecord(
        features=CorridorCandidateFeatures(
            candidate_id=candidate_id,
            position=position,
            corridor_mode_id=mode,
            assignment_status=status,
            membership_strength=membership,
            core_distance=distance,
            mode_support=support,
            difficulty_score=difficulty,
        ),
        feature_provenance=provenance,
    )


def test_multiple_modes_are_separate_and_pools_are_bounded() -> None:
    artifact = build_corridor_candidate_leaderboards(
        [_record(f"candidate-{index}", mode=index % 3) for index in range(18)],
        CorridorLeaderboardPolicy(candidate_pool_cap=2),
    )

    assert [mode.corridor_mode_id for mode in artifact.modes] == [0, 1, 2]
    assert all(len(mode.candidates) <= 2 for mode in artifact.modes)
    assert artifact.summary["retained_candidate_count"] == 6


def test_highest_ranked_eligible_candidates_survive_pool_retention() -> None:
    records = [
        _record(f"candidate-{index}", difficulty=index / 10) for index in range(8)
    ]
    artifact = build_corridor_candidate_leaderboards(
        records,
        CorridorLeaderboardPolicy(candidate_pool_cap=3),
    )

    assert [item.candidate_id for item in artifact.modes[0].candidates] == [
        "candidate-7",
        "candidate-6",
        "candidate-5",
    ]


def test_rejections_are_counted_by_reason_and_not_retained() -> None:
    artifact = build_corridor_candidate_leaderboards(
        [
            _record("weak", membership=0.1),
            _record("far", distance=0.9),
            _record("failed", status="failed"),
        ]
    )

    assert artifact.summary["candidates_eligible"] == 0
    assert artifact.summary["candidates_rejected"] == 3
    assert (
        artifact.summary["rejection_counts_by_reason"]["membership_below_minimum"] == 1
    )
    assert artifact.summary["modes_with_empty_pools"] == 1
    assert artifact.modes[0].candidates == ()


def test_central_uncertainty_heavy_candidate_can_rank() -> None:
    artifact = build_corridor_candidate_leaderboards(
        [_record("uncertain", difficulty=1.0, membership=0.95, distance=0.0)]
    )

    assert artifact.modes[0].candidates[0].candidate_id == "uncertain"


def test_ties_are_stable_under_reversed_and_shuffled_input() -> None:
    records = [_record(f"candidate-{index}") for index in range(8)]
    first = build_corridor_candidate_leaderboards(records, CorridorLeaderboardPolicy(3))
    reversed_result = build_corridor_candidate_leaderboards(
        list(reversed(records)), CorridorLeaderboardPolicy(3)
    )
    shuffled = list(records)
    random.Random(17).shuffle(shuffled)
    shuffled_result = build_corridor_candidate_leaderboards(
        shuffled, CorridorLeaderboardPolicy(3)
    )

    def ids(artifact):
        return [item.candidate_id for item in artifact.modes[0].candidates]

    assert ids(first) == ids(reversed_result) == ids(shuffled_result)


def test_identical_duplicates_collapse_and_conflicts_fail() -> None:
    record = _record("duplicate")
    artifact = build_corridor_candidate_leaderboards([record, record])
    assert artifact.summary["duplicates_collapsed"] == 1
    assert artifact.modes[0].duplicate_count == 1

    with pytest.raises(CorridorLeaderboardError, match="conflicting duplicate"):
        build_corridor_candidate_leaderboards(
            [record, _record("duplicate", difficulty=0.9)]
        )


def test_one_coordinate_cannot_be_assigned_to_multiple_modes() -> None:
    with pytest.raises(CorridorLeaderboardError, match="conflicting duplicate"):
        build_corridor_candidate_leaderboards(
            [_record("same", mode=1), _record("same", mode=2)]
        )


def test_empty_input_is_valid_and_deterministic() -> None:
    first = build_corridor_candidate_leaderboards(())
    second = build_corridor_candidate_leaderboards(())

    assert first.to_dict() == second.to_dict()
    assert first.modes == ()
    assert first.summary["candidates_seen"] == 0


def test_empty_input_round_trips_as_a_valid_artifact(tmp_path: Path) -> None:
    output = write_corridor_candidate_leaderboards(
        build_corridor_candidate_leaderboards(()), tmp_path / "empty"
    )

    assert validate_corridor_candidate_leaderboards(output).ok


def test_proxy_features_fail_by_default_and_are_conspicuous_when_allowed() -> None:
    proxy = CorridorFeatureProvenance(
        source_artifact_schema="synthetic",
        source_artifact_id="test-only",
        fidelity="compatibility_proxy",
        compatibility_proxy_used=True,
    )
    with pytest.raises(CorridorLeaderboardError, match="compatibility_proxy"):
        build_corridor_candidate_leaderboards([_record("proxy", provenance=proxy)])

    policy = CorridorLeaderboardPolicy(
        allow_compatibility_proxies=True,
        proxy_override_reason="unit test only",
    )
    artifact = build_corridor_candidate_leaderboards(
        [_record("proxy", provenance=proxy)], policy
    )
    assert artifact.summary["production_grade"] is False
    assert artifact.warnings


def test_provenance_must_be_shared_and_real_features_are_explicit() -> None:
    other = CorridorFeatureProvenance(
        source_artifact_schema="other",
        source_artifact_id="other",
        fidelity="derived",
    )
    with pytest.raises(CorridorLeaderboardError, match="one feature provenance"):
        build_corridor_candidate_leaderboards(
            [_record("a"), _record("b", provenance=other)]
        )


def test_artifact_round_trip_hashes_and_inspection(tmp_path: Path) -> None:
    artifact = build_corridor_candidate_leaderboards(
        [_record(f"candidate-{index}", mode=index % 2) for index in range(8)]
    )
    output = write_corridor_candidate_leaderboards(artifact, tmp_path / "leaderboards")

    validation = validate_corridor_candidate_leaderboards(output)
    inspection = inspect_corridor_candidate_leaderboards(output)
    manifest = json.loads((output / "manifest.json").read_text())

    assert validation.ok
    assert inspection["modes_observed"] == 2
    assert inspection["feature_fidelity"] == "explicit"
    assert manifest["files"]["mode_leaderboards.jsonl"]["sha256"]
    assert (output / "validation_report.json").is_file()


def test_tampered_order_score_mode_and_hash_fail_validation(tmp_path: Path) -> None:
    artifact = build_corridor_candidate_leaderboards(
        [_record("a", difficulty=0.9), _record("b", difficulty=0.2)]
    )
    output = write_corridor_candidate_leaderboards(artifact, tmp_path / "leaderboards")
    mode_path = output / "mode_leaderboards.jsonl"
    lines = mode_path.read_text().splitlines()
    payload = json.loads(lines[0])
    payload["candidates"][0]["corridor_mode_id"] = 99
    mode_path.write_text(json.dumps(payload) + "\n")

    validation = validate_corridor_candidate_leaderboards(output)
    assert validation.status == "fail"
    assert any("hash mismatch" in item for item in validation.blockers)


def test_candidate_jsonl_loader_preserves_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "features.jsonl"
    path.write_text(
        json.dumps(
            {
                "features": {
                    "candidate_id": "current-0",
                    "position": 2,
                    "corridor_mode_id": 4,
                    "assignment_status": "linked",
                    "membership_strength": 0.8,
                    "core_distance": 0.2,
                    "mode_support": 4,
                    "difficulty_score": 0.7,
                },
                "source_artifact_schema": "corridor_mode_assignments_v3",
                "fidelity": "explicit",
            }
        )
        + "\n"
    )

    records = load_candidate_records_jsonl(path)
    assert records[0].features.candidate_id == "current-0"
    assert records[0].feature_provenance.source_artifact_hash


def test_retained_state_is_bounded_by_modes_times_cap() -> None:
    mode_count = 20
    cap = 3
    artifact = build_corridor_candidate_leaderboards(
        [
            _record(f"candidate-{index}", mode=index % mode_count)
            for index in range(200)
        ],
        CorridorLeaderboardPolicy(candidate_pool_cap=cap),
    )

    assert artifact.summary["retained_candidate_count"] <= mode_count * cap


def test_offline_cli_builds_from_explicit_feature_jsonl(tmp_path: Path) -> None:
    artifact = tmp_path / "source-artifact"
    artifact.mkdir()
    source = tmp_path / "candidate_features.jsonl"
    source.write_text(
        json.dumps(
            {
                "features": {
                    "candidate_id": "cli-candidate",
                    "position": 3,
                    "corridor_mode_id": 2,
                    "assignment_status": "linked",
                    "membership_strength": 0.9,
                    "core_distance": 0.1,
                    "mode_support": 2,
                    "difficulty_score": 0.7,
                },
                "fidelity": "explicit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        ROOT,
        "build-fingerprint-corridor-leaderboards",
        "--artifact",
        str(artifact),
        "--candidate-jsonl",
        str(source),
        "--output",
        str(tmp_path / "leaderboards"),
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "pass"' in result.stdout


def test_offline_cli_fails_closed_without_real_feature_input(tmp_path: Path) -> None:
    artifact = tmp_path / "source-artifact"
    artifact.mkdir()

    result = run_cli(
        ROOT,
        "build-fingerprint-corridor-leaderboards",
        "--artifact",
        str(artifact),
        "--output",
        str(tmp_path / "leaderboards"),
    )

    assert result.returncode == 1
    assert "real compact corridor feature fields are missing" in result.stderr
