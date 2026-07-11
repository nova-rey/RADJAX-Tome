from __future__ import annotations

import json
import math

import pytest

from radjax_tome.fingerprint.corridor_archetypes import (
    CorridorArchetypePolicy,
    CorridorCandidateFeatures,
    score_corridor_archetype_candidate,
)

POLICY = CorridorArchetypePolicy()


def _features(**overrides: object) -> CorridorCandidateFeatures:
    values: dict[str, object] = {
        "candidate_id": "example-0:3",
        "position": 3,
        "corridor_mode_id": 2,
        "assignment_status": "linked",
        "membership_strength": 0.9,
        "core_distance": 0.2,
        "mode_support": 20,
        "difficulty_score": 0.6,
        "quality_score": 0.8,
        "corridor_fingerprint_id": "fp-2",
    }
    values.update(overrides)
    return CorridorCandidateFeatures(**values)


def test_strong_central_moderately_difficult_candidate_is_eligible() -> None:
    result = score_corridor_archetype_candidate(_features(), POLICY)

    assert result.eligible is True
    assert result.eligibility_reasons == ()
    assert 0.0 <= result.corridor_training_utility <= 1.0
    assert 0.0 <= result.membership_score <= 1.0
    assert 0.0 <= result.centrality_score <= 1.0
    assert 0.0 <= result.useful_difficulty_score <= 1.0


def test_weak_membership_cannot_be_rescued_by_extreme_difficulty() -> None:
    result = score_corridor_archetype_candidate(
        _features(membership_strength=0.1, difficulty_score=1.0),
        POLICY,
    )

    assert result.eligible is False
    assert "membership_below_minimum" in result.eligibility_reasons
    assert result.corridor_training_utility is None


def test_outside_core_cannot_be_rescued_by_extreme_difficulty() -> None:
    result = score_corridor_archetype_candidate(
        _features(core_distance=0.9, difficulty_score=1.0),
        POLICY,
    )

    assert result.eligible is False
    assert result.eligibility_reasons == ("outside_corridor_core",)
    assert result.corridor_training_utility is None


def test_easy_central_candidate_scores_below_identical_harder_candidate() -> None:
    easy = score_corridor_archetype_candidate(_features(difficulty_score=0.1), POLICY)
    hard = score_corridor_archetype_candidate(_features(difficulty_score=0.9), POLICY)

    assert easy.eligible and hard.eligible
    assert hard.corridor_training_utility > easy.corridor_training_utility


def test_uncertainty_heavy_candidate_can_be_central_and_eligible() -> None:
    result = score_corridor_archetype_candidate(
        _features(difficulty_score=1.0, membership_strength=0.95),
        POLICY,
    )

    assert result.eligible is True


@pytest.mark.parametrize(
    ("features", "reason"),
    (
        (
            _features(corridor_mode_id=None, assignment_status="missing"),
            "unassigned_corridor",
        ),
        (_features(assignment_status="broken"), "invalid_assignment_status"),
        (_features(position=-1), "invalid_position"),
        (_features(position_valid=False), "invalid_position"),
        (_features(membership_strength=math.nan), "nonfinite_feature"),
        (_features(core_distance=math.inf), "nonfinite_feature"),
        (_features(difficulty_score=1.1), "feature_out_of_range"),
        (_features(mode_support=0), "mode_support_below_minimum"),
    ),
)
def test_eligibility_reasons_are_stable(
    features: CorridorCandidateFeatures,
    reason: str,
) -> None:
    result = score_corridor_archetype_candidate(features, POLICY)

    assert result.eligible is False
    assert reason in result.eligibility_reasons
    assert result.corridor_training_utility is None


def test_membership_monotonicity() -> None:
    lower = score_corridor_archetype_candidate(
        _features(membership_strength=0.6), POLICY
    )
    higher = score_corridor_archetype_candidate(
        _features(membership_strength=0.8), POLICY
    )

    assert higher.corridor_training_utility >= lower.corridor_training_utility


def test_difficulty_monotonicity() -> None:
    lower = score_corridor_archetype_candidate(_features(difficulty_score=0.2), POLICY)
    higher = score_corridor_archetype_candidate(_features(difficulty_score=0.8), POLICY)

    assert higher.corridor_training_utility >= lower.corridor_training_utility


def test_core_distance_monotonicity() -> None:
    near = score_corridor_archetype_candidate(_features(core_distance=0.1), POLICY)
    far = score_corridor_archetype_candidate(_features(core_distance=0.4), POLICY)

    assert near.centrality_score > far.centrality_score
    assert near.corridor_training_utility > far.corridor_training_utility


def test_boundary_values_are_bounded() -> None:
    result = score_corridor_archetype_candidate(
        _features(membership_strength=1.0, core_distance=0.0, difficulty_score=1.0),
        POLICY,
    )

    assert result.corridor_training_utility == pytest.approx(1.0)
    assert result.centrality_score == 1.0


def test_repeated_identical_features_are_identical() -> None:
    first = score_corridor_archetype_candidate(_features(), POLICY)
    second = score_corridor_archetype_candidate(_features(), POLICY)

    assert first == second


def test_default_weights_prioritize_membership_and_centrality() -> None:
    assert POLICY.membership_weight >= POLICY.difficulty_weight
    assert POLICY.centrality_weight >= POLICY.difficulty_weight
    assert sum(POLICY.normalized_weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "overrides",
    (
        {"minimum_membership_strength": -0.1},
        {"maximum_core_distance": 1.1},
        {"minimum_mode_support": -1},
        {"membership_weight": math.nan},
        {"membership_weight": 0.0, "centrality_weight": 0.0, "difficulty_weight": 0.0},
        {"difficulty_weight": 0.5, "membership_weight": 0.4},
    ),
)
def test_invalid_policies_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CorridorArchetypePolicy(**overrides)


def test_current_corridor_record_mapping_uses_existing_fields() -> None:
    features = CorridorCandidateFeatures.from_mapping(
        {
            "selected_example_id": "corpus-0",
            "selected_position": 4,
            "corridor_mode_id": 3,
            "corridor_fingerprint_id": "fp-3",
            "corridor_assignment_status": "linked",
        }
    )

    result = score_corridor_archetype_candidate(features, POLICY)
    assert result.eligible is True
    assert result.candidate_id == "corpus-0"
    assert result.corridor_mode_id == 3


@pytest.mark.parametrize("mode_id", (True, False, 1.0, "1"))
def test_features_reject_non_integer_corridor_mode_ids(mode_id: object) -> None:
    with pytest.raises(ValueError):
        _features(corridor_mode_id=mode_id)


@pytest.mark.parametrize("mode_support", (True, False, 1.0, "1", -1))
def test_features_reject_invalid_mode_support(mode_support: object) -> None:
    with pytest.raises(ValueError):
        _features(mode_support=mode_support)


@pytest.mark.parametrize("position_valid", ("false", "FALSE", "true", "TRUE"))
def test_mapping_parses_only_explicit_position_valid_strings(
    position_valid: str,
) -> None:
    features = CorridorCandidateFeatures.from_mapping(
        {
            "candidate_id": "candidate",
            "position": 1,
            "corridor_mode_id": 1,
            "assignment_status": "linked",
            "position_valid": position_valid,
        }
    )

    assert features.position_valid is (position_valid.lower() == "true")


def test_mapping_rejects_ambiguous_position_valid_values() -> None:
    with pytest.raises(ValueError):
        CorridorCandidateFeatures.from_mapping(
            {
                "candidate_id": "candidate",
                "position": 1,
                "corridor_mode_id": 1,
                "assignment_status": "linked",
                "position_valid": "yes",
            }
        )


def test_serialization_is_json_safe_and_ineligible_utility_is_null() -> None:
    result = score_corridor_archetype_candidate(
        _features(membership_strength=0.0, difficulty_score=1.0), POLICY
    )
    encoded = json.dumps(result.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == "radjax.c1_corridor_archetype_score.v1"
    assert decoded["corridor_training_utility"] is None
    assert json.dumps(result.to_dict(), sort_keys=True) == encoded
