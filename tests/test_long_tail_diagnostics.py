from __future__ import annotations

import pytest

from radjax_tome.builder.exemplar_selection import (
    PATH_B_FULFILLMENT_POLICY,
    ExemplarCandidate,
    select_exemplars,
)
from radjax_tome.builder.long_tail import (
    FULL_VOCAB_OR_NEAR_FULL_VOCAB,
    LONG_TAIL,
    NORMAL,
    SUSPICIOUS_FLAT,
    VERY_LONG_TAIL,
    LongTailPolicy,
    is_perverse_long_tail,
    long_tail_diagnostics,
    long_tail_summary,
)

POLICY = LongTailPolicy(
    long_tail_warning_k=8,
    very_long_tail_warning_k=16,
    perverse_tail_warning_k=32,
)


@pytest.mark.parametrize(
    ("effective_top_k", "expected_class"),
    (
        (7, NORMAL),
        (8, LONG_TAIL),
        (16, VERY_LONG_TAIL),
        (32, SUSPICIOUS_FLAT),
        (100, FULL_VOCAB_OR_NEAR_FULL_VOCAB),
    ),
)
def test_dynamic_top_k_long_tail_classification(
    effective_top_k: int,
    expected_class: str,
) -> None:
    diagnostic = long_tail_diagnostics(
        effective_top_k=effective_top_k,
        top_mass=0.95,
        vocab_size=100,
        dynamic_mass_threshold=0.95,
        dynamic_top_k_max=100,
        policy=POLICY,
    )

    assert diagnostic["long_tail_class"] == expected_class
    assert diagnostic["effective_top_k_fraction_of_vocab"] == pytest.approx(
        effective_top_k / 100.0
    )
    assert diagnostic["top_k_saturated"] is (effective_top_k == 100)
    if expected_class == NORMAL:
        assert diagnostic["long_tail_warnings"] == []
    else:
        assert diagnostic["long_tail_warnings"]


def test_long_tail_summary_counts_classes_correctly() -> None:
    diagnostics = [
        long_tail_diagnostics(
            effective_top_k=value,
            top_mass=0.95,
            vocab_size=100,
            dynamic_mass_threshold=0.95,
            dynamic_top_k_max=100,
            policy=POLICY,
        )
        for value in (7, 8, 16, 32, 100)
    ]

    summary = long_tail_summary(diagnostics)

    assert summary == {
        "count": 5,
        "normal_count": 1,
        "long_tail_count": 1,
        "very_long_tail_count": 1,
        "suspicious_flat_count": 1,
        "full_vocab_or_near_full_vocab_count": 1,
        "saturated_count": 1,
        "max_effective_top_k": 100,
        "mean_effective_top_k": pytest.approx(32.6),
        "max_effective_top_k_fraction_of_vocab": 1.0,
    }


def test_long_tail_diagnostic_clamps_probability_mass_for_reporting() -> None:
    diagnostic = long_tail_diagnostics(
        effective_top_k=100,
        top_mass=1.0078125,
        vocab_size=100,
        dynamic_mass_threshold=0.95,
        dynamic_top_k_max=100,
        policy=POLICY,
    )

    assert diagnostic["top_mass"] == 1.0
    assert diagnostic["raw_top_mass"] == pytest.approx(1.0078125)
    assert diagnostic["top_mass_clamped"] is True


def test_reject_perverse_filter_replaces_candidate_with_next_eligible() -> None:
    perverse = _candidate("perverse", score=10.0, effective_top_k=32)
    eligible = _candidate("eligible", score=9.0, effective_top_k=7)

    kept = select_exemplars(
        (perverse, eligible),
        capture_mode="two_pass_sparse_exemplar",
        fulfillment_policy=PATH_B_FULFILLMENT_POLICY,
        board_capacity=1,
        budget_examples=1,
        created_at="2026-01-01T00:00:00+00:00",
    )
    rejected = select_exemplars(
        (perverse, eligible),
        capture_mode="two_pass_sparse_exemplar",
        fulfillment_policy=PATH_B_FULFILLMENT_POLICY,
        board_capacity=1,
        budget_examples=1,
        created_at="2026-01-01T00:00:00+00:00",
        candidate_filter=lambda candidate: (
            not is_perverse_long_tail(
                long_tail_diagnostics(
                    effective_top_k=int(
                        candidate.score_fields["diagnostic_effective_top_k"]
                    ),
                    top_mass=0.95,
                    vocab_size=100,
                    dynamic_mass_threshold=0.95,
                    dynamic_top_k_max=100,
                    policy=POLICY,
                )
            )
        ),
        candidate_filter_name="reject_perverse_dynamic_top_k",
    )

    assert kept["selected_examples"][0]["example_id"] == "perverse"
    assert rejected["selected_examples"][0]["example_id"] == "eligible"
    assert rejected["candidate_filter_rejected_count"] == 1


def _candidate(
    example_id: str,
    *,
    score: float,
    effective_top_k: int,
) -> ExemplarCandidate:
    return ExemplarCandidate(
        example_id=example_id,
        source_shard_id=0,
        source_row=0,
        selected_position=0,
        candidate_positions=(0,),
        sequence_length=1,
        capture_mode="two_pass_sparse_exemplar",
        source_policy="0",
        score_fields={
            "max_entropy": score,
            "mean_entropy": score,
            "selected_position_entropy": score,
            "confidence": 0.5,
            "score_top_token_id": 1.0,
            "source_policy_id": 0.0,
            "position_bucket": 0.0,
            "length_bucket": 0.0,
            "diagnostic_effective_top_k": float(effective_top_k),
        },
        payload_ref={
            "kind": "corridor_exemplar_score_pass_v1",
            "source_shard_id": 0,
            "source_row": 0,
            "source_position": 0,
            "source_score": score,
            "source_top_token_id": 1,
        },
    )
