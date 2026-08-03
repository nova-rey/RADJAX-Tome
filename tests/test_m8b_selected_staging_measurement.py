"""Frozen M8B.1 comparison rules; these are benchmark-only, never authority."""

from __future__ import annotations

import pytest

from radjax_tome.builder.delivery.measurement import (
    M8BStagingStatistics,
    validate_m8b_statistics_receipt,
)


def test_frozen_statistics_follow_the_approved_formulas() -> None:
    statistics = M8BStagingStatistics()

    assert statistics.summarize([5.0, 1.0, 3.0]) == {
        "median": 3.0,
        "spread": 4.0,
        "normalized_spread": 4.0 / 3.0,
    }
    assert statistics.summarize([0.0, 0.0, 0.0])["normalized_spread"] == 0.0
    assert statistics.combined_spread([10.0, 12.0, 11.0], [7.0, 8.0, 9.0]) == (8.0**0.5)
    assert statistics.improvement_beyond_noise(
        [100.0, 100.0, 100.0], [90.0, 90.0, 90.0]
    )
    assert statistics.materially_regresses([100.0, 100.0, 100.0], [106.0, 106.0, 106.0])
    assert statistics.memory_limit([10, 12, 11]) == 14


def test_statistics_require_three_measurements_and_reject_changed_receipt() -> None:
    statistics = M8BStagingStatistics()
    with pytest.raises(ValueError, match="exactly three"):
        statistics.summarize([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly three"):
        statistics.memory_limit([1, 2])

    receipt: dict[str, object] = {"statistics": statistics.receipt_projection()}
    validate_m8b_statistics_receipt(receipt)
    receipt["statistics"] = {**statistics.receipt_projection(), "noise_multiplier": 1.0}
    with pytest.raises(ValueError, match="frozen definitions"):
        validate_m8b_statistics_receipt(receipt)


def test_zero_median_with_nonzero_spread_is_explicitly_undefined() -> None:
    assert M8BStagingStatistics().summarize([-1.0, 0.0, 1.0]) == {
        "median": 0.0,
        "spread": 2.0,
        "normalized_spread": None,
    }
