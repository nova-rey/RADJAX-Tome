"""Exact-rational full-width composition policy for ordinary leaderboards."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd
from typing import Any


@dataclass(frozen=True)
class FullWidthCompositionPolicy:
    numerator: int = 1
    denominator: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
            raise ValueError("full-width numerator must be an integer")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise ValueError("full-width denominator must be an integer")
        if self.numerator < 1 or self.denominator < 1:
            raise ValueError("full-width ratio must be positive")
        divisor = gcd(self.numerator, self.denominator)
        if divisor != 1:
            raise ValueError("full-width ratio must be reduced")

    @property
    def allowance(self):
        return lambda capacity: max(1, capacity * self.numerator // self.denominator)

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


_DEFAULT_POLICY = FullWidthCompositionPolicy()


def select_final_composition(
    candidates: Iterable[dict[str, Any]],
    capacity: int,
    *,
    policy: FullWidthCompositionPolicy = _DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    """Select a final composition without reserving slots by arrival order."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate["example_id"]), int(candidate["position"]))
        previous = unique.get(key)
        if previous is None or _rank(candidate) < _rank(previous):
            unique[key] = dict(candidate)
    ranked = sorted(unique.values(), key=_rank)
    allowance = policy.allowance(capacity)
    full = [candidate for candidate in ranked if candidate.get("full_width")]
    narrow = [candidate for candidate in ranked if not candidate.get("full_width")]
    return sorted(
        (full[:allowance] + narrow[: capacity - allowance])[:capacity], key=_rank
    )


def _rank(candidate: dict[str, Any]) -> tuple[float, str, int]:
    return (
        -float(candidate["score"]),
        str(candidate["example_id"]),
        int(candidate["position"]),
    )
