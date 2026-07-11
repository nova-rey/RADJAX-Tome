from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_LONG_TAIL_WARNING_K = 8_192
DEFAULT_VERY_LONG_TAIL_WARNING_K = 32_768
DEFAULT_PERVERSE_TAIL_WARNING_K = 65_536

NORMAL = "normal"
LONG_TAIL = "long_tail"
VERY_LONG_TAIL = "very_long_tail"
SUSPICIOUS_FLAT = "suspicious_flat"
FULL_VOCAB_OR_NEAR_FULL_VOCAB = "full_vocab_or_near_full_vocab"

PROPER_NAME_TAIL = "proper_name_tail"
NUMERIC_TAIL = "numeric_tail"
URL_OR_MARKUP_TAIL = "url_or_markup_tail"
WHITESPACE_OR_BOUNDARY_TAIL = "whitespace_or_boundary_tail"
CODE_OR_SYMBOL_TAIL = "code_or_symbol_tail"
UNKNOWN_OPEN_CLASS_TAIL = "unknown_open_class_tail"
TRUE_FLAT_OR_UNCLASSIFIED_TAIL = "true_flat_or_unclassified_tail"

LONG_TAIL_UNCERTAINTY_BOARD = "long_tail_uncertainty"
PERVERSE_TAIL_DIAGNOSTIC_BOARD = "perverse_tail_diagnostic"
PRIMARY_SELECTED_BOARD = "primary"


@dataclass(frozen=True)
class LongTailPolicy:
    long_tail_warning_k: int = DEFAULT_LONG_TAIL_WARNING_K
    very_long_tail_warning_k: int = DEFAULT_VERY_LONG_TAIL_WARNING_K
    perverse_tail_warning_k: int = DEFAULT_PERVERSE_TAIL_WARNING_K
    reject_perverse_exemplars: bool = False

    def __post_init__(self) -> None:
        if self.long_tail_warning_k < 1:
            raise ValueError("long_tail_warning_k must be >= 1")
        if self.very_long_tail_warning_k < self.long_tail_warning_k:
            raise ValueError("very_long_tail_warning_k must be >= long_tail_warning_k")
        if self.perverse_tail_warning_k < self.very_long_tail_warning_k:
            raise ValueError(
                "perverse_tail_warning_k must be >= very_long_tail_warning_k"
            )


def long_tail_diagnostics(
    *,
    effective_top_k: int,
    top_mass: float,
    vocab_size: int,
    dynamic_mass_threshold: float,
    dynamic_top_k_max: int,
    policy: LongTailPolicy,
) -> dict[str, Any]:
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    if effective_top_k < 1:
        raise ValueError("effective_top_k must be >= 1")
    long_tail_class = classify_long_tail(
        effective_top_k=effective_top_k,
        vocab_size=vocab_size,
        policy=policy,
    )
    effective_cap = min(dynamic_top_k_max, vocab_size)
    saturated = effective_top_k >= effective_cap
    warnings = _warnings(
        long_tail_class=long_tail_class,
        effective_top_k=effective_top_k,
        dynamic_top_k_max=dynamic_top_k_max,
        saturated=saturated,
    )
    raw_top_mass = float(top_mass)
    reported_top_mass = min(max(raw_top_mass, 0.0), 1.0)
    return {
        "effective_top_k": effective_top_k,
        "top_mass": reported_top_mass,
        "raw_top_mass": raw_top_mass,
        "top_mass_clamped": reported_top_mass != raw_top_mass,
        "dynamic_mass_threshold": float(dynamic_mass_threshold),
        "dynamic_top_k_max": int(dynamic_top_k_max),
        "top_k_saturated": saturated,
        "long_tail_class": long_tail_class,
        "long_tail_warnings": warnings,
        "effective_top_k_fraction_of_vocab": float(effective_top_k) / float(vocab_size),
    }


def classify_long_tail(
    *,
    effective_top_k: int,
    vocab_size: int,
    policy: LongTailPolicy,
) -> str:
    if effective_top_k >= vocab_size:
        return FULL_VOCAB_OR_NEAR_FULL_VOCAB
    if effective_top_k >= policy.perverse_tail_warning_k:
        return SUSPICIOUS_FLAT
    if effective_top_k >= policy.very_long_tail_warning_k:
        return VERY_LONG_TAIL
    if effective_top_k >= policy.long_tail_warning_k:
        return LONG_TAIL
    return NORMAL


def is_perverse_long_tail(diagnostic: dict[str, Any]) -> bool:
    return diagnostic.get("long_tail_class") in {
        SUSPICIOUS_FLAT,
        FULL_VOCAB_OR_NEAR_FULL_VOCAB,
    }


def selected_board_for_long_tail(
    long_tail_class: str,
    *,
    include_long_tail_in_primary: bool = False,
    include_perverse_tail_in_primary: bool = False,
) -> str:
    if long_tail_class == NORMAL:
        return PRIMARY_SELECTED_BOARD
    if long_tail_class in {LONG_TAIL, VERY_LONG_TAIL}:
        return (
            PRIMARY_SELECTED_BOARD
            if include_long_tail_in_primary
            else LONG_TAIL_UNCERTAINTY_BOARD
        )
    if long_tail_class in {SUSPICIOUS_FLAT, FULL_VOCAB_OR_NEAR_FULL_VOCAB}:
        return (
            PRIMARY_SELECTED_BOARD
            if include_perverse_tail_in_primary
            else PERVERSE_TAIL_DIAGNOSTIC_BOARD
        )
    raise ValueError(f"unsupported long_tail_class: {long_tail_class}")


def semantic_tail_tag(
    *,
    long_tail_class: str,
    top_token_texts: Iterable[str] = (),
) -> str:
    """Classify decoded top-token text when it is available to the producer.

    Current compact artifacts intentionally retain token ids rather than a second
    decoded-token surface. Callers without decoded pieces get a conservative
    class-derived fallback instead of inventing text semantics from token ids.
    """

    pieces = [
        str(value).replace("▁", " ").replace("Ġ", " ").strip()
        for value in top_token_texts
    ]
    pieces = [piece for piece in pieces if piece]
    if pieces:
        labels = [_piece_tail_tag(piece) for piece in pieces]
        for tag in (
            PROPER_NAME_TAIL,
            NUMERIC_TAIL,
            URL_OR_MARKUP_TAIL,
            WHITESPACE_OR_BOUNDARY_TAIL,
            CODE_OR_SYMBOL_TAIL,
        ):
            if sum(label == tag for label in labels) * 2 >= len(labels):
                return tag
    if long_tail_class in {SUSPICIOUS_FLAT, FULL_VOCAB_OR_NEAR_FULL_VOCAB}:
        return TRUE_FLAT_OR_UNCLASSIFIED_TAIL
    return UNKNOWN_OPEN_CLASS_TAIL


def _piece_tail_tag(piece: str) -> str:
    if piece.startswith(("http", "www.", "<", "</", "#")) or any(
        marker in piece for marker in ("://", "=", "/")
    ):
        return URL_OR_MARKUP_TAIL
    if all(character.isdigit() or character in ".,:%+-/" for character in piece):
        return NUMERIC_TAIL
    if not piece.strip():
        return WHITESPACE_OR_BOUNDARY_TAIL
    if any(character in "{}[]()<>_=;`\\|~" for character in piece):
        return CODE_OR_SYMBOL_TAIL
    if (
        (len(piece) == 1 and piece.isupper())
        or (piece[:1].isupper() and piece[1:].isalpha())
        or piece.startswith(("Mc", "Mac", "O'", "van", "de"))
    ):
        return PROPER_NAME_TAIL
    return UNKNOWN_OPEN_CLASS_TAIL


def long_tail_summary(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(items)
    classes = (
        NORMAL,
        LONG_TAIL,
        VERY_LONG_TAIL,
        SUSPICIOUS_FLAT,
        FULL_VOCAB_OR_NEAR_FULL_VOCAB,
    )
    counts = {
        value: sum(item.get("long_tail_class") == value for item in records)
        for value in classes
    }
    effective = [int(item.get("effective_top_k") or 0) for item in records]
    fractions = [
        float(item.get("effective_top_k_fraction_of_vocab") or 0.0) for item in records
    ]
    return {
        "count": len(records),
        "normal_count": counts[NORMAL],
        "long_tail_count": counts[LONG_TAIL],
        "very_long_tail_count": counts[VERY_LONG_TAIL],
        "suspicious_flat_count": counts[SUSPICIOUS_FLAT],
        "full_vocab_or_near_full_vocab_count": counts[FULL_VOCAB_OR_NEAR_FULL_VOCAB],
        "saturated_count": sum(bool(item.get("top_k_saturated")) for item in records),
        "max_effective_top_k": max(effective, default=0),
        "mean_effective_top_k": (
            float(sum(effective)) / float(len(effective)) if effective else 0.0
        ),
        "max_effective_top_k_fraction_of_vocab": max(fractions, default=0.0),
    }


def _warnings(
    *,
    long_tail_class: str,
    effective_top_k: int,
    dynamic_top_k_max: int,
    saturated: bool,
) -> list[str]:
    warnings: list[str] = []
    if long_tail_class != NORMAL:
        warnings.append(f"{long_tail_class}: effective_top_k={effective_top_k}")
    if saturated:
        warnings.append(
            "dynamic_top_k_max reached: "
            f"effective_top_k={effective_top_k}, dynamic_top_k_max="
            f"{dynamic_top_k_max}"
        )
    return warnings
