from __future__ import annotations

from radjax_tome.artifact_validation.long_tail import (  # noqa: F401
    LongTailPolicy,
    classify_long_tail,
    is_perverse_long_tail,
    long_tail_diagnostics,
    long_tail_summary,
    selected_board_for_long_tail,
    semantic_tail_tag,
)

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
