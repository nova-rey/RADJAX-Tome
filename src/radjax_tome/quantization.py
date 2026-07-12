from __future__ import annotations

from typing import Any

import numpy as np

ENTROPY_PARITY_QUANTIZATION_STEP = 0.00390625


def entropy_absolute_delta(left: Any, right: Any) -> float | None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return None
    return abs(left_value - right_value)


def entropy_parity_close(left: Any, right: Any) -> bool:
    delta = entropy_absolute_delta(left, right)
    return delta is not None and delta <= ENTROPY_PARITY_QUANTIZATION_STEP
