from __future__ import annotations

from typing import Protocol

import numpy as np


class TeacherBackend(Protocol):
    backend_id: str
    vocab_size: int

    def emit_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Emit logits shaped [batch, sequence, vocab]."""
