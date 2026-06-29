from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FakeTeacherBackend:
    vocab_size: int = 8
    backend_id: str = "fake"

    def emit_logits(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.float32)
        vocab = np.arange(self.vocab_size, dtype=np.float32)[None, None, :]
        return np.sin(ids[:, :, None] * 0.17 + vocab * 0.23).astype(np.float32)
