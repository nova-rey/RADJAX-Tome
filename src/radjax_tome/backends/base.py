from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from radjax_tome.targets import TargetStoreMetadata


class TeacherBackend(Protocol):
    backend_id: str
    vocab_size: int

    def emit_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Emit logits shaped [batch, sequence, vocab]."""


class TeacherTargetEmitter(Protocol):
    name: str

    def build_metadata(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> TargetStoreMetadata: ...

    def emit_targets(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> Mapping[str, np.ndarray]: ...
