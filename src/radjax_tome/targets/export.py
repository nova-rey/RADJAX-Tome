from __future__ import annotations

from pathlib import Path

from radjax_tome.backends.emission import emit_teacher_target_store
from radjax_tome.backends.synthetic import SyntheticTeacherBackend
from radjax_tome.targets.store import TeacherTargetStore


def export_synthetic_teacher_targets(
    path: str | Path,
    *,
    num_examples: int,
    sequence_length: int,
    vocab_size: int = 8,
    overwrite: bool = False,
) -> TeacherTargetStore:
    backend = SyntheticTeacherBackend(vocab_size=vocab_size)
    return emit_teacher_target_store(
        backend,
        path,
        num_examples=num_examples,
        sequence_length=sequence_length,
        overwrite=overwrite,
    )
