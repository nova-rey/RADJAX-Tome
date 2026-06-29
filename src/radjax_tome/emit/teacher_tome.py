from __future__ import annotations

from pathlib import Path

import numpy as np
from radjax_contract.artifacts import TeacherTomeManifest
from radjax_contract.io import write_json, write_jsonl
from radjax_contract.provenance import stable_hash
from radjax_contract.validation import validate_teacher_tome

from radjax_tome.backends.base import TeacherBackend


def emit_toy_teacher_tome(
    *,
    output_dir: str | Path,
    backend: TeacherBackend,
    records: list[dict[str, str]],
    sequence_length: int = 4,
) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    input_ids = _toy_tokenize(records, sequence_length=sequence_length)
    logits = backend.emit_logits(input_ids)
    np.save(target / "logits.npy", logits, allow_pickle=False)
    write_jsonl(target / "records.jsonl", records)
    manifest = TeacherTomeManifest(
        artifact_id=stable_hash(
            {
                "backend": backend.backend_id,
                "records": records,
                "sequence_length": sequence_length,
            }
        ),
        metadata={
            "backend_id": backend.backend_id,
            "num_examples": len(records),
            "sequence_length": sequence_length,
            "vocab_size": backend.vocab_size,
        },
    )
    write_json(target / "manifest.json", manifest.to_dict())
    validation = validate_teacher_tome(target)
    if not validation.ok:
        raise ValueError("emitted teacher tome failed contract validation")
    return target


def _toy_tokenize(records: list[dict[str, str]], *, sequence_length: int) -> np.ndarray:
    rows: list[list[int]] = []
    for record in records:
        seed = sum(ord(char) for char in record["text"])
        rows.append([(seed + offset) % 251 for offset in range(sequence_length)])
    return np.asarray(rows, dtype=np.int32)
