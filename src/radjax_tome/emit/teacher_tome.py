from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from radjax_contract.io import write_json, write_jsonl
from radjax_contract.provenance import stable_hash
from radjax_contract.validation import validate_teacher_tome

from radjax_tome.backends.base import TeacherBackend

try:
    from radjax_contract.tome import (
        TomeBehavioralSummary,
        TomeCompression,
        TomeContentsSummary,
        TomeCorpusSource,
        TomeCorpusSummary,
        TomeCoverPage,
        TomeManifest,
        TomePayloadFormat,
        TomeRole,
        TomeSplitSummary,
        TomeStudentConsumptionSummary,
        TomeTeacherSummary,
    )
    from radjax_contract.vocab import VocabContract
except ImportError:  # pragma: no cover - compatibility with older Contract installs.
    from radjax_contract.artifacts import TeacherTomeManifest

    TomeBehavioralSummary = None
    TomeCompression = None
    TomeContentsSummary = None
    TomeCorpusSource = None
    TomeCorpusSummary = None
    TomeCoverPage = None
    TomeManifest = None
    TomePayloadFormat = None
    TomeRole = None
    TomeSplitSummary = None
    TomeStudentConsumptionSummary = None
    TomeTeacherSummary = None
    VocabContract = None
else:
    TeacherTomeManifest = None


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
    manifest = _build_manifest(
        backend=backend,
        records=records,
        sequence_length=sequence_length,
    )
    write_json(target / "manifest.json", manifest.to_dict())
    cover_page = _build_cover_page(
        backend=backend,
        records=records,
        sequence_length=sequence_length,
        logits=logits,
    )
    if cover_page is not None:
        write_json(target / "cover_page.json", cover_page.to_dict())
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


def _build_manifest(
    *,
    backend: TeacherBackend,
    records: list[dict[str, str]],
    sequence_length: int,
) -> Any:
    artifact_id = stable_hash(
        {
            "backend": backend.backend_id,
            "records": records,
            "sequence_length": sequence_length,
        }
    )
    if TomeManifest is None:
        return TeacherTomeManifest(
            artifact_id=artifact_id,
            metadata={
                "backend_id": backend.backend_id,
                "num_examples": len(records),
                "sequence_length": sequence_length,
                "vocab_size": backend.vocab_size,
            },
        )
    return TomeManifest(
        artifact_id=artifact_id,
        payload_format=TomePayloadFormat.DENSE_LOGITS_V0,
        compression=TomeCompression(),
        vocab_contract=VocabContract(
            tokenizer_id=f"{backend.backend_id}-toy-tokenizer",
            vocab_size=backend.vocab_size,
            model_id=backend.backend_id,
            model_family="fake",
        ),
        record_count=len(records),
        sequence_length=sequence_length,
        metadata={
            "backend_id": backend.backend_id,
            "payload_path": "logits.npy",
        },
    )


def _build_cover_page(
    *,
    backend: TeacherBackend,
    records: list[dict[str, str]],
    sequence_length: int,
    logits: np.ndarray,
) -> Any | None:
    if TomeCoverPage is None:
        return None
    return TomeCoverPage(
        title="Tiny dense logits smoke Tome",
        description="Dense teacher-output Tome generated from tiny examples.",
        teacher=TomeTeacherSummary(
            teacher_id=backend.backend_id,
            teacher_family="fake",
            backend=backend.backend_id,
            teacher_dtype=str(logits.dtype),
            teacher_vocab_size=backend.vocab_size,
        ),
        corpus=TomeCorpusSummary(
            summary="Tiny synthetic smoke corpus.",
            sources=(
                TomeCorpusSource(
                    source_id="toy_smoke",
                    source_type="synthetic",
                    description="Small checked test fixture corpus.",
                    record_count=len(records),
                ),
            ),
            contains_synthetic_examples=True,
        ),
        contents=TomeContentsSummary(
            role=TomeRole.TRAINING,
            record_count=len(records),
            sequence_length=sequence_length,
            payload_format=TomePayloadFormat.DENSE_LOGITS_V0,
            compression=TomeCompression(),
        ),
        behavioral_fingerprint=TomeBehavioralSummary(),
        splits=TomeSplitSummary(split_role=TomeRole.TRAINING),
        student_consumption=TomeStudentConsumptionSummary(
            expected_adapter="dense_logits",
            implemented_by_contract=True,
            notes="Toy dense logits smoke artifact.",
        ),
    )
