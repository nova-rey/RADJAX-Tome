from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.io.jsonl import read_jsonl_objects


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class SourceTextRecord:
    example_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SourceTextRecord:
        example_id = payload.get("example_id", payload.get("id", ""))
        return cls(
            example_id=str(example_id),
            text=str(payload.get("text", "")),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key not in {"example_id", "id", "text"}
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {"example_id": self.example_id, "text": self.text, **self.metadata}


@dataclass(frozen=True)
class SourceLineageRecord:
    source_file: str
    source_file_sha256: str
    ordered_example_ids: tuple[str, ...]
    ordered_example_ids_sha256: str
    source_example_set_sha256: str
    ordered_source_text_sha256: str
    tokenized_inputs_sha256: str | None
    artifact_manifest_sha256: str | None
    source_join_kind: str
    source_join_complete: bool
    lineage_confidence: str
    publication_grade_lineage: bool
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ordered_example_ids"] = list(self.ordered_example_ids)
        payload["warnings"] = list(self.warnings)
        return payload


def read_source_text_records(path: str | Path) -> tuple[SourceTextRecord, ...]:
    return tuple(
        SourceTextRecord.from_payload(row) for row in read_jsonl_objects(Path(path))
    )


def join_sources_by_example_id(
    source_file: str | Path,
    artifact_example_ids: tuple[str, ...],
    *,
    allow_legacy_positional_source_join: bool = False,
) -> dict[str, Any]:
    rows = read_source_text_records(source_file)
    if not rows:
        raise ValueError("source file contains zero text records")
    explicit_ids = [row.example_id for row in rows]
    if all(value.strip() for value in explicit_ids):
        source_by_id: dict[str, str] = {}
        for row in rows:
            if row.example_id in source_by_id:
                raise ValueError(f"duplicate source example_id: {row.example_id}")
            source_by_id[row.example_id] = row.text.strip()
        missing = [
            example_id
            for example_id in artifact_example_ids
            if example_id not in source_by_id
        ]
        if missing:
            raise ValueError(
                "source file missing artifact example IDs: " + ", ".join(missing)
            )
        return {
            "source_join_kind": "example_id",
            "source_join_complete": True,
            "lineage_confidence": "full",
            "publication_grade_lineage": True,
            "ordered_source_texts": [
                source_by_id[example_id] for example_id in artifact_example_ids
            ],
            "warnings": [],
        }

    if not allow_legacy_positional_source_join:
        raise ValueError(
            "source rows require explicit example_id; pass "
            "allow_legacy_positional_source_join only for legacy fixtures"
        )
    if any(value.strip() for value in explicit_ids):
        raise ValueError(
            "source rows must either all include example_id or all omit it"
        )
    if len(rows) < len(artifact_example_ids):
        raise ValueError("legacy source file has fewer rows than artifact examples")
    return {
        "source_join_kind": "legacy_positional",
        "source_join_complete": True,
        "lineage_confidence": "reduced",
        "publication_grade_lineage": False,
        "ordered_source_texts": [
            row.text.strip() for row in rows[: len(artifact_example_ids)]
        ],
        "warnings": [
            "legacy positional source join enabled; lineage confidence reduced"
        ],
    }


def build_source_lineage(
    source_file: str | Path,
    artifact_example_ids: tuple[str, ...],
    *,
    token_sequences: tuple[tuple[int, ...], ...] = (),
    artifact_manifest_path: str | Path | None = None,
    allow_legacy_positional_source_join: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SourceLineageRecord:
    source_join = join_sources_by_example_id(
        source_file,
        artifact_example_ids,
        allow_legacy_positional_source_join=allow_legacy_positional_source_join,
    )
    source_text_hashes = [
        stable_hash(text) for text in source_join["ordered_source_texts"]
    ]
    token_sequence_hashes = [stable_hash(sequence) for sequence in token_sequences]
    return SourceLineageRecord(
        source_file=str(source_file),
        source_file_sha256=file_sha256(source_file),
        ordered_example_ids=artifact_example_ids,
        ordered_example_ids_sha256=stable_hash(artifact_example_ids),
        source_example_set_sha256=stable_hash(sorted(artifact_example_ids)),
        ordered_source_text_sha256=stable_hash(source_text_hashes),
        tokenized_inputs_sha256=(
            stable_hash(token_sequence_hashes) if token_sequence_hashes else None
        ),
        artifact_manifest_sha256=(
            file_sha256(artifact_manifest_path) if artifact_manifest_path else None
        ),
        source_join_kind=str(source_join["source_join_kind"]),
        source_join_complete=bool(source_join["source_join_complete"]),
        lineage_confidence=str(source_join["lineage_confidence"]),
        publication_grade_lineage=bool(source_join["publication_grade_lineage"]),
        warnings=tuple(str(item) for item in source_join["warnings"]),
        metadata=dict(metadata or {}),
    )


def build_artifact_source_lineage(
    artifact_dir: str | Path,
    source_file: str | Path,
    *,
    allow_legacy_positional_source_join: bool = False,
) -> SourceLineageRecord:
    root = Path(artifact_dir)
    manifest = read_json_object(root / "manifest.json")
    examples = _ordered_artifact_examples(root, manifest)
    return build_source_lineage(
        source_file,
        tuple(example_id for example_id, _tokens in examples),
        token_sequences=tuple(tokens for _example_id, tokens in examples),
        artifact_manifest_path=root / "manifest.json",
        allow_legacy_positional_source_join=allow_legacy_positional_source_join,
        metadata={
            "capture_summary_sha256": (
                file_sha256(root / "capture_summary.json")
                if (root / "capture_summary.json").is_file()
                else None
            ),
            "teacher_identity_sha256": stable_hash(manifest.get("teacher", {})),
        },
    )


def write_source_lineage(path: str | Path, lineage: SourceLineageRecord) -> Path:
    output = Path(path)
    write_json(output, lineage.to_dict())
    return output


def _ordered_artifact_examples(
    artifact_dir: Path, manifest: dict[str, Any]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    output: dict[str, tuple[int, ...]] = {}
    for shard in manifest.get("target_shards", ()):
        shard_path = artifact_dir / str(shard.get("path", ""))
        for line_number, row in enumerate(read_jsonl_objects(shard_path), start=1):
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise ValueError(f"{shard_path}:{line_number} missing example_id")
            input_ids = tuple(int(item) for item in row.get("input_ids", ()))
            prior = output.setdefault(example_id, input_ids)
            if prior != input_ids:
                raise ValueError(f"inconsistent tokenized inputs for {example_id}")
    if not output:
        raise ValueError("fingerprint artifact contains zero source examples")
    return tuple(output.items())
