"""Strict, deterministic local source adapters for corpus v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.corpora.config import CorpusSourceSpec
from radjax_tome.corpora.records import SourceRecord


def iter_source_records(spec: CorpusSourceSpec) -> Iterator[SourceRecord]:
    """Yield source records without retaining the source in memory."""

    if spec.adapter == "local_text_tree_v1":
        yield from _iter_text_tree(spec)
    elif spec.adapter == "local_jsonl_text_v1":
        yield from _iter_jsonl(spec)
    else:  # defensive: config validation should have rejected this
        raise ValueError(f"unsupported source adapter: {spec.adapter}")


def _iter_text_tree(spec: CorpusSourceSpec) -> Iterator[SourceRecord]:
    root = spec.path
    if not root.exists():
        raise ValueError(f"source does not exist: {spec.source_id}")
    files = (
        (root,)
        if root.is_file()
        else tuple(path for path in root.rglob("*") if path.is_file())
    )
    candidates: list[Path] = []
    for path in files:
        if path.suffix.lower() not in {".txt", ".md", ".markdown", ".py"}:
            continue
        relative = _logical_path(path, root)
        if spec.include and not any(
            _matches(relative, pattern) for pattern in spec.include
        ):
            continue
        if any(_matches(relative, pattern) for pattern in spec.exclude):
            continue
        candidates.append(path)
    for path in sorted(candidates, key=lambda item: _logical_path(item, root)):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        yield SourceRecord(
            source_id=spec.source_id,
            source_ordinal=0,
            logical_locator=_logical_path(path, root),
            chunk_index=0,
            chunk_count=1,
            text=text,
            normalized_text_digest=_sha256(text.encode("utf-8")),
            source_digest=_sha256(raw),
        )


def _iter_jsonl(spec: CorpusSourceSpec) -> Iterator[SourceRecord]:
    if not spec.path.is_file():
        raise ValueError(f"JSONL source does not exist: {spec.source_id}")
    with spec.path.open("rb") as handle:
        source_digest = _sha256_file(spec.path)
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                line = raw_line.decode("utf-8", errors="strict")
                if not line.strip():
                    raise ValueError("blank JSONL record")
                item = json.loads(line, object_pairs_hook=_unique_pairs)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"source {spec.source_id} JSONL record {line_number} "
                    f"is invalid: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"source {spec.source_id} JSONL record {line_number} "
                    "must be an object"
                )
            text = item.get(spec.text_field)
            if not isinstance(text, str):
                raise ValueError(
                    f"source {spec.source_id} JSONL record {line_number} "
                    "text field must be a string"
                )
            record_id = item.get(spec.record_id_field) if spec.record_id_field else None
            if record_id is not None and not isinstance(record_id, str):
                raise ValueError(
                    f"source {spec.source_id} JSONL record {line_number} "
                    "ID field must be a string"
                )
            yield SourceRecord(
                source_id=spec.source_id,
                source_ordinal=0,
                logical_locator=f"{spec.path.name}#record-{line_number:09d}",
                chunk_index=0,
                chunk_count=1,
                text=text,
                normalized_text_digest=_sha256(text.encode("utf-8")),
                source_digest=source_digest,
                declared_record_id=record_id,
            )


def _logical_path(path: Path, root: Path) -> str:
    base = root if root.is_dir() else root.parent
    return PurePosixPath(path.relative_to(base).as_posix()).as_posix()


def _matches(value: str, pattern: str) -> bool:
    from fnmatch import fnmatch

    short_pattern = pattern[3:] if pattern.startswith("**/") else pattern
    return fnmatch(value, pattern) or fnmatch(value, short_pattern)


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


__all__ = ["iter_source_records"]
