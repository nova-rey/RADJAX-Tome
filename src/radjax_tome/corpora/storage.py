"""Canonical shard and offset-index storage for corpus v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import canonical_bytes
from radjax_tome.corpora.records import CanonicalCorpusRecord

SHARDS_DIR = "shards"
INDEXES_DIR = "indexes"


def write_shards(
    root: Path,
    records: Iterable[CanonicalCorpusRecord],
    *,
    shard_capacity: int = 128,
    max_shard_bytes: int | None = None,
) -> list[dict[str, Any]]:
    if shard_capacity < 1:
        raise ValueError("shard capacity must be positive")
    if max_shard_bytes is not None and max_shard_bytes < 1:
        raise ValueError("max shard bytes must be positive")
    shards = root / SHARDS_DIR
    indexes = root / INDEXES_DIR
    shards.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    batch: list[CanonicalCorpusRecord] = []

    def flush(items: list[CanonicalCorpusRecord], shard_number: int) -> None:
        if not items:
            return
        shard_name = f"records-{shard_number:05d}.jsonl"
        index_name = f"records-{shard_number:05d}.index.jsonl"
        shard_path = shards / shard_name
        index_path = indexes / index_name
        offset = 0
        index_digest = hashlib.sha256()
        with shard_path.open("wb") as handle:
            with index_path.open("wb") as index_handle:
                for row_number, record in enumerate(items):
                    encoded = canonical_bytes(record.to_dict()) + b"\n"
                    if (
                        max_shard_bytes is not None
                        and offset + len(encoded) > max_shard_bytes
                    ):
                        raise ValueError("shard exceeds resources.max_shard_bytes")
                    handle.write(encoded)
                    index_line = (
                        canonical_bytes(
                            {
                                "example_id": record.example_id,
                                "row": row_number,
                                "offset": offset,
                                "length": len(encoded),
                            }
                        )
                        + b"\n"
                    )
                    index_handle.write(index_line)
                    index_digest.update(index_line)
                    offset += len(encoded)
        inventory.append(
            {
                "shard_id": shard_number,
                "shard": f"{SHARDS_DIR}/{shard_name}",
                "index": f"{INDEXES_DIR}/{index_name}",
                "record_count": len(items),
                "first_example_id": items[0].example_id,
                "last_example_id": items[-1].example_id,
                "raw_sha256": _file_digest(shard_path),
                "index_sha256": "sha256:" + index_digest.hexdigest(),
                "size_bytes": shard_path.stat().st_size,
            }
        )

    for record in records:
        batch.append(record)
        if len(batch) == shard_capacity:
            flush(batch, len(inventory))
            batch = []
    flush(batch, len(inventory))
    return inventory


class VerifiedCorpusReader:
    """Verify each complete shard/index pair before yielding its first row."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._inventory = _read_json(self.root / "shard_inventory.json")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for item in self._inventory:
            shard_path = _safe_member(self.root, str(item["shard"]))
            index_path = _safe_member(self.root, str(item["index"]))
            _verify_digest(shard_path, item["raw_sha256"])
            _verify_digest(index_path, item["index_sha256"])
            count = 0
            with (
                index_path.open("rb") as index_handle,
                shard_path.open("rb") as shard_handle,
            ):
                expected_offset = 0
                for raw_index in index_handle:
                    index_row = json.loads(raw_index)
                    count += 1
                    offset = int(index_row["offset"])
                    length = int(index_row["length"])
                    if (
                        int(index_row.get("row", -1)) != count - 1
                        or offset != expected_offset
                    ):
                        raise ValueError(
                            f"index offsets are not contiguous: {shard_path.name}"
                        )
                    shard_handle.seek(offset)
                    encoded = shard_handle.read(length)
                    if len(encoded) != length or not encoded.endswith(b"\n"):
                        raise ValueError(f"index range mismatch: {shard_path.name}")
                    row = json.loads(encoded)
                    if row.get("example_id") != index_row.get("example_id"):
                        raise ValueError(f"index identity mismatch: {shard_path.name}")
                    yield row
                    expected_offset = offset + length
            if count != int(item["record_count"]):
                raise ValueError(f"index count mismatch: {shard_path.name}")
            if expected_offset != shard_path.stat().st_size:
                raise ValueError(f"index does not cover shard: {shard_path.name}")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing corpus member: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_digest(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"corpus member digest mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if "sha256:" + digest.hexdigest() != expected:
        raise ValueError(f"corpus member digest mismatch: {path.name}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_member(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"corpus member escapes artifact root: {relative}")
    return candidate


__all__ = ["VerifiedCorpusReader", "write_json", "write_shards"]
