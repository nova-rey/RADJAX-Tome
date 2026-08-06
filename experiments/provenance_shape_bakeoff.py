"""Private next-version provenance-shape construction bake-off.

This is intentionally not a Tome writer, package format, or public validator.
It consumes already-selected records and writes disposable audit projections.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import tarfile
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "radjax_tome_provenance_bakeoff_experimental_vnext"
RUNS = 3
MATERIAL_REDUCTION = 0.20
NOISE_MULTIPLIER = 2.0


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _configuration(shape: str, capacity: int) -> dict[str, Any]:
    """Return private construction settings that must not cross a resume."""
    return {
        "schema_version": SCHEMA + ".transaction.v1",
        "shape": shape,
        "capacity": capacity,
    }


def _member_path(root: Path, value: Any, *, label: str) -> Path:
    """Resolve an experimental public member without accepting path traversal."""
    if not isinstance(value, str):
        raise ValueError(f"invalid {label} pointer")
    member = Path(value)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"invalid {label} pointer")
    path = root / member
    if not path.is_file():
        raise ValueError(f"missing {label} member")
    return path


@dataclass
class Counters:
    serialization_calls: int = 0
    serialization_bytes: int = 0
    bytes_written: int = 0
    bytes_reread: int = 0
    bytes_rewritten: int = 0
    parse_calls: int = 0
    parsed_bytes: int = 0
    hash_calls: int = 0
    hashed_bytes: int = 0
    journal_operations: int = 0
    shard_seals: int = 0

    def canonical(self, value: Any) -> bytes:
        encoded = _canonical(value)
        self.serialization_calls += 1
        self.serialization_bytes += len(encoded)
        return encoded

    def digest(self, value: bytes) -> str:
        self.hash_calls += 1
        self.hashed_bytes += len(value)
        return _digest(value)

    def write(self, path: Path, value: bytes, *, rewrite: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        self.bytes_written += len(value)
        if rewrite:
            self.bytes_rewritten += len(value)

    def read(self, path: Path) -> bytes:
        value = path.read_bytes()
        self.bytes_reread += len(value)
        return value

    def parse(self, value: bytes) -> Any:
        self.parse_calls += 1
        self.parsed_bytes += len(value)
        return json.loads(value)

    def projection(self) -> dict[str, int]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class BakeoffResult:
    root: Path
    sequence_digest: str
    archive_digest: str
    counters: dict[str, int]
    wall_seconds: float
    peak_rss_bytes: int
    configuration: dict[str, Any]


def _logical_id(record: Mapping[str, Any], counter: Counters) -> str:
    return counter.digest(
        counter.canonical(
            {
                "selected_example_id": record["selected_example_id"],
                "selected_position": record["selected_position"],
            }
        )
    )


def _sequence(records: Iterable[Mapping[str, Any]], counter: Counters) -> str:
    """Hash full canonical records in an unambiguous ordered frame stream."""
    frames = [b"radjax-tome-selected-sequence-vnext\\x00"]
    for record in records:
        encoded = counter.canonical(dict(record))
        frames.extend((len(encoded).to_bytes(8, "big"), encoded))
    return counter.digest(b"".join(frames))


def build_projection(
    records: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    authority: Mapping[str, Any],
    capacity: int,
    shape: str,
) -> BakeoffResult:
    """Build a disposable current-model or candidate-model projection.

    ``current`` models the temporary per-record hash and duplicate final index
    fields. ``candidate`` replaces it with an authority-bound sealed-shard
    journal and writes each final shard/index field exactly once.
    """
    if shape not in {"current", "candidate"} or capacity < 1:
        raise ValueError("invalid private bake-off shape or capacity")
    started = time.perf_counter()
    counter = Counters()
    material = [dict(r) for r in records]
    authority_digest = counter.digest(counter.canonical(dict(authority)))
    sequence_digest = _sequence(material, counter)
    configuration = _configuration(shape, capacity)
    journal = {
        "schema_version": SCHEMA + ".journal.v1",
        "authority_digest": authority_digest,
        "configuration_digest": counter.digest(counter.canonical(configuration)),
        "state": "open",
        "sealed": [],
    }
    counter.journal_operations += 1
    counter.write(output / ".journal.json", counter.canonical(journal))
    index: list[dict[str, Any]] = []
    shard_entries: list[dict[str, Any]] = []
    for shard_id, first in enumerate(range(0, len(material), capacity)):
        chunk = material[first : first + capacity]
        lines: list[bytes] = []
        for row, record in enumerate(chunk):
            encoded = counter.canonical(record)
            record_digest = counter.digest(encoded)
            if shape == "current":
                native = {
                    "record": record,
                    "payload_hash": counter.digest(
                        counter.canonical({"record": record})
                    ),
                }
                native_path = output / ".native" / f"{first + row:05d}.json"
                counter.write(native_path, counter.canonical(native))
                # Model post-linkage reread/rehash/rewrite on this private copy.
                reread = counter.read(native_path)
                parsed = counter.parse(reread)
                parsed["payload_hash"] = counter.digest(
                    counter.canonical({"record": parsed["record"]})
                )
                counter.write(native_path, counter.canonical(parsed), rewrite=True)
            lines.append(encoded + b"\n")
            index.append(
                {
                    "logical_id": _logical_id(record, counter),
                    "selection_index": first + row,
                    "shard_id": shard_id,
                    "row": row,
                    "record_digest": record_digest,
                }
            )
        shard_path = output / "shards" / f"shard-{shard_id:05d}.jsonl"
        shard_bytes = b"".join(lines)
        counter.write(shard_path, shard_bytes)
        shard_hash = counter.digest(counter.read(shard_path))
        counter.shard_seals += 1
        entry = {
            "shard_id": shard_id,
            "path": shard_path.relative_to(output).as_posix(),
            "sha256": shard_hash,
            "size_bytes": len(shard_bytes),
            "first": first,
            "count": len(chunk),
        }
        shard_entries.append(entry)
        journal["sealed"].append(entry)
        counter.journal_operations += 1
        counter.write(
            output / ".journal.json", counter.canonical(journal), rewrite=True
        )
        for row in index[first : first + len(chunk)]:
            if shape == "current":
                row["payload_sha256"] = row["record_digest"]
                row["payload_semantic_digest"] = row["record_digest"]
                row["shard_sha256"] = shard_hash
    counter.write(
        output / "payload-index.jsonl",
        b"".join(counter.canonical(row) + b"\n" for row in index),
    )
    counter.write(
        output / "shard-index.jsonl",
        b"".join(counter.canonical(row) + b"\n" for row in shard_entries),
    )
    layout = {
        "schema_version": SCHEMA + ".public.v1",
        "shape": shape,
        "authority_digest": authority_digest,
        "sequence_digest": sequence_digest,
        "selected_count": len(material),
        "shard_index": {
            "path": "shard-index.jsonl",
            "sha256": counter.digest(counter.read(output / "shard-index.jsonl")),
        },
        "payload_index": {
            "path": "payload-index.jsonl",
            "sha256": counter.digest(counter.read(output / "payload-index.jsonl")),
        },
    }
    counter.write(output / "layout.json", counter.canonical(layout))
    inventory = []
    for path in sorted(
        p
        for p in output.rglob("*")
        if p.is_file() and not p.name.startswith(".journal")
    ):
        inventory.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": counter.digest(counter.read(path)),
                "size_bytes": path.stat().st_size,
            }
        )
    counter.write(output / "inventory.json", counter.canonical(inventory))
    header = {
        "schema_version": SCHEMA + ".manifest.v1",
        "inventory": "inventory.json",
        "inventory_sha256": counter.digest(counter.read(output / "inventory.json")),
        "sequence_digest": sequence_digest,
    }
    counter.write(output / "manifest-header.json", counter.canonical(header))
    cover = {
        "schema_version": SCHEMA + ".cover.v1",
        "authority_digest": authority_digest,
        "sequence_digest": sequence_digest,
        "manifest_header": {
            "path": "manifest-header.json",
            "sha256": counter.digest(counter.read(output / "manifest-header.json")),
        },
    }
    counter.write(output / "cover.json", counter.canonical(cover))
    journal["state"] = "complete"
    journal["promotion_marker"] = {
        "sequence_digest": sequence_digest,
        "shard_count": len(shard_entries),
    }
    counter.journal_operations += 1
    counter.write(output / ".journal.json", counter.canonical(journal), rewrite=True)
    archive = output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        for path in sorted(
            p for p in output.rglob("*") if p.is_file() and not p.name.startswith(".")
        ):
            tar.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    archive_digest = counter.digest(counter.read(archive))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if os.uname().sysname == "Darwin" else 1024
    )
    return BakeoffResult(
        output,
        sequence_digest,
        archive_digest,
        counter.projection(),
        time.perf_counter() - started,
        rss,
        configuration,
    )


def validate_candidate(
    root: Path,
    *,
    authority: Mapping[str, Any],
    configuration: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Validate the private projection before yielding any final-shard row.

    This is deliberately an experimental producer-side lifecycle validator, not
    a public or Student-facing Tome opener.  It checks the private journal only
    to exercise transaction and promotion safety for this bake-off.
    """
    counter = Counters()
    cover = counter.parse(counter.read(root / "cover.json"))
    header_path = _member_path(
        root, cover.get("manifest_header", {}).get("path"), label="manifest header"
    )
    header = counter.parse(counter.read(header_path))
    inventory_path = _member_path(root, header.get("inventory"), label="inventory")
    inventory = counter.parse(counter.read(inventory_path))
    journal = counter.parse(counter.read(root / ".journal.json"))
    authority_digest = counter.digest(counter.canonical(dict(authority)))
    if (
        journal.get("state") != "complete"
        or journal.get("authority_digest") != authority_digest
        or cover.get("authority_digest") != authority_digest
    ):
        raise ValueError("unsafe or cross-authority journal")
    if configuration is not None and journal.get(
        "configuration_digest"
    ) != counter.digest(counter.canonical(dict(configuration))):
        raise ValueError("stale transaction configuration")
    if cover["manifest_header"]["sha256"] != counter.digest(
        counter.read(header_path)
    ) or header["inventory_sha256"] != counter.digest(counter.read(inventory_path)):
        raise ValueError("cover or manifest mismatch")
    inventory_paths = {member["path"] for member in inventory}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".journal.json"
    }
    public_control_paths = {
        "cover.json",
        "manifest-header.json",
        "inventory.json",
    }
    if actual_paths - public_control_paths != inventory_paths:
        raise ValueError("partial or unreceipted public member")
    for member in inventory:
        path = root / member["path"]
        if not path.is_file() or member["sha256"] != counter.digest(counter.read(path)):
            raise ValueError("inventory member mismatch")
    layout = counter.parse(counter.read(root / "layout.json"))
    if (
        layout.get("authority_digest") != authority_digest
        or layout.get("sequence_digest") != cover.get("sequence_digest")
        or header.get("sequence_digest") != cover.get("sequence_digest")
    ):
        raise ValueError("cover, authority, or sequence incoherence")
    shards = [
        counter.parse(line)
        for line in counter.read(root / "shard-index.jsonl").splitlines()
    ]
    if (
        journal.get("sealed") != shards
        or journal.get("promotion_marker", {}).get("sequence_digest")
        != cover.get("sequence_digest")
        or journal.get("promotion_marker", {}).get("shard_count") != len(shards)
    ):
        raise ValueError("unreceipted shard or incomplete promotion")
    index_rows = [
        counter.parse(line)
        for line in counter.read(root / "payload-index.jsonl").splitlines()
    ]
    expected = 0
    observed: list[dict[str, Any]] = []
    for shard in shards:
        raw = counter.read(root / shard["path"])
        if shard["sha256"] != counter.digest(raw) or shard["first"] != expected:
            raise ValueError("unsealed, corrupt, or noncontiguous shard")
        rows = [counter.parse(line) for line in raw.splitlines()]
        if len(rows) != shard["count"]:
            raise ValueError("shard count mismatch")
        for row, record in enumerate(rows):
            try:
                index = index_rows[expected + row]
            except IndexError as exc:
                raise ValueError("missing payload index row") from exc
            expected_index = {
                "logical_id": _logical_id(record, counter),
                "selection_index": expected + row,
                "shard_id": shard["shard_id"],
                "row": row,
                "record_digest": counter.digest(counter.canonical(record)),
            }
            if any(index.get(key) != value for key, value in expected_index.items()):
                raise ValueError("payload index does not bind the shard row")
            observed.append(record)
        expected += len(rows)
        yield from rows
    if expected != layout.get("selected_count") or len(index_rows) != expected:
        raise ValueError("payload index count mismatch")
    if _sequence(observed, counter) != cover["sequence_digest"]:
        raise ValueError("sequence mismatch")


def validate_archive(archive: Path, *, expected_digest: str) -> None:
    """Check the experimental transport-level raw identity before extraction."""
    if _file_digest(archive) != expected_digest:
        raise ValueError("archive raw-integrity mismatch")


def summarize(values: list[float]) -> dict[str, float]:
    if len(values) != RUNS:
        raise ValueError("exactly three measurements required")
    ordered = sorted(values)
    return {"median": ordered[1], "spread": ordered[-1] - ordered[0]}


def materially_reduced(baseline: list[float], candidate: list[float]) -> bool:
    left, right = summarize(baseline), summarize(candidate)
    return left["median"] - right["median"] >= max(
        MATERIAL_REDUCTION * left["median"],
        NOISE_MULTIPLIER * ((left["spread"] ** 2 + right["spread"] ** 2) ** 0.5),
    )
