"""Simple compact-K body/metadata storage.

The body is immutable after its first write.  Linkage changes are confined to
the small JSON metadata record, so a consumer never needs to reread or hash a
large numeric payload merely to update provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import body_raw_digest, encode_compact_body_packed

from .modes import compact_body_from_logical_payload, compact_payload_for_storage


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def write_compact_body_store_from_compact(
    root: Path,
    payloads: Iterable[dict[str, Any]],
    *,
    profile: str = "compact_k_monolithic",
) -> dict[str, Any]:
    """Write each compact body once and return an authority-bound inventory."""

    root = Path(root)
    bodies = root / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for compact in payloads:
        body = compact_body_from_logical_payload(compact, profile=profile)
        encoded = encode_compact_body_packed(body)
        digest = body_raw_digest(encoded).hex()
        body_path = bodies / f"{digest}.body"
        if body_path.exists():
            if body_path.read_bytes() != encoded:
                raise ValueError("compact body digest collision")
        else:
            temporary = body_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, body_path)
        metadata.append(
            {
                "schema_version": "compact_exemplar_metadata_v1",
                "selected_example_id": str(compact["selected_example_id"]),
                "selected_position": int(compact["selected_position"]),
                "body_semantic_id": body.semantic_id.hex(),
                "body_raw_digest": digest,
                "body_size_bytes": len(encoded),
                "linkage": compact.get("linkage") or compact.get("mode_key"),
            }
        )
    data = b"".join(_json_bytes(item) + b"\n" for item in metadata)
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, metadata_path)
    return {
        "schema_version": "compact_body_store_v1",
        "body_count": len(metadata),
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "body_digests": [item["body_raw_digest"] for item in metadata],
    }


def write_compact_body_store(
    root: Path,
    payloads: Iterable[dict[str, Any]],
    *,
    profile: str = "compact_k_monolithic",
) -> dict[str, Any]:
    """Compatibility boundary for padded logical payloads.

    New canonical callers should use ``write_compact_body_store_from_compact``
    after preparing K-length records exactly once.
    """
    return write_compact_body_store_from_compact(
        root,
        (compact_payload_for_storage(payload) for payload in payloads),
        profile=profile,
    )


class _ByteBoundedQueue:
    """Small byte-bounded handoff queue for encoded private bodies."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._condition = threading.Condition()
        self._items: queue.Queue[tuple[dict[str, Any], bytes] | None] = queue.Queue()
        self._bytes = 0
        self.high_water_items = 0
        self.high_water_bytes = 0
        self.blocked_seconds = 0.0

    def put(self, item: tuple[dict[str, Any], bytes]) -> None:
        size = len(item[1])
        started = None
        with self._condition:
            while self._bytes and self._bytes + size > self.limit:
                if started is None:
                    started = time.perf_counter()
                self._condition.wait()
            self._bytes += size
            self._items.put(item)
            self.high_water_items = max(self.high_water_items, self._items.qsize())
            self.high_water_bytes = max(self.high_water_bytes, self._bytes)
            if started is not None:
                self.blocked_seconds += time.perf_counter() - started
            self._condition.notify_all()

    def get(self) -> tuple[dict[str, Any], bytes] | None:
        item = self._items.get()
        if item is not None:
            with self._condition:
                self._bytes -= len(item[1])
                self._condition.notify_all()
        return item

    def stop(self, workers: int) -> None:
        for _ in range(workers):
            self._items.put(None)


def write_compact_body_store_pipelined_from_compact(
    root: Path,
    payloads: Iterable[dict[str, Any]],
    *,
    profile: str = "compact_k_monolithic",
    worker_count: int = 2,
    queue_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Encode once, then write private bodies through a bounded conveyor."""

    if worker_count < 1 or worker_count > 4:
        raise ValueError("worker_count must be between 1 and 4")
    root = Path(root)
    bodies = root / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    handoff = _ByteBoundedQueue(queue_bytes)
    descriptors: list[dict[str, Any]] = []
    worker_metrics = [
        {
            "busy_seconds": 0.0,
            "written": 0,
            "bytes": 0,
            "body_write_seconds": 0.0,
            "durability_seconds": 0.0,
        }
        for _ in range(worker_count)
    ]
    stage_metrics = {
        "projection_seconds": 0.0,
        "encoding_seconds": 0.0,
        "hashing_seconds": 0.0,
        "final_drain_seconds": 0.0,
    }
    first_error: list[BaseException] = []

    def write_one(item: tuple[dict[str, Any], bytes], metric: dict[str, Any]) -> None:
        compact, encoded = item
        digest = body_raw_digest(encoded).hex()
        body_path = bodies / f"{digest}.body"
        started = time.perf_counter()
        if body_path.exists():
            if body_path.read_bytes() != encoded:
                raise ValueError("compact body digest collision")
        else:
            temporary = body_path.with_suffix(".tmp")
            write_started = time.perf_counter()
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                metric["body_write_seconds"] += time.perf_counter() - write_started
                durability_started = time.perf_counter()
                os.fsync(handle.fileno())
            metric["durability_seconds"] += time.perf_counter() - durability_started
            os.replace(temporary, body_path)
            metric["written"] += 1
            metric["bytes"] += len(encoded)
        metric["busy_seconds"] += time.perf_counter() - started

    def worker(index: int) -> None:
        try:
            while True:
                item = handoff.get()
                if item is None:
                    return
                write_one(item, worker_metrics[index])
        except BaseException as error:
            first_error.append(error)

    workers = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(worker_count)
    ]
    for worker_thread in workers:
        worker_thread.start()
    drain_started: float | None = None
    try:
        for compact in payloads:
            projection_started = time.perf_counter()
            body = compact_body_from_logical_payload(compact, profile=profile)
            stage_metrics["projection_seconds"] += (
                time.perf_counter() - projection_started
            )
            encoding_started = time.perf_counter()
            encoded = encode_compact_body_packed(body)
            stage_metrics["encoding_seconds"] += time.perf_counter() - encoding_started
            hashing_started = time.perf_counter()
            digest = body_raw_digest(encoded).hex()
            stage_metrics["hashing_seconds"] += time.perf_counter() - hashing_started
            descriptors.append(
                {
                    "schema_version": "compact_exemplar_metadata_v1",
                    "selected_example_id": str(compact["selected_example_id"]),
                    "selected_position": int(compact["selected_position"]),
                    "body_semantic_id": body.semantic_id.hex(),
                    "body_raw_digest": digest,
                    "body_size_bytes": len(encoded),
                    "linkage": compact.get("linkage") or compact.get("mode_key"),
                }
            )
            handoff.put((compact, encoded))
        drain_started = time.perf_counter()
        handoff.stop(worker_count)
    except BaseException as error:
        first_error.append(error)
        if drain_started is None:
            drain_started = time.perf_counter()
        handoff.stop(worker_count)
    for worker_thread in workers:
        worker_thread.join()
    stage_metrics["final_drain_seconds"] = (
        time.perf_counter() - drain_started if drain_started is not None else 0.0
    )
    if first_error:
        for path in bodies.glob("*.tmp"):
            path.unlink(missing_ok=True)
        raise first_error[0]
    data = b"".join(_json_bytes(item) + b"\n" for item in descriptors)
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, metadata_path)
    return {
        "schema_version": "compact_body_store_v1",
        "body_count": len(descriptors),
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "body_digests": [item["body_raw_digest"] for item in descriptors],
        "worker_count": worker_count,
        "queue_high_water_items": handoff.high_water_items,
        "queue_high_water_bytes": handoff.high_water_bytes,
        "producer_blocked_seconds": handoff.blocked_seconds,
        "persistent_bodies_written": sum(item["written"] for item in worker_metrics),
        "persistent_body_bytes": sum(item["bytes"] for item in worker_metrics),
        "body_reread_count": 0,
        "body_rewrite_count": 0,
        "projection_count": 0,
        "python_scalar_array_conversion_count": 0,
        "worker_busy_seconds": sum(item["busy_seconds"] for item in worker_metrics),
        **stage_metrics,
        "body_write_seconds": sum(
            item["body_write_seconds"] for item in worker_metrics
        ),
        "durability_seconds": sum(
            item["durability_seconds"] for item in worker_metrics
        ),
    }


def update_compact_linkage(
    root: Path,
    updates: dict[tuple[str, int], Any],
    *,
    counters: dict[str, int] | None = None,
) -> int:
    """Update metadata only; body files are never opened or rewritten."""

    path = Path(root) / "metadata.jsonl"
    raw = path.read_bytes()
    if counters is not None:
        counters.setdefault("source_payload_reads", 0)
        counters.setdefault("body_reads", 0)
        counters.setdefault("body_hashes", 0)
        counters.setdefault("body_rewrites", 0)
        counters["metadata_reads"] = counters.get("metadata_reads", 0) + 1
        counters["metadata_bytes_read"] = counters.get("metadata_bytes_read", 0) + len(
            raw
        )
    rows = [json.loads(line) for line in raw.splitlines() if line]
    changed = 0
    for row in rows:
        key = (row["selected_example_id"], int(row["selected_position"]))
        if key in updates:
            row["linkage"] = updates[key]
            changed += 1
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    if counters is not None:
        counters["metadata_writes"] = counters.get("metadata_writes", 0) + 1
        counters["metadata_bytes_written"] = counters.get(
            "metadata_bytes_written", 0
        ) + len(data)
    return changed
