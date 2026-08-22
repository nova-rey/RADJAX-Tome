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

import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import (body_raw_digest, compact_body_from_buffers, encode_compact_body_packed, encode_compact_body_packed_from_buffers)

from .modes import compact_body_from_logical_payload, compact_payload_for_storage


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _buffer_payload(compact: dict[str, Any]) -> dict[str, Any]:
    result = dict(compact)
    result["top_token_ids"] = np.asarray(compact["top_token_ids"], dtype="<u4")
    result["top_probs"] = np.asarray(compact["top_probs"], dtype="<f4")
    result["top_log_probs"] = np.asarray(compact["top_log_probs"], dtype="<f4")
    result["bucket_masses"] = np.asarray(compact["bucket_masses"], dtype="<f4")
    return result


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
        compact = _buffer_payload(compact)
        body = compact_body_from_buffers(
            profile=profile, vocab_size=int(compact["vocab_size"]),
            num_buckets=int(compact["num_buckets"]), top_token_ids=compact["top_token_ids"],
            top_probs=compact["top_probs"], top_log_probs=compact["top_log_probs"],
            effective_top_k=int(compact["effective_top_k"]), top_mass=float(compact["top_mass"]),
            tail_mass=float(compact["tail_mass"]), bucket_masses=compact["bucket_masses"],
        )
        encoded = encode_compact_body_packed_from_buffers(body)
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
                "body_semantic_id": digest,
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


@dataclass
class RawCompactDescriptor:
    payload: dict[str, Any]
    estimated_bytes: int
    ordinal: int = 0
    ready_event: Any | None = None
    release: Any | None = None

    def wait_ready(self) -> None:
        if self.ready_event is not None:
            method = getattr(self.ready_event, "synchronize", None)
            if method is not None:
                method()

    def release_owner(self) -> None:
        if self.release is not None:
            self.release()


def _raw_descriptor(payload: dict[str, Any], ordinal: int = 0) -> RawCompactDescriptor:
    estimate = 512
    for key in ("top_token_ids", "top_probs", "top_log_probs"):
        value = payload.get(key)
        nbytes = getattr(value, "nbytes", None)
        estimate += int(nbytes) if nbytes is not None else len(value or ()) * 8
    return RawCompactDescriptor(payload, estimate, ordinal)


class _ByteBoundedQueue:
    """Bounded queue of raw exact-K descriptors, not encoded bodies."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._condition = threading.Condition()
        self._items: queue.Queue[RawCompactDescriptor | None] = queue.Queue()
        self._bytes = 0
        self._oversized = False
        self._failure: BaseException | None = None
        self.high_water_items = 0
        self.high_water_bytes = 0
        self.blocked_seconds = 0.0
        self.oversized_admissions = 0

    def put(self, item: RawCompactDescriptor) -> None:
        size = max(1, int(item.estimated_bytes))
        oversized = size > self.limit
        started = None
        with self._condition:
            while self._oversized or self._bytes + size > self.limit:
                if oversized and self._bytes == 0 and not self._oversized:
                    break
                if self._failure is not None:
                    raise RuntimeError(
                        "compact body pipeline worker failed"
                    ) from self._failure
                if started is None:
                    started = time.perf_counter()
                self._condition.wait()
            self._bytes += size
            self._oversized = oversized
            if oversized:
                self.oversized_admissions += 1
            self._items.put(item)
            self.high_water_items = max(self.high_water_items, self._items.qsize())
            self.high_water_bytes = max(self.high_water_bytes, self._bytes)
            if started is not None:
                self.blocked_seconds += time.perf_counter() - started
            self._condition.notify_all()

    def get(self) -> RawCompactDescriptor | None:
        return self._items.get()

    def release(self, size: int) -> None:
        with self._condition:
            self._bytes -= int(size)
            if self._bytes <= 0:
                self._bytes = 0
                self._oversized = False
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            self._failure = error
            self._condition.notify_all()

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
    """Encode/hash/write raw descriptors only after queue admission."""
    if worker_count < 1 or worker_count > 4:
        raise ValueError("worker_count must be between 1 and 4")
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"compact body destination already exists: {root}")
    staging = root.with_name(root.name + ".staging")
    if staging.exists():
        raise FileExistsError(
            f"compact body staging destination already exists: {staging}"
        )
    bodies = staging / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    metadata_path = staging / "metadata.jsonl"
    handoff = _ByteBoundedQueue(queue_bytes)
    results: list[dict[str, Any] | None] = []
    metrics = [
        {
            "busy_seconds": 0.0,
            "written": 0,
            "bytes": 0,
            "encoding_seconds": 0.0,
            "hashing_seconds": 0.0,
            "projection_seconds": 0.0,
            "durability_seconds": 0.0,
        }
        for _ in range(worker_count)
    ]
    first_error: list[BaseException] = []
    digest_lock = threading.Lock()

    def worker(index: int) -> None:
        metric = metrics[index]
        try:
            while True:
                descriptor = handoff.get()
                if descriptor is None:
                    return
                try:
                    descriptor.wait_ready()
                    t = time.perf_counter()
                    compact = descriptor.payload
                    body = compact_body_from_buffers(
                        profile=profile, vocab_size=int(compact["vocab_size"]),
                        num_buckets=int(compact["num_buckets"]), top_token_ids=compact["top_token_ids"],
                        top_probs=compact["top_probs"], top_log_probs=compact["top_log_probs"],
                        effective_top_k=int(compact["effective_top_k"]), top_mass=float(compact["top_mass"]),
                        tail_mass=float(compact["tail_mass"]), bucket_masses=compact["bucket_masses"],
                    )
                    metric["projection_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    encoded = encode_compact_body_packed_from_buffers(body)
                    metric["encoding_seconds"] += time.perf_counter() - t
                    t = time.perf_counter()
                    digest = body_raw_digest(encoded).hex()
                    metric["hashing_seconds"] += time.perf_counter() - t
                    body_path = bodies / f"{digest}.body"
                    t = time.perf_counter()
                    with digest_lock:
                        if body_path.exists():
                            if body_path.read_bytes() != encoded:
                                raise ValueError("compact body digest collision")
                        else:
                            temporary = body_path.with_suffix(
                                f".{threading.get_ident()}.tmp"
                            )
                            with temporary.open("wb") as handle:
                                handle.write(encoded)
                                handle.flush()
                                t2 = time.perf_counter()
                                os.fsync(handle.fileno())
                                metric["durability_seconds"] += time.perf_counter() - t2
                            os.replace(temporary, body_path)
                            metric["written"] += 1
                            metric["bytes"] += len(encoded)
                    metric["busy_seconds"] += time.perf_counter() - t
                    results[descriptor.ordinal] = {
                        "schema_version": "compact_exemplar_metadata_v1",
                        "selected_example_id": str(
                            descriptor.payload["selected_example_id"]
                        ),
                        "selected_position": int(
                            descriptor.payload["selected_position"]
                        ),
                        "body_semantic_id": digest,
                        "body_raw_digest": digest,
                        "body_size_bytes": len(encoded),
                        "linkage": descriptor.payload.get("linkage")
                        or descriptor.payload.get("mode_key"),
                    }
                finally:
                    handoff.release(descriptor.estimated_bytes)
                    descriptor.release_owner()
        except BaseException as error:
            first_error.append(error)
            handoff.fail(error)

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    started = time.perf_counter()
    try:
        for ordinal, item in enumerate(payloads):
            descriptor = (
                item
                if isinstance(item, RawCompactDescriptor)
                else _raw_descriptor(_buffer_payload(item), ordinal)
            )
            descriptor.ordinal = ordinal
            results.append(None)
            handoff.put(descriptor)
        handoff.stop(worker_count)
    except BaseException as error:
        first_error.append(error)
        handoff.stop(worker_count)
    for thread in threads:
        thread.join()
    drain_seconds = time.perf_counter() - started
    if first_error or any(item is None for item in results):
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise (
            first_error[0]
            if first_error
            else RuntimeError("pipeline worker dropped a descriptor")
        )
    metadata = [item for item in results if item is not None]
    data = b"".join(_json_bytes(item) + bytes([10]) for item in metadata)
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, metadata_path)
    os.replace(staging, root)
    return {
        "schema_version": "compact_body_store_v1",
        "body_count": len(metadata),
        "metadata_path": str(root / "metadata.jsonl"),
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "body_digests": [item["body_raw_digest"] for item in metadata],
        "worker_count": worker_count,
        "queue_high_water_items": handoff.high_water_items,
        "queue_high_water_bytes": handoff.high_water_bytes,
        "producer_blocked_seconds": handoff.blocked_seconds,
        "oversized_admissions": handoff.oversized_admissions,
        "persistent_bodies_written": sum(item["written"] for item in metrics),
        "persistent_body_bytes": sum(item["bytes"] for item in metrics),
        "body_reread_count": 0,
        "body_rewrite_count": 0,
        "projection_count": 0,
        "python_scalar_array_conversion_count": 0,
        "worker_busy_seconds": sum(item["busy_seconds"] for item in metrics),
        "final_drain_seconds": drain_seconds,
        "encoding_seconds": sum(item["encoding_seconds"] for item in metrics),
        "hashing_seconds": sum(item["hashing_seconds"] for item in metrics),
        "projection_seconds": sum(item["projection_seconds"] for item in metrics),
        "durability_seconds": sum(item["durability_seconds"] for item in metrics),
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
