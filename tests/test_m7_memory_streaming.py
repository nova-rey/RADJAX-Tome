"""Enforce the M7 direct-streaming memory and termination contract.

The records deliberately contain incompressible-ish payload text.  That makes
the archive large enough to distinguish a sequential reader from an
implementation that first drains it into an in-memory payload collection.
These tests compare two total payload sizes while holding the largest encoded
record fixed; the acceptance bound is consequently about one record plus
fixed parser/I/O buffers, never the total archive or configured shard size.
"""

from __future__ import annotations

import gc
import os
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from radjax_tome.tome import (
    open_streaming_student_tome,
    pack_sharded_tome_v4,
    write_sharded_tome_v4,
)


def _record(index: int) -> dict[str, Any]:
    """Minimal complete v4 semantic record; kept local for standalone tests."""
    return {
        "selected_example_id": f"memory-example-{index}",
        "selected_position": index,
        "selected_score": 1.0,
        "score_selected_position_entropy": 1.0,
        "score_top_token_id": 3,
        "source_shard_id": 0,
        "source_row": index,
        "source_position": index,
        "source_score": 1.0,
        "source_top_token_id": 3,
        "source_score_policy": "fixture",
        "payload_ref": {"fixture": "payload"},
        "selected_policy": "fixture",
        "source_delivery_path": "fixture",
        "top_token_ids": [3],
        "top_log_probs": [-0.1],
        "top_probs": [0.9],
        "top_selection_mask": [True],
        "effective_top_k": 1,
        "top_mass": 0.9,
        "tail_mass": 0.1,
        "bucket_masses": [0.1],
        "teacher_entropy": 1.0,
        "sequence_length": 8,
        "vocab_size": 16,
        "num_buckets": 1,
        "dynamic_top_k": False,
        "dynamic_mass_threshold": 0.9,
        "dynamic_top_k_max": 1,
        "top_k_saturated": False,
        "long_tail_class": "fixture",
        "long_tail_warnings": [],
        "effective_top_k_fraction_of_vocab": 0.0625,
        "semantic_tail_tag": "fixture",
        "selected_board": "primary",
        "corridor_mode_id": 0,
        "corridor_fingerprint_id": 0,
        "corridor_assignment_status": "fixture",
    }


def _large_record(index: int, payload_bytes: int) -> dict[str, Any]:
    """Return a valid semantic record with a fixed maximum encoded size."""
    record = _record(index)
    # Hex stays JSON-safe and avoids a highly-compressible repeated character
    # fixture.  Its contents are intentionally irrelevant to the assertion:
    # only the encoded record maximum is held constant across both packages.
    record["payload_ref"] = {"bytes": os.urandom(payload_bytes).hex()}
    return record


def _archive(
    tmp_path: Path, *, count: int, payload_bytes: int, name: str
) -> tuple[Path, int]:
    root = write_sharded_tome_v4(
        (_large_record(index, payload_bytes) for index in range(count)),
        tmp_path / name,
        training_contract={"target_type": "memory-fixture"},
        authority={"selection": "memory-fixture"},
        payload_records_per_shard=4,
    ).root
    archive = pack_sharded_tome_v4(root, tmp_path / f"{name}.tgz")
    encoded_total = sum(
        path.stat().st_size
        for path in (root / "selected_exemplars/shards").glob("*.jsonl")
    )
    return archive, encoded_total


def _peak_python_allocation(archive: Path) -> tuple[int, int]:
    # Initialize immutable parser/codec tables outside the measurement.  The
    # contract is about retained payload memory, not one-time import setup.
    with open_streaming_student_tome(archive) as warm_reader:
        for _ in warm_reader:
            pass
    gc.collect()
    tracemalloc.start()
    try:
        with open_streaming_student_tome(archive) as reader:
            count = sum(1 for _ in reader)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return count, peak


def test_direct_archive_streaming_peak_does_not_scale_with_total_payload(
    tmp_path: Path,
) -> None:
    """A larger total payload must not induce a whole-payload allocation."""
    # 32 KiB raw entropy becomes a ~64 KiB JSON string.  The large case is
    # four times as many records with precisely the same maximum record size.
    small, small_total = _archive(
        tmp_path, count=8, payload_bytes=32 * 1024, name="small"
    )
    large, large_total = _archive(
        tmp_path, count=64, payload_bytes=32 * 1024, name="large"
    )

    small_count, small_peak = _peak_python_allocation(small)
    large_count, large_peak = _peak_python_allocation(large)
    assert (small_count, large_count) == (8, 64)
    assert large_total >= small_total * 7
    # This is intentionally a generous fixed envelope: JSON decoding and
    # gzip/tar buffers may retain more than one Python allocation, but it is
    # still materially below a 2 MiB payload and cannot grow with record count.
    fixed_envelope = 1_250_000
    assert small_peak < fixed_envelope
    assert large_peak < fixed_envelope
    assert large_peak <= small_peak + 600_000
    assert large_peak * 2 < large_total


def test_direct_archive_yields_before_complete_logical_payload_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Track compressed input reads to reject hidden full-archive spooling."""
    archive, _ = _archive(
        tmp_path, count=48, payload_bytes=24 * 1024, name="sequential"
    )
    import radjax_contract.tome.streaming_validation as streaming_validation

    original_open = streaming_validation.tarfile.open
    observed: dict[str, int] = {"bytes": 0}

    class _TrackingFile:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def read(self, size: int = -1) -> bytes:
            # The production reader must not ask the input for an unbounded
            # full-file read.  gzip/tar may choose bounded internal reads.
            assert size != -1
            block = self._handle.read(size)
            observed["bytes"] += len(block)
            return block

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def _tracked_open(
        name: str | Path | None = None,
        mode: str = "r",
        fileobj: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if Path(name) == archive and mode == "r|*" and fileobj is None:
            return original_open(
                name=None,
                mode=mode,
                fileobj=_TrackingFile(archive.open("rb")),
                **kwargs,
            )
        return original_open(name=name, mode=mode, fileobj=fileobj, **kwargs)

    monkeypatch.setattr(streaming_validation.tarfile, "open", _tracked_open)
    with open_streaming_student_tome(archive) as reader:
        first = next(iter(reader))
        assert first["selected_position"] == 0
        bytes_at_first_yield = observed["bytes"]
        assert reader.verification_state == "open"
        # Deliberately do not drain.  Closing early must not manufacture a
        # fully verified package by reading the remainder in __exit__.
    assert reader.verification_state == "closed_early"
    assert bytes_at_first_yield < archive.stat().st_size * 0.80
    assert observed["bytes"] == bytes_at_first_yield
