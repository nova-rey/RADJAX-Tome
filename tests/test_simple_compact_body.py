from __future__ import annotations

from pathlib import Path

import radjax_tome.builder.delivery.simple_compact_body as simple_module
from radjax_tome.builder.delivery.simple_compact_body import (
    update_compact_linkage,
    write_compact_body_store,
    write_compact_body_store_from_compact,
    write_compact_body_store_pipelined_from_compact,
)


def _payload():
    return {
        "selected_example_id": "example-1",
        "selected_position": 2,
        "effective_top_k": 2,
        "vocab_size": 8,
        "num_buckets": 2,
        "top_token_ids": [1, 2],
        "top_probs": [0.6, 0.3],
        "top_log_probs": [-0.5, -1.2],
        "top_mass": 0.9,
        "tail_mass": 0.1,
        "bucket_masses": [0.05, 0.05],
    }


def test_body_store_writes_compact_body_and_metadata_only(tmp_path: Path) -> None:
    result = write_compact_body_store(tmp_path, [_payload()])
    body = next((tmp_path / "bodies").glob("*.body"))
    before = body.read_bytes()
    assert result["body_count"] == 1
    assert body.stat().st_size > 0
    assert (
        update_compact_linkage(tmp_path, {("example-1", 2): {"corridor": "global"}})
        == 1
    )
    assert body.read_bytes() == before


def test_explicit_compact_boundary_does_not_project_again(
    tmp_path: Path, monkeypatch
) -> None:
    compact = simple_module.compact_payload_for_storage(_payload())
    original = simple_module.compact_payload_for_storage
    calls = 0

    def counted(payload):
        nonlocal calls
        calls += 1
        return original(payload)

    monkeypatch.setattr(simple_module, "compact_payload_for_storage", counted)
    write_compact_body_store_from_compact(tmp_path, [compact])
    assert calls == 0


def test_compact_linkage_has_no_payload_or_body_work(tmp_path: Path) -> None:
    write_compact_body_store(tmp_path, [_payload()])
    counters: dict[str, int] = {}
    assert (
        update_compact_linkage(
            tmp_path, {("example-1", 2): {"corridor": "global"}}, counters=counters
        )
        == 1
    )
    assert counters["source_payload_reads"] == 0
    assert counters["body_reads"] == 0
    assert counters["body_hashes"] == 0
    assert counters["body_rewrites"] == 0
    assert counters["metadata_reads"] == 1
    assert counters["metadata_writes"] == 1


def test_pipelined_workers_are_deterministic(tmp_path: Path) -> None:
    compact = simple_module.compact_payload_for_storage(_payload())
    digests = []
    for workers in (1, 2, 4):
        root = tmp_path / f"w{workers}"
        result = write_compact_body_store_pipelined_from_compact(
            root, [compact], worker_count=workers
        )
        digests.append(
            (
                result["metadata_sha256"],
                sorted(path.read_bytes() for path in (root / "bodies").glob("*.body")),
                (root / "metadata.jsonl").read_bytes(),
            )
        )
        assert result["projection_count"] == 0
        assert result["body_reread_count"] == 0
        assert result["body_rewrite_count"] == 0
        assert result["queue_high_water_bytes"] <= 16 * 1024 * 1024
    assert digests[0] == digests[1] == digests[2]
