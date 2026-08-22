from __future__ import annotations

from pathlib import Path

from radjax_tome.builder.delivery.simple_compact_body import (
    update_compact_linkage,
    write_compact_body_store,
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
