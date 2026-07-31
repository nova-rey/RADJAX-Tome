"""Regression characterizations rejected by the independent M7 review.

These tests intentionally lock the former additive-v4 gaps before the
corrective production and consumer work changes the paved road.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.tome.payload_sharding_v4 import (
    pack_sharded_tome_v4,
    write_sharded_tome_v4,
)

from tests.test_m7d_v4_selected_payload_writer import _record


def _write(tmp_path: Path) -> Path:
    return write_sharded_tome_v4(
        (_record(index) for index in range(2)),
        tmp_path / "directory",
        training_contract={"target_type": "fixture"},
        authority={"selection": "fixture"},
        payload_records_per_shard=1,
    ).root


def test_writer_rejects_missing_required_semantic_field_before_publication(
    tmp_path: Path,
) -> None:
    records = [_record(0)]
    records[0].pop("selected_board")

    with pytest.raises(ValueError, match="required semantic"):
        write_sharded_tome_v4(
            records,
            tmp_path / "invalid",
            training_contract={"target_type": "fixture"},
            authority={"selection": "fixture"},
        )

    assert not (tmp_path / "invalid").exists()


def test_archive_transport_declaration_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _write(tmp_path)
    archive = pack_sharded_tome_v4(root, tmp_path / "tome.tgz")

    from radjax_contract.tome import validate_streaming_tome

    result = validate_streaming_tome(archive)
    assert not result.ok
    assert "transport_mismatch" in result.errors


def test_packaged_archive_declares_its_actual_transport(tmp_path: Path) -> None:
    root = _write(tmp_path)
    archive = pack_sharded_tome_v4(root, tmp_path / "tome.tgz")

    import tarfile

    with tarfile.open(archive, "r:gz") as handle:
        cover = json.loads(handle.extractfile("cover_page.json").read())  # type: ignore[union-attr]
    assert cover["package"]["transport"] == "tgz"
