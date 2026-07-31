"""Tome-facing M7 consumer adapters exercise the Contract portable reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from radjax_tome.tome import (
    open_indexed_student_tome,
    open_streaming_student_tome,
    pack_sharded_tome_v4,
    write_sharded_tome_v4,
)
from tests.test_m7d_v4_selected_payload_writer import _record


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = write_sharded_tome_v4(
        (_record(index) for index in range(3)),
        tmp_path / "directory",
        training_contract={"target_type": "fixture"},
        authority={"selection": "fixture"},
        payload_records_per_shard=2,
    ).root
    return root, pack_sharded_tome_v4(root, tmp_path / "student.tgz")


def test_streaming_student_reader_preserves_order_and_honest_early_close(
    tmp_path: Path,
) -> None:
    _, archive = _artifact(tmp_path)
    with open_streaming_student_tome(archive) as reader:
        assert next(iter(reader))["selected_example_id"] == "example-0"
        assert reader.verification_state == "open"
    assert reader.verification_state == "closed_early"


def test_streaming_student_reader_is_fully_verified_only_when_drained(
    tmp_path: Path,
) -> None:
    _, archive = _artifact(tmp_path)
    with open_streaming_student_tome(archive) as reader:
        assert [record["selected_position"] for record in reader] == [0, 1, 2]
        assert reader.verification_state == "fully_verified"


def test_extracted_indexed_reader_reads_only_the_addressed_record(
    tmp_path: Path,
) -> None:
    root, archive = _artifact(tmp_path)
    reader = open_indexed_student_tome(root)
    assert reader.read(shard_id=1, row=0)["selected_example_id"] == "example-2"
    with pytest.raises(ValueError, match="random_access_transport_unsupported"):
        open_indexed_student_tome(archive)
