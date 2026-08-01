"""Regression characterizations rejected by the independent M7 review.

These tests intentionally lock the former additive-v4 gaps before the
corrective production and consumer work changes the paved road.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from radjax_tome.tome import payload_sharding_v4
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


def test_writer_rejects_contract_invalid_semantic_type_before_publication(
    tmp_path: Path,
) -> None:
    """Presence-only checks cannot substitute for Contract schema validation."""
    records = [_record(0)]
    records[0]["selected_position"] = "not-an-int"

    with pytest.raises(ValueError, match="payload_semantic_projection_invalid"):
        write_sharded_tome_v4(
            records,
            tmp_path / "invalid-type",
            training_contract={"target_type": "fixture"},
            authority={"selection": "fixture"},
        )

    assert not (tmp_path / "invalid-type").exists()


def test_archive_transport_declaration_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _write(tmp_path)
    archive = pack_sharded_tome_v4(root, tmp_path / "tome.tgz")
    mismatched = tmp_path / "mismatched.tgz"

    with (
        tarfile.open(archive, "r:gz") as source,
        tarfile.open(mismatched, "w:gz", format=tarfile.USTAR_FORMAT) as destination,
    ):
        for member in source:
            contents = source.extractfile(member)
            if member.name == "cover_page.json":
                cover = json.loads(contents.read())  # type: ignore[union-attr]
                cover["package"]["transport"] = "directory"
                encoded = json.dumps(
                    cover, sort_keys=True, separators=(",", ":")
                ).encode()
                replacement = tarfile.TarInfo(member.name)
                replacement.size = len(encoded)
                replacement.mtime = member.mtime
                replacement.uid = member.uid
                replacement.gid = member.gid
                replacement.mode = member.mode
                destination.addfile(replacement, io.BytesIO(encoded))
            else:
                destination.addfile(member, contents)

    from radjax_contract.tome import validate_streaming_tome

    result = validate_streaming_tome(mismatched)
    assert not result.ok
    assert "transport_mismatch" in result.errors


def test_packaged_archive_declares_its_actual_transport(tmp_path: Path) -> None:
    root = _write(tmp_path)
    archive = pack_sharded_tome_v4(root, tmp_path / "tome.tgz")

    with tarfile.open(archive, "r:gz") as handle:
        cover = json.loads(handle.extractfile("cover_page.json").read())  # type: ignore[union-attr]
    assert cover["package"]["transport"] == "tgz"


def test_failed_archive_pack_never_exposes_a_partial_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write(tmp_path)
    archive = tmp_path / "interrupted.tgz"

    def interrupted_write(_: Path, temporary: Path, *, compression: str) -> None:
        temporary.write_bytes(b"partial archive bytes")
        raise OSError("simulated interruption")

    monkeypatch.setattr(payload_sharding_v4, "_write_archive", interrupted_write)

    with pytest.raises(OSError, match="simulated interruption"):
        pack_sharded_tome_v4(root, archive)

    assert not archive.exists()
    assert not list(tmp_path.glob(".interrupted.tgz.*"))


def test_archive_validation_failure_never_promotes_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write(tmp_path)
    archive = tmp_path / "invalid.tgz"
    monkeypatch.setattr(
        payload_sharding_v4,
        "_archive_cover_bytes",
        lambda _source, *, compression: b"{}",
    )

    with pytest.raises(ValueError, match="Contract validation"):
        pack_sharded_tome_v4(root, archive)

    assert not archive.exists()
    assert not list(tmp_path.glob(".invalid.tgz.*"))
