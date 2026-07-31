"""M7D writer parity for the additive v4 streaming payload path."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from radjax_tome.tome import (
    pack_sharded_tome_v4,
    package_legacy_artifact_as_sharded_tome_v4,
    write_sharded_tome_v4,
    write_sharded_tome_v4_from_legacy_artifact,
)
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract_v2.py"


def _record(index: int) -> dict[str, object]:
    return {
        "selected_example_id": f"example-{index}",
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


def _validate(path: Path) -> dict[str, object]:
    run = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)], capture_output=True, text=True
    )
    return {"returncode": run.returncode, **json.loads(run.stdout)}


def test_v4_writer_emits_a_portably_valid_count_sharded_package(tmp_path: Path) -> None:
    result = write_sharded_tome_v4(
        (_record(index) for index in range(5)),
        tmp_path / "tome",
        training_contract={"target_type": "fixture"},
        authority={"selection": "fixture"},
        payload_records_per_shard=2,
    )
    assert result.selected_count == 5
    shard_entries = [
        json.loads(line)
        for line in (result.root / "selected_exemplars/payload-shards.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [entry["record_count"] for entry in shard_entries] == [2, 2, 1]
    assert _validate(result.root) == {
        "returncode": 0,
        "errors": [],
        "ok": True,
        "warnings": [],
    }


def test_v4_writer_regrouping_changes_layout_not_semantic_identity(
    tmp_path: Path,
) -> None:
    common = {
        "training_contract": {"target_type": "fixture"},
        "authority": {"selection": "fixture"},
    }
    first = write_sharded_tome_v4(
        [_record(index) for index in range(5)],
        tmp_path / "first",
        payload_records_per_shard=2,
        **common,
    )
    second = write_sharded_tome_v4(
        [_record(index) for index in range(5)],
        tmp_path / "second",
        payload_records_per_shard=3,
        **common,
    )
    assert first.semantic_identity_digest == second.semantic_identity_digest
    assert (
        hashlib.sha256(
            (first.root / "selected_exemplars/payload-layout.json").read_bytes()
        ).hexdigest()
        != hashlib.sha256(
            (second.root / "selected_exemplars/payload-layout.json").read_bytes()
        ).hexdigest()
    )


def test_v4_profile_debug_receipts_change_inventory_not_identity(
    tmp_path: Path,
) -> None:
    common = {
        "training_contract": {"target_type": "fixture"},
        "authority": {"selection": "fixture"},
    }
    student = write_sharded_tome_v4(
        [_record(index) for index in range(2)],
        tmp_path / "student",
        profile="student",
        **common,
    )
    debug = write_sharded_tome_v4(
        [_record(index) for index in range(2)],
        tmp_path / "debug",
        profile="full_debug_provenance",
        **common,
    )
    student_header = json.loads(
        (student.root / "manifests/content-manifest-header.json").read_text()
    )
    debug_header = json.loads(
        (debug.root / "manifests/content-manifest-header.json").read_text()
    )
    assert student.semantic_identity_digest == debug.semantic_identity_digest
    assert student_header["inventory_sha256"] != debug_header["inventory_sha256"]
    assert _validate(student.root)["ok"] is True
    assert _validate(debug.root)["ok"] is True


def test_v4_writer_rejects_duplicate_logical_ids_without_publishing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate selected logical identifier"):
        write_sharded_tome_v4(
            [_record(0), _record(0)],
            tmp_path / "tome",
            training_contract={"target_type": "fixture"},
            authority={"selection": "fixture"},
        )
    assert not (tmp_path / "tome").exists()


def test_v4_legacy_adapter_does_not_mutate_its_v3_source(tmp_path: Path) -> None:
    source = build_fake_teacher_textbook_artifact(tmp_path)
    before = {
        path.relative_to(source).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    result = write_sharded_tome_v4_from_legacy_artifact(
        source, tmp_path / "v4", payload_records_per_shard=2
    )
    after = {
        path.relative_to(source).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert _validate(result.root)["ok"] is True


def test_v4_package_adapter_copies_a_complete_profile_without_legacy_payloads(
    tmp_path: Path,
) -> None:
    source = build_fake_teacher_textbook_artifact(tmp_path)
    result = package_legacy_artifact_as_sharded_tome_v4(source, tmp_path / "v4")
    assert (result.root / "metadata.json").is_file()
    assert (
        not (result.root / "cover_page.json").read_bytes()
        == (source / "cover_page.json").read_bytes()
    )
    assert not list(
        (result.root / "selected_exemplars").glob("selected-exemplars-*.json")
    )
    assert not (result.root / "shards").exists()
    assert _validate(result.root)["ok"] is True


def test_v4_package_adapter_projects_legacy_selected_payloads(tmp_path: Path) -> None:
    source = build_fake_teacher_textbook_artifact(tmp_path)
    legacy = source / "selected_exemplars" / "selected-exemplars-00000.json"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_text(
        json.dumps(
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "selected_exemplars": [_record(0), _record(1)],
            }
        )
    )
    result = package_legacy_artifact_as_sharded_tome_v4(
        source, tmp_path / "v4", payload_records_per_shard=1
    )
    assert result.selected_count == 2
    assert (
        len(
            (result.root / "selected_exemplars/payload-shards.jsonl")
            .read_text()
            .splitlines()
        )
        == 2
    )
    assert _validate(result.root)["ok"] is True


def test_v4_transport_is_byte_deterministic(tmp_path: Path) -> None:
    root = write_sharded_tome_v4(
        [_record(index) for index in range(3)],
        tmp_path / "root",
        training_contract={"target_type": "fixture"},
        authority={"selection": "fixture"},
    ).root
    first = pack_sharded_tome_v4(root, tmp_path / "first.tgz")
    second = pack_sharded_tome_v4(root, tmp_path / "second.tgz")
    assert first.read_bytes() == second.read_bytes()
    assert _validate(first)["ok"] is True


def test_v4_archive_validator_rejects_unsafe_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.rtome"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo("../cover_page.json")
        info.size = 2
        output.addfile(info, io.BytesIO(b"{}"))
    assert _validate(archive)["errors"] == ["path_unsafe"]
