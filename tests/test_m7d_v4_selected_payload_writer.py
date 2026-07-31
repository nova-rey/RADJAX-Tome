"""M7D writer parity for the additive v4 streaming payload path."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_tome.tome import write_sharded_tome_v4

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
    layout = json.loads(
        (result.root / "selected_exemplars/payload-layout.json").read_text()
    )
    assert result.selected_count == 5
    assert [entry["record_count"] for entry in layout["shards"]] == [2, 2, 1]
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
