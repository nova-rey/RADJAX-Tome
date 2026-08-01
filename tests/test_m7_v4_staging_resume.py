from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.builder.delivery.staging import (
    complete_v4_shard_staging,
    prepare_v4_shard_staging,
    seal_v4_shard,
)


def _config() -> dict[str, object]:
    return {"authority": "sha256:abc", "selected_count": 5}


def test_v4_staging_seals_contiguous_prefix_and_reuses_it(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    state = prepare_v4_shard_staging(
        stage, config=_config(), payload_records_per_shard=2
    )
    state = seal_v4_shard(state, (b'{"selection_index":0}', b'{"selection_index":1}'))
    resumed = prepare_v4_shard_staging(
        stage, config=_config(), payload_records_per_shard=2
    )

    assert resumed.completed_record_count == 2
    assert resumed.next_shard_id == 1
    assert resumed.sealed_shards[0].first_selection_index == 0
    assert (stage / resumed.sealed_shards[0].path).is_file()


def test_v4_staging_discards_interrupted_temporary_shard(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "shards").mkdir(parents=True)
    temporary = stage / "shards" / "shard-00000.jsonl.tmp"
    temporary.write_bytes(b'{"selection_index":0}\n')

    state = prepare_v4_shard_staging(
        stage, config=_config(), payload_records_per_shard=2
    )

    assert state.sealed_shards == ()
    assert not temporary.exists()


def test_v4_staging_rejects_config_drift_and_unreceipted_shards(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    state = prepare_v4_shard_staging(
        stage, config=_config(), payload_records_per_shard=2
    )
    seal_v4_shard(state, (b'{"selection_index":0}',))

    with pytest.raises(ValueError, match="configuration mismatch"):
        prepare_v4_shard_staging(
            stage,
            config={"authority": "sha256:different", "selected_count": 5},
            payload_records_per_shard=2,
        )
    (stage / "shards" / "shard-00001.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="unreceipted"):
        prepare_v4_shard_staging(stage, config=_config(), payload_records_per_shard=2)


def test_v4_staging_rejects_receipt_gap_and_seals_completion(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    state = prepare_v4_shard_staging(
        stage, config=_config(), payload_records_per_shard=2
    )
    state = seal_v4_shard(state, (b'{"selection_index":0}', b'{"selection_index":1}'))
    complete_v4_shard_staging(state, expected_record_count=2)
    assert (stage / "v4-shard-transaction-complete.json").is_file()

    receipt_path = stage / "v4-shard-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sealed_shards"][0]["shard_id"] = 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="gap or reorder"):
        prepare_v4_shard_staging(stage, config=_config(), payload_records_per_shard=2)


def test_v4_staging_refuses_finalization_of_incomplete_prefix(tmp_path: Path) -> None:
    state = prepare_v4_shard_staging(
        tmp_path / "stage", config=_config(), payload_records_per_shard=2
    )
    state = seal_v4_shard(state, (b'{"selection_index":0}',))

    with pytest.raises(ValueError, match="transaction incomplete"):
        complete_v4_shard_staging(state, expected_record_count=2)
