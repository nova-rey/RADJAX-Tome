from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from radjax_tome.tome.student_consumption_v2 import (
    ASSIGNMENT_PATH,
    OBSERVED_STATISTICS_PATH,
    TARGET_ROWS_PATH,
    materialize_native_v3_student_consumption_v2,
)


def _write_native_v3_source(root: Path) -> None:
    assignments = root / "corridors" / "mode_assignments"
    assignments.mkdir(parents=True)
    arrays = {
        "position_example_index": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "position": np.asarray([0, 1, 0, 1], dtype=np.int32),
        "mode_id": np.asarray([4, 4, 9, 9], dtype=np.int32),
        "weight": np.asarray([1.0, 0.5, 1.0, 0.75], dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(assignments / f"{name}.npy", values)
    (assignments / "examples_metadata.jsonl").write_text(
        '{"example_id":"example-a","example_index":0}\n'
        '{"example_id":"example-b","example_index":1}\n',
        encoding="utf-8",
    )
    (root / "corridors").mkdir(exist_ok=True)
    (root / "corridors" / "mode_assignments.json").write_text(
        json.dumps(
            {
                "num_examples": 2,
                "arrays": {
                    name: {"path": f"corridors/mode_assignments/{name}.npy"}
                    for name in arrays
                },
                "examples_metadata": {
                    "path": "corridors/mode_assignments/examples_metadata.jsonl"
                },
            }
        ),
        encoding="utf-8",
    )
    shards = root / "shards"
    shards.mkdir()
    np.savez(
        shards / "shard-00000.npz",
        input_ids=np.asarray([[3, 5], [7, 11]], dtype=np.int32),
        attention_mask=np.asarray([[1, 1], [1, 0]], dtype=np.int8),
        corridor_lengths=np.asarray([2, 1], dtype=np.int32),
        score_example_ids=np.asarray(["example-a", "example-b"]),
        corridor_entropy=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        corridor_top1_margin=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        corridor_top8_mass=np.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        corridor_top32_mass=np.asarray([[0.8, 0.85], [0.9, 0.95]], dtype=np.float32),
        corridor_tail_mass=np.asarray([[0.2, 0.15], [0.1, 0.05]], dtype=np.float32),
    )
    (root / "metadata.json").write_text(
        '{"score_pass_authority_hash":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
        encoding="utf-8",
    )
    leaderboards = root / "leaderboards"
    leaderboards.mkdir()
    (leaderboards / "selected_exemplars.json").write_text(
        '{"selected_exemplars":[{"selected_example_id":"example-a","selected_position":0}]}',
        encoding="utf-8",
    )
    payloads = root / "selected_exemplars"
    payloads.mkdir()
    (payloads / "selected-exemplars-00000.json").write_text(
        '{"selected_exemplars":[{"selected_example_id":"example-a","selected_position":0}]}',
        encoding="utf-8",
    )


def test_materializes_explicit_npz_sidecar_without_changing_native_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native-v3"
    _write_native_v3_source(root)
    original_assignment = (root / "corridors" / "mode_assignments.json").read_bytes()

    result = materialize_native_v3_student_consumption_v2(root)

    assert result.example_count == 2
    assert result.assignment_count == 4
    assert result.role_paths["corridor_assignment"] == ASSIGNMENT_PATH
    assert result.role_paths["corridor_observed_statistics"] == OBSERVED_STATISTICS_PATH
    assert result.role_paths["target_shard"] == TARGET_ROWS_PATH
    assert (
        root / "corridors" / "mode_assignments.json"
    ).read_bytes() == original_assignment
    with np.load(root / ASSIGNMENT_PATH, allow_pickle=False) as assignment:
        assert set(assignment.files) == {
            "position_example_index",
            "position",
            "mode_id",
            "weight",
        }
        assert assignment["position"].tolist() == [0, 1, 0, 1]
    with np.load(root / OBSERVED_STATISTICS_PATH, allow_pickle=False) as statistics:
        assert statistics["entropy"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert statistics["tail_mass"].tolist() == pytest.approx([0.2, 0.15, 0.1, 0.05])
    with np.load(root / TARGET_ROWS_PATH, allow_pickle=False) as targets:
        assert targets["input_ids"].tolist() == [[3, 5], [7, 11]]
        assert targets["corridor_lengths"].tolist() == [2, 1]


def test_refuses_missing_source_statistics(tmp_path: Path) -> None:
    root = tmp_path / "native-v3"
    _write_native_v3_source(root)
    source = root / "shards" / "shard-00000.npz"
    with np.load(source, allow_pickle=False) as arrays:
        np.savez(
            source,
            **{
                name: arrays[name]
                for name in arrays.files
                if name != "corridor_entropy"
            },
        )

    try:
        materialize_native_v3_student_consumption_v2(root)
    except ValueError as error:
        assert "corridor_entropy" in str(error)
    else:
        raise AssertionError("missing source semantic evidence must fail closed")


def test_writes_only_package_destination_while_reading_source_shards(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native-v3"
    _write_native_v3_source(source)
    destination = tmp_path / "student-package"
    shutil.copytree(source / "corridors", destination / "corridors")
    shutil.copytree(source / "leaderboards", destination / "leaderboards")
    shutil.copytree(source / "selected_exemplars", destination / "selected_exemplars")

    result = materialize_native_v3_student_consumption_v2(
        source,
        destination_root=destination,
    )

    assert result.root == destination
    assert (destination / ASSIGNMENT_PATH).is_file()
    assert not (source / ASSIGNMENT_PATH).exists()
