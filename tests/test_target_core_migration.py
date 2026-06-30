from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from radjax_tome.backends import SyntheticTeacherBackend, emit_teacher_target_store
from radjax_tome.targets import (
    OfflineTargetBatch,
    TeacherTargetStore,
    inspect_target_store,
    iter_offline_target_batches,
    iter_target_store_shard_ids,
    load_offline_target_batch,
    run_multishard_target_store_smoke,
)
from radjax_tome.targets.export import export_synthetic_teacher_targets
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {"PYTHONPATH": str(ROOT / "src")}


def test_synthetic_teacher_backend_emits_valid_target_store(tmp_path: Path) -> None:
    backend = SyntheticTeacherBackend(vocab_size=8)

    store = emit_teacher_target_store(
        backend,
        tmp_path / "targets",
        num_examples=2,
        sequence_length=3,
    )

    store.validate()
    assert store.metadata.target_type == "synthetic"
    arrays = store.read_shard(0)
    assert arrays["input_ids"].shape == (2, 3)
    assert arrays["logits"].shape == (2, 3, 8)
    assert float(arrays["logits"][1, 2, 7]) == 2.125


def test_load_offline_target_batch_round_trips_arrays(tmp_path: Path) -> None:
    store = export_synthetic_teacher_targets(
        tmp_path / "targets",
        num_examples=2,
        sequence_length=3,
        vocab_size=7,
    )

    batch = load_offline_target_batch(store)

    assert isinstance(batch, OfflineTargetBatch)
    assert batch.input_ids.dtype == np.int32
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.teacher_logits.shape == (2, 3, 7)


def test_multishard_helpers_visit_each_shard(tmp_path: Path) -> None:
    store = TeacherTargetStore.create(tmp_path / "targets", _metadata())
    store.write_shard(0, _arrays(offset=0))
    store.write_shard(1, _arrays(offset=100))

    assert iter_target_store_shard_ids(store) == (0, 1)
    batches = iter_offline_target_batches(store)
    assert len(batches) == 2
    assert batches[1].input_ids[0, 0] == 100

    result = run_multishard_target_store_smoke(store)
    assert result.status == "pass"
    assert result.examples_seen == 4
    assert "training_ready" in result.claims_not_made


def test_inspect_target_store_reports_shapes_and_keys(tmp_path: Path) -> None:
    export_synthetic_teacher_targets(
        tmp_path / "targets",
        num_examples=2,
        sequence_length=3,
        vocab_size=8,
    )

    summary = inspect_target_store(tmp_path / "targets")

    assert summary["target_type"] == "synthetic"
    assert summary["array_keys"] == ["attention_mask", "input_ids", "logits"]
    assert summary["shards"][0]["arrays"]["logits"]["shape"] == [2, 3, 8]


def test_export_and_inspect_target_clis(tmp_path: Path) -> None:
    export_script = ROOT / "scripts" / "export_teacher_targets.py"
    inspect_script = ROOT / "scripts" / "inspect_targets.py"
    output = tmp_path / "targets"
    report = tmp_path / "report.json"

    export = subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--out",
            str(output),
            "--num-examples",
            "2",
            "--sequence-length",
            "3",
            "--report-json",
            str(report),
        ],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )
    assert export.returncode == 0, export.stderr
    assert "status=pass" in export.stdout
    assert json.loads(report.read_text())["target_type"] == "synthetic"

    inspect = subprocess.run(
        [sys.executable, str(inspect_script), str(output), "--json"],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(inspect.stdout)["shard_count"] == 1


def _metadata() -> TargetStoreMetadata:
    return TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="synthetic-multishard-teacher",
        model_family="synthetic",
        tokenizer_id="smoke-tokenizer",
        tokenizer_hash=None,
        vocab_size=5,
        target_type="synthetic",
        dtype="float32",
        sequence_length=3,
        num_examples=4,
        shard_count=2,
        created_by="test",
        created_at="2026-06-30T00:00:00Z",
        source={"kind": "unit"},
        provenance={"phase": "spec-2.7"},
    )


def _arrays(*, offset: int) -> dict[str, np.ndarray]:
    input_ids = np.arange(offset, offset + 6, dtype=np.int32).reshape(2, 3)
    logits = np.arange(offset, offset + 30, dtype=np.float32).reshape(2, 3, 5)
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones((2, 3), dtype=np.int32),
        "logits": logits / 10.0,
    }
